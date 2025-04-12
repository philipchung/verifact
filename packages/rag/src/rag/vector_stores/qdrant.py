"""
LlamaIndex Qdrant Vector Store Index wrapper.

An index that is built on top of an existing Qdrant collection, that is customized
for multiple dense and sparse vectors. At the current moment, Qdrant does not
support colbert vectors, so this LlamaIndex wrapper does not support it either.

This is heavily modified from llama-index qdrant vectorstore:
https://github.com/run-llama/llama_index/blob/main/llama_index/vector_stores/qdrant.py
"""

import asyncio
import logging
import os
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Any, cast

import numpy as np
import qdrant_client
from grpc import RpcError
from llama_index.core.bridge.pydantic import Field, PrivateAttr
from llama_index.core.utils import iter_batch
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from qdrant_client import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Filter
from tqdm.asyncio import tqdm
from utils import flatten_list_of_list, run_jobs

from rag.embedding import sparse_embedding_dict_to_index_vector_lists
from rag.embedding.m3_embedding import M3Embedding
from rag.embedding.utils import NamedEmbedding
from rag.fusion import FusionCallable, distribution_based_score_fusion
from rag.retry import create_retry_decorator
from rag.schema import BaseNode, EmbeddingItem, NodeWithScore
from rag.vector_stores.payload_utils import node_to_qdrant_payload_dict
from rag.vector_stores.query_utils import qdrant_query_response_to_node_with_score

logger = logging.getLogger(__name__)
import_err_msg = "`qdrant-client` package not found, please run `pip install qdrant-client`"


QueryEncoderCallable = Callable[[list[str]], tuple[list[list[int]], list[list[float]]]]


qdrant_retry_decorator = create_retry_decorator(
    max_retries=10,
    random_exponential=True,
    stop_after_delay_seconds=60,
    min_seconds=4,
    max_seconds=60,
)


def default_bge_m3_encoder(
    embed_model: M3Embedding | None = None,
    embed_batch_size: int = 10,
    num_workers: int = 12,
    return_dense: bool = True,
    return_sparse: bool = True,
    return_colbert: bool = False,
    dense_name: str = "dense",
    sparse_name: str = "sparse",
    colbert_name: str = "colbert",
    default_vector_name: str = "dense",
    timeout: float | None = None,
) -> QueryEncoderCallable:
    """Returns a callable function that computes embeddings using BGE-M3 Flag Embedding."""
    # Configure a new embedding model with default settings if existing embed_model not passed in
    if embed_model is None:
        embed_model = M3Embedding(
            model_name=os.environ["EMBED_MODEL_NAME"],
            api_base=os.environ["EMBED_URL_BASE"],
            embed_batch_size=embed_batch_size,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert=return_colbert,
            dense_name=dense_name,
            sparse_name=sparse_name,
            colbert_name=colbert_name,
            default_vector_name=default_vector_name,
            num_workers=num_workers,
            timeout=timeout,
        )

    def compute_vectors(texts: list[str]) -> NamedEmbedding:
        """Computes vectors from text, formats and returns them."""
        return embed_model.get_text_embeddings(texts)

    return compute_vectors


class QdrantVectorStore(BasePydanticVectorStore):
    """
    Qdrant Vector Store.

    In this vector store, embeddings and docs are stored within a
    Qdrant collection.

    During query time, the index uses Qdrant to query for the top
    k most similar nodes.

    Args:
        collection_name: (str): name of the Qdrant collection
        client (Optional[Any]): QdrantClient instance from `qdrant-client` package
    """

    stores_text: bool = True
    flat_metadata: bool = False

    collection_name: str
    url: str | None
    api_key: str | None
    upsert_batch_size: int
    query_batch_size: int
    num_query_workers: int
    client_kwargs: dict = Field(default_factory=dict)
    on_disk_payload: bool
    enable_hybrid: bool
    enable_dense: bool
    enable_sparse: bool
    enable_colbert: bool

    _client: qdrant_client.QdrantClient = PrivateAttr()
    _aclient: qdrant_client.AsyncQdrantClient = PrivateAttr()
    _collection_initialized: bool = PrivateAttr()
    _query_encoder_fn: QueryEncoderCallable | None = PrivateAttr()
    _fusion_fn: FusionCallable | None = PrivateAttr()

    def __init__(
        self,
        collection_name: str,
        client: Any | None = None,
        aclient: Any | None = None,
        url: str | None = None,
        api_key: str | None = None,
        upsert_batch_size: int = 100,
        query_batch_size: int = 10,
        num_query_workers: int = 16,
        client_kwargs: dict | None = None,
        on_disk_payload: bool | None = True,
        enable_hybrid: bool = True,
        enable_dense: bool = True,
        enable_sparse: bool = True,
        enable_colbert: bool = False,
        query_encoder_fn: QueryEncoderCallable | None = None,
        fusion_fn: FusionCallable | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Qdrant Vector Database Client, Embedding Encoder, Fusion Function."""

        super().__init__(
            collection_name=collection_name,
            url=url,
            api_key=api_key,
            upsert_batch_size=upsert_batch_size,
            query_batch_size=query_batch_size,
            num_query_workers=num_query_workers,
            client_kwargs=client_kwargs or {},
            on_disk_payload=on_disk_payload,
            enable_hybrid=enable_hybrid,
            enable_dense=enable_dense,
            enable_sparse=enable_sparse,
            enable_colbert=enable_colbert,
        )
        # Initialize private attributes
        if client is None and aclient is None:
            if url is None or api_key is None:
                raise ValueError(
                    "Must provide either a QdrantClient instance or a url and api_key."
                )
            client_kwargs = client_kwargs or {}
            self._client = qdrant_client.QdrantClient(url=url, api_key=api_key, **client_kwargs)
            self._aclient = qdrant_client.AsyncQdrantClient(
                url=url, api_key=api_key, **client_kwargs
            )
        else:
            if client is not None and aclient is not None:
                logger.info(
                    "Both client and aclient are provided. If using `:memory:` "
                    "mode, the data between clients is not synced."
                )
            self._client = client
            self._aclient = aclient

        # Init sync client; Lazy init for async clients
        self._collection_initialized = self._client is not None and self._collection_exists(
            collection_name
        )

        # Sanity Check Hybrid Search (for now, colbert does not count and ignored)
        if int(enable_dense) + int(enable_sparse) > 1:
            enable_hybrid = True

        # Setup hybrid search if enabled
        if enable_hybrid:
            # Query Encoder Fn is a function that returns both sparse and dense
            # encoder representations as dict[str, Any]
            self._query_encoder_fn = query_encoder_fn or default_bge_m3_encoder()
            default_fusion_fn = partial(distribution_based_score_fusion, save_score=True)
            self._fusion_fn = fusion_fn or cast(FusionCallable, default_fusion_fn)

    @classmethod
    def class_name(cls) -> str:
        return "QdrantVectorStore"

    @property
    def client(self) -> Any:
        """Return the Qdrant client."""
        return self._client

    def _build_points(self, nodes: list[BaseNode]) -> tuple[list[Any], list[str]]:
        """Format llama-index node into Qdrant points data format.
        Assumes `nodes` is a list of modified llama-index nodes with default embedding
        stored in attribute `node.embedding`, but also has an attribute `node.embeddings`
        which is a dict with multiple embeddings. The value for each entry in the dict
        is an EmbeddingItem which stores the embedding, embedding name, and embedding kind
        (e.g. "sparse", "dense", etc.)
        """
        # Iterate through all nodes in batches
        ids = []
        points = []
        for node_batch in iter_batch(nodes, self.upsert_batch_size):
            # Prepare batch vectors and payloads
            node_ids = []
            vectors: list[Any] = []
            payloads = []
            for _, node in enumerate(node_batch):
                assert isinstance(node, BaseNode)
                ## Create ID for Qdrant Point
                node_ids.append(node.node_id)

                ## Create Vector for Qdrant Point
                # Unpack named embeddings from node and format into Qdrant vectors
                named_embeddings: dict[str, EmbeddingItem] = node.get_embeddings()
                vector_dict: dict[str, Any] = {}
                for key, embedding_item in named_embeddings.items():
                    vector_name = embedding_item.name if embedding_item.name is not None else key
                    embedding = embedding_item.embedding
                    vector_kind = embedding_item.kind

                    if self.enable_dense and vector_kind == "dense":
                        # Ensure dense embeddings converted to python list of list
                        if isinstance(embedding, np.ndarray):
                            embedding = embedding.astype(float).tolist()
                        vector_dict |= {vector_name: embedding}
                    if self.enable_sparse and vector_kind == "sparse":
                        # Sparse embeddings are key-val dict; convert to index and vector lists
                        sparse_indices, sparse_vectors = (
                            sparse_embedding_dict_to_index_vector_lists(embedding)
                        )
                        vector_dict |= {
                            vector_name: rest.SparseVector(
                                indices=sparse_indices,
                                values=sparse_vectors,
                            )
                        }
                    if self.enable_colbert and vector_kind == "colbert":
                        # NOTE: not yet supported by qdrant
                        vector_dict |= {vector_name: embedding}
                vectors.append(vector_dict)

                # Create Payload for Qdrant Point
                payload = node_to_qdrant_payload_dict(node)
                payloads.append(payload)

            # Format Vector and Payload into Qdrant Points
            points.extend(
                [
                    rest.PointStruct(id=node_id, payload=payload, vector=vector)
                    for node_id, payload, vector in zip(node_ids, payloads, vectors, strict=False)
                ]
            )
            ids.extend(node_ids)

        return points, ids

    @qdrant_retry_decorator
    def add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        """
        Add nodes to index.

        Args:
            nodes: List[BaseNode]: list of nodes with embeddings

        Returns:
            List of node IDs that were added to the index.

        Raises:
            ValueError: If trying to using async methods without aclient
        """
        collection_initialized = self._collection_exists(self.collection_name)

        # If collection not initialized, create vector config from first node
        if len(nodes) > 0 and not collection_initialized:
            embeddings: dict[str, EmbeddingItem] = nodes[0].get_embeddings()
            self._create_collection_from_embeddings(
                embeddings, collection_name=self.collection_name
            )

        # Build Qdrant Points from LlamaIndex nodes
        points, ids = self._build_points(nodes)

        # Batch upsert the points into Qdrant collection to avoid large payloads
        for points_batch in iter_batch(points, self.upsert_batch_size):
            try:
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=points_batch,
                )
            except Exception as ex:
                raise ValueError(
                    f"Error with collection {self.collection_name} in "
                    f"upserting points: {points_batch}. "
                    f"Exception: {ex}"
                ) from ex

        return ids

    @qdrant_retry_decorator
    async def async_add(self, nodes: list[BaseNode], **kwargs: Any) -> Awaitable[list[str]]:
        """
        Asynchronous method to add nodes to Qdrant index.

        Args:
            nodes: List[BaseNode]: List of nodes with embeddings.

        Returns:
            List of node IDs that were added to the index.

        Raises:
            ValueError: If trying to using async methods without aclient
        """
        collection_initialized = await self._acollection_exists(self.collection_name)

        # If collection not initialized, create vector config from first node
        if len(nodes) > 0 and not collection_initialized:
            embeddings: dict[str, EmbeddingItem] = nodes[0].get_embeddings()
            await self._acreate_collection_from_embeddings(
                embeddings, collection_name=self.collection_name
            )

        # Build Qdrant Points from LlamaIndex nodes
        points, ids = self._build_points(nodes)

        # Batch upsert the points into Qdrant collection to avoid large payloads
        for points_batch in iter_batch(points, self.upsert_batch_size):
            try:
                await self._aclient.upsert(
                    collection_name=self.collection_name,
                    points=points_batch,
                )
            except Exception as ex:
                raise ValueError(
                    f"Error with collection {self.collection_name} in "
                    f"upserting points: {points_batch}"
                    f"Exception: {ex}"
                ) from ex

        return ids

    @qdrant_retry_decorator
    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        if isinstance(ref_doc_id, bytes):
            ref_doc_id = ref_doc_id.decode("utf-8")
        self._client.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=ref_doc_id))]
            ),
        )

    @qdrant_retry_decorator
    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Asynchronous method to delete nodes using with ref_doc_id.

        Args:
            ref_doc_id (str): The doc_id of the document to delete.

        """
        if isinstance(ref_doc_id, bytes):
            ref_doc_id = ref_doc_id.decode("utf-8")
        await self._aclient.delete(
            collection_name=self.collection_name,
            points_selector=rest.Filter(
                must=[rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=ref_doc_id))]
            ),
        )

    @qdrant_retry_decorator
    def _create_collection_from_embeddings(
        self,
        embeddings: dict[str, EmbeddingItem],
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Create Qdrant Collection with vector config inferred from embeddings.

        Args:
            collection_name (str): Name of collection. If not provided, will use the
                default collection name provided during vectorstore object construction.
            embeddings (dict[str, Any]): dictionary of named embeddings, may be mix of
                dense and sparse embeddings. Dense embedding keys should end with `-dense`
                and sparse embedding keys should end with `-sparse`.
        """
        # Create Vector Config
        dense_vectors_config = {}
        sparse_vectors_config = {}
        for key, embedding_item in embeddings.items():
            vector_name = embedding_item.name if embedding_item.name is not None else key
            embedding = embedding_item.embedding
            vector_kind = embedding_item.kind

            if self.enable_dense and vector_kind == "dense":
                dense_vectors_config[vector_name] = {
                    "vector_size": len(embedding),
                    "distance": rest.Distance.COSINE,
                }
            if self.enable_sparse and vector_kind == "sparse":
                sparse_vectors_config[vector_name] = {}

        # Lazily Create Qdrant Collection
        collection_name = collection_name or self.collection_name
        self._create_collection(
            collection_name=collection_name,
            dense_vectors_config=dense_vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        # Poll Qdrant to Ensure Collection Initialized before Existing Function
        while not self._collection_exists(self.collection_name):
            time.sleep(1)

    @qdrant_retry_decorator
    async def _acreate_collection_from_embeddings(
        self,
        embeddings: dict[str, EmbeddingItem],
        collection_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Create Vector Config
        dense_vectors_config = {}
        sparse_vectors_config = {}
        for key, embedding_item in embeddings.items():
            vector_name = embedding_item.name if embedding_item.name is not None else key
            embedding = embedding_item.embedding
            vector_kind = embedding_item.kind

            if self.enable_dense and vector_kind == "dense":
                dense_vectors_config[vector_name] = {
                    "vector_size": len(embedding),
                    "distance": rest.Distance.COSINE,
                }
            if self.enable_sparse and vector_kind == "sparse":
                sparse_vectors_config[vector_name] = {}

        # Lazily Create Qdrant Collection
        collection_name = collection_name or self.collection_name

        async def create_collection(flag: asyncio.Event) -> None:
            "Create collection and signal completion."
            await self._acreate_collection(
                collection_name=collection_name,
                dense_vectors_config=dense_vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
            flag.set()

        async def wait_for_create_collection_to_complete() -> None:
            "Asynchronously blocks until collection is created."
            flag = asyncio.Event()
            asyncio.create_task(create_collection(flag))
            await flag.wait()

        await wait_for_create_collection_to_complete()

    @qdrant_retry_decorator
    def _create_collection(
        self,
        collection_name: str,
        dense_vectors_config: dict[str, Any],
        sparse_vectors_config: dict[str, Any],
    ) -> None:
        """Create Qdrant Collection

        Args:
            collection_name (str): Name of collection.
            vectors_config (dict[str, Any]): dictionary that defines database schema
                for all dense vectors in Qdrant collection. Example:
                {
                    "dense": {vector_size: 1024, distance: "Dot"},
                    ...
                }
            sparse_vectors_config (dict[str, Any]): dictionary that defines database schema
                for all sparse vectors in Qdrant collection. Example:
                {
                    "sparse": {},
                    ...
                }
        """
        # Reformat vectors config to Qdrant VectorParams
        dense_vectors_config = {
            k: rest.VectorParams(
                size=v.get("vector_size"),
                distance=v.get("distance", rest.Distance.COSINE),
            )
            for k, v in dense_vectors_config.items()
        }
        sparse_vectors_config = {
            k: rest.SparseVectorParams(index=rest.SparseIndexParams())
            for k, v in sparse_vectors_config.items()
            if k.split("-")[-1].lower() == "sparse"
        }

        try:
            ## Create Qdrant Collection
            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=dense_vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                on_disk_payload=self.on_disk_payload,
            )
            ## Configure Payload Indices
            # (https://qdrant.tech/documentation/concepts/indexing/#payload-index)
            # Index `text`` field for full-text search
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="text",
                field_schema=rest.TextIndexParams(
                    type="text",
                    tokenizer=rest.TokenizerType.PREFIX,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )
            # Index properties that source file was derived from
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="file_path",
                field_schema=rest.TextIndexParams(
                    type="text",
                    tokenizer=rest.TokenizerType.PREFIX,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="file_name",
                field_schema=rest.TextIndexParams(
                    type="text",
                    tokenizer=rest.TokenizerType.PREFIX,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )
            # Index derived node metadata
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="created_at",
                field_schema="datetime",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="node_kind",
                field_schema="keyword",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="node_monotonic_idx",
                field_schema="integer",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="num_tokens",
                field_schema="integer",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="num_characters",
                field_schema="integer",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="start_char_idx",
                field_schema="integer",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="end_char_idx",
                field_schema="integer",
            )
            # Index fields specific to MIMIC-III data
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="ROW_ID",
                field_schema="integer",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="SUBJECT_ID",
                field_schema="integer",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="HADM_ID",
                field_schema="integer",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="CHARTDATE",
                field_schema="datetime",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="CHARTTIME",
                field_schema="datetime",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="STORETIME",
                field_schema="datetime",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="CATEGORY",
                field_schema="keyword",
            )
            self._client.create_payload_index(
                collection_name=collection_name,
                field_name="DESCRIPTION",
                field_schema=rest.TextIndexParams(
                    type="text",
                    tokenizer=rest.TokenizerType.PREFIX,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )

        except (ValueError, UnexpectedResponse) as exc:
            if "already exists" not in str(exc):
                raise exc  # noqa: TRY201
            logger.info(
                "Collection %s already exists, skipping collection creation.",
                collection_name,
            )
        self._collection_initialized = True

    @qdrant_retry_decorator
    async def _acreate_collection(
        self,
        collection_name: str,
        dense_vectors_config: dict[str, Any],
        sparse_vectors_config: dict[str, Any],
    ) -> None:
        """Asynchronous Method to Create Qdrant Collection

        Args:
            collection_name (str): Name of collection.
            vectors_config (dict[str, Any]): dictionary that defines database schema
                for all dense vectors in Qdrant collection. Example:
                {
                    "dense": {vector_size: 1024, distance: "Dot"},
                    ...
                }
            sparse_vectors_config (dict[str, Any]): dictionary that defines database schema
                for all sparse vectors in Qdrant collection. Example:
                {
                    "sparse": {},
                    ...
                }
        """
        # Reformat vectors config to Qdrant VectorParams
        dense_vectors_config = {
            k: rest.VectorParams(
                size=v.get("vector_size"),
                distance=v.get("distance", rest.Distance.COSINE),
            )
            for k, v in dense_vectors_config.items()
        }

        sparse_vectors_config = {
            k: rest.SparseVectorParams(index=rest.SparseIndexParams())
            for k, v in sparse_vectors_config.items()
        }

        try:
            ## Create Qdrant Collection
            await self._aclient.create_collection(
                collection_name=collection_name,
                vectors_config=dense_vectors_config,
                sparse_vectors_config=sparse_vectors_config,
                on_disk_payload=self.on_disk_payload,
            )
            ## Configure Payload Indices
            # (https://qdrant.tech/documentation/concepts/indexing/#payload-index)
            # Index `text`` field for full-text search
            await self._aclient.create_payload_index(
                collection_name=collection_name,
                field_name="text",
                field_schema=rest.TextIndexParams(
                    type="text",
                    tokenizer=rest.TokenizerType.PREFIX,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )
            # Index `node_kind` field for match
            await self._aclient.create_payload_index(
                collection_name=collection_name,
                field_name="node_kind",
                field_schema="keyword",
            )
            # Index fields specific to MIMIC-III data
            await self._aclient.create_payload_index(
                collection_name=collection_name,
                field_name="CHARTDATE",
                field_schema="datetime",
            )
            await self._aclient.create_payload_index(
                collection_name=collection_name,
                field_name="CHARTTIME",
                field_schema="datetime",
            )
            await self._aclient.create_payload_index(
                collection_name=collection_name,
                field_name="STORETIME",
                field_schema="datetime",
            )
            await self._aclient.create_payload_index(
                collection_name=collection_name,
                field_name="CATEGORY",
                field_schema="keyword",
            )
            await self._aclient.create_payload_index(
                collection_name=collection_name,
                field_name="DESCRIPTION",
                field_schema=rest.TextIndexParams(
                    type="text",
                    tokenizer=rest.TokenizerType.PREFIX,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True,
                ),
            )
        except (ValueError, UnexpectedResponse) as exc:
            if "already exists" not in str(exc):
                raise exc  # noqa: TRY201
            logger.info(
                "Collection %s already exists, skipping collection creation.",
                collection_name,
            )
        self._collection_initialized = True

    @qdrant_retry_decorator
    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        try:
            self._client.get_collection(collection_name)
        except (RpcError, UnexpectedResponse, ValueError):
            return False
        return True

    @qdrant_retry_decorator
    async def _acollection_exists(self, collection_name: str) -> Awaitable[bool]:
        """Asynchronous method to check if a collection exists."""
        try:
            await self._aclient.get_collection(collection_name)
        except (RpcError, UnexpectedResponse, ValueError):
            return False
        return True

    def query(
        self,
        query: VectorStoreQuery | list[VectorStoreQuery],
        show_progress: bool = False,
        desc: str = "Querying",
        **kwargs: Any,
    ) -> VectorStoreQueryResult | list[VectorStoreQueryResult]:
        """
        Query index for top k most similar nodes.

        NOTE: We support querying multi-vector embeddings. This is when a single
        entry in the vectorstore has multiple associated embeddings (which may be
        sparse or dense). There are two ways to specify the embedding we want to
        search against:
        (1) Put the vector name in the query.embedding_field VectorStoreQuery attribute
        when performing dense-only or sparse-only search.
        (2) Pass in `dense_vector_name` and `sparse_vector_name` kwargs to the query method.
        The vector type (dense vs. sparse) should match the query.mode which will be
        VectorStoreQueryMode.DEFAULT for dense search and VectorStoreQueryMode.SPARSE for
        sparse search.
        For hybrid search, use query.mode=VectorStoreQueryMode.HYBRID and it is required
        to pass in both `dense_vector_name` and `sparse_vector_name` kwargs since it is
        not possible to infer both vector names from the query.embedding attribute.

        Args:
            query (VectorStoreQuery | list[VectorStoreQuery]): query or list of queries

        Returns:
            VectorStoreQuery | list[VectorStoreQueryResult]:
                If `query` is a single query, returns a single VectorStoreQueryResult.
                If `query` is a list of queries, returns a list of VectorStoreQueryResult
                and the search queries are executed as a batch.
        """
        # Single Query
        if isinstance(query, VectorStoreQuery):
            query_results = self._query_batch([query], **kwargs)
            return query_results[0]
        # Multiple Queries
        else:
            query_batches = list(iter_batch(query, self.query_batch_size))
            query_results = [
                self._query_batch(qb, **kwargs)
                for qb in tqdm(
                    query_batches,
                    disable=not show_progress,
                    desc=desc,
                )
            ]
            query_results = flatten_list_of_list(query_results)
            return query_results

    def _query_batch(
        self,
        query: list[VectorStoreQuery],
        **kwargs: Any,
    ) -> list[VectorStoreQueryResult]:
        # Build Search Requests, Add Query Index to Key
        all_search_requests = self._prepare_search_queries(query, **kwargs)
        # Execute Multiple Search Requests as a Batch
        search_results = self._execute_search_request(all_search_requests)
        # Group Search Results by Query, Fuse Results for Each Query
        query_results = self._postprocess_search_results(search_results)
        return query_results

    async def aquery(
        self,
        query: VectorStoreQuery | list[VectorStoreQuery],
        show_progress: bool = False,
        desc: str = "Querying",
        **kwargs: Any,
    ) -> Awaitable[VectorStoreQueryResult | list[VectorStoreQueryResult]]:
        """
        Asynchronous method to query index for top k most similar nodes.

        Args:
            query (VectorStoreQuery | list[VectorStoreQuery]): query or list of queries

        Returns:
            VectorStoreQuery | list[VectorStoreQueryResult]:
                If `query` is a single query, returns a single VectorStoreQueryResult.
                If `query` is a list of queries, returns a list of VectorStoreQueryResult
                and the search queries are executed as a batch.
        """
        # Single Query
        if isinstance(query, VectorStoreQuery):
            query_results = await self._aquery_batch([query], **kwargs)
            return query_results[0]
        # Multiple Queries
        else:
            query_batches = list(iter_batch(query, self.query_batch_size))
            jobs = [self._aquery_batch(qb, **kwargs) for qb in query_batches]
            query_results = await run_jobs(
                jobs,
                workers=self.num_query_workers,
                show_progress=show_progress,
                desc=desc,
            )
            query_results = flatten_list_of_list(query_results)
            return query_results

    async def _aquery_batch(
        self, query: list[VectorStoreQuery], **kwargs: Any
    ) -> Awaitable[list[VectorStoreQueryResult]]:
        # Build Search Requests, Add Query Index to Key
        all_search_requests = self._prepare_search_queries(query, **kwargs)
        # Execute Multiple Search Requests as a Batch
        search_results = await self._aexecute_search_request(all_search_requests)
        # Group Search Results by Query, Fuse Results for Each Query
        query_results = self._postprocess_search_results(search_results)
        return query_results

    def _prepare_search_queries(
        self, query: list[VectorStoreQuery], **kwargs: Any
    ) -> dict[str, rest.SearchRequest]:
        """Create search requests for all queries.
        NOTE: Some VectorStoreQuery will become multiple SearchRequests (e.g. Hybrid Queries)
        so we additionally prepend the query index to the key to ensure unique keys and
        to ensure that we can group the correct queries together to fuse results.
        """
        all_search_requests: dict[str, rest.SearchRequest] = {}
        for i, q in enumerate(query):
            search_requests = self._make_search_request(query=q, **kwargs)
            # Rename key with query index to ensure queries with same vector names have unique keys
            search_requests = {f"{i}-{k}": v for k, v in search_requests.items()}
            all_search_requests |= search_requests
        return all_search_requests

    def _postprocess_search_results(
        self, search_results: dict[str, rest.ScoredPoint | rest.Record]
    ) -> dict[str, list[VectorStoreQueryResult]]:
        """Postprocess result and fuse results for each query."""
        # Parse each response list of Qdrant Points to list of NodeWithScore
        search_results_nws = {
            k: qdrant_query_response_to_node_with_score(v) for k, v in search_results.items()
        }
        # Group Search Results by Query Index
        results_grouped_by_query: dict[int, dict[str, list[NodeWithScore]]] = defaultdict(dict)
        for key, nws_list in search_results_nws.items():
            query_idx, vector_name = key.split("-")
            query_idx = int(query_idx)
            if query_idx not in results_grouped_by_query:
                results_grouped_by_query[query_idx] = {vector_name: nws_list}
            else:
                results_grouped_by_query[query_idx] |= {vector_name: nws_list}
        # Apply Fusion to Grouped Search Results (yields one VectorStoreQueryResult per Query)
        query_results = [
            self._fuse_nws_results(results_for_query_idx)
            for results_for_query_idx in results_grouped_by_query.values()
        ]
        return query_results

    def _make_search_request(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> dict[str, rest.SearchRequest]:
        match query.mode:
            case VectorStoreQueryMode.HYBRID:
                if not self.enable_hybrid:
                    raise ValueError(
                        "Hybrid search is not enabled. Please build the query with "
                        "`enable_hybrid=True` in the constructor."
                    )
                else:
                    return self._hybrid_search_request(query, **kwargs)
            case VectorStoreQueryMode.DEFAULT:
                return self._dense_search_request(query, **kwargs)
            case VectorStoreQueryMode.SPARSE:
                return self._sparse_search_request(query, **kwargs)
            case _:
                return self._dense_search_request(query, **kwargs)

    def _dense_search_request(
        self, query: VectorStoreQuery, dense_vector_name: str | None = None, **kwargs: Any
    ) -> dict[str, rest.SearchRequest]:
        """
        Create search request using only dense embeddings.

        Args:
            query (VectorStoreQuery): query object which contains string and optionally
                query embeddings. If query embeddings are not provided, they are computed
                from the query string using the query encoder function.
            dense_vector_name (str, optional): name of the dense vector to search with.
                If not provided, will be inferred from query.embedding_field.
        """
        if dense_vector_name is None:
            dense_vector_name = query.embedding_field or "dense"
        search_requests = self._build_search_requests(
            query=query,
            search_dense=True,
            search_sparse=False,
            dense_vector_name=dense_vector_name,
            **kwargs,
        )
        return search_requests

    def _sparse_search_request(
        self, query: VectorStoreQuery, sparse_vector_name: str | None = None, **kwargs: Any
    ) -> dict[str, rest.SearchRequest]:
        """
        Create search request using only sparse embeddings.

        Args:
            query (VectorStoreQuery): query object which contains string and optionally
                query embeddings. If query embeddings are not provided, they are computed
                from the query string using the query encoder function.
            sparse_vector_name (str, optional): name of the sparse vector to search with.
                If not provided, will be inferred from query.embedding_field.
        """
        if sparse_vector_name is None:
            sparse_vector_name = query.embedding_field or "sparse"
        search_requests = self._build_search_requests(
            query=query,
            search_dense=False,
            search_sparse=True,
            sparse_vector_name=sparse_vector_name,
            **kwargs,
        )
        return search_requests

    def _hybrid_search_request(
        self,
        query: VectorStoreQuery,
        dense_vector_name: str,
        sparse_vector_name: str,
        **kwargs: Any,
    ) -> dict[str, rest.SearchRequest]:
        """
        Create two different search requests (one using dense embeddings
        and one using sparse embeddings). Executing the search requests will return
        the top k results for each search request.

        This method also assumes that `query.query_embedding` is a dict with named embeddings
        as it is not possible to perform hybrid_query with a single dense or sparse embedding.

        Args:
            query (VectorStoreQuery): query object which contains string and optionally
                query embeddings. If query embeddings are not provided, they are computed
                from the query string using the query encoder function.
            dense_vector_name (str): name of the dense vector to search with.
                NOTE: for hybrid query this is required to distinguish between
                dense and sparse vectors.
            sparse_vector_name (str): name of the sparse vector to search with.
                NOTE: for hybrid query this is required to distinguish between
                dense and sparse vectors.
        """
        if not dense_vector_name:
            raise ValueError("Missing `dense_vector_name`. Required for Hybrid Query.")
        if not sparse_vector_name:
            raise ValueError("Missing `sparse_vector_name`. Required for Hybrid Query.")
        search_requests = self._build_search_requests(
            query=query,
            search_dense=True,
            search_sparse=True,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            **kwargs,
        )
        assert isinstance(search_requests, dict), (
            "Hybrid query requires encoder that generates multiple named embeddings, "
            "but only a single embedding is present."
        )
        return search_requests

    def _build_search_requests(
        self,
        query: VectorStoreQuery,
        search_dense: bool = True,
        search_sparse: bool = True,
        dense_vector_name: str | None = None,
        sparse_vector_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, rest.SearchRequest]:
        """Convert vector store query to a dict of Qdrant Search Requests.
        If a a multivector query is generated (e.g. query with multiple embeddings),
        then a separate search request is generated for each embedding/vector.

        Use `dense_vector_name` and `sparse_vector_name` for dense and sparse search,
        respectively, to search specific named vectors in Qdrant as it is possible to
        have multiple dense vectors and multiple sparse vectors per Qdrant Point.
        """
        # Merge native qdrant_filters & LlamaIndex metadata filters.
        metadata_filters: Filter = cast(Filter, self._build_query_filter(query))
        qdrant_filters: Filter = kwargs.get("qdrant_filters")
        if metadata_filters and qdrant_filters:
            query_filter = self._merge_filters([metadata_filters, qdrant_filters])
        elif metadata_filters:
            query_filter = metadata_filters
        elif qdrant_filters:
            query_filter = qdrant_filters
        else:
            query_filter = None

        # Compute Embeddings from query_str only if no embeddings in VectorStoreQuery
        if not query.query_embedding:
            query_embedding = self._query_encoder_fn([query.query_str])
        else:
            query_embedding = query.query_embedding
        if isinstance(query_embedding, list):
            query_embedding = query_embedding[0]

        dense_top_k = query.similarity_top_k
        sparse_top_k = query.sparse_top_k or query.similarity_top_k

        # Create Dictionary of Search Requests from Query Embeddings
        search_requests = {}
        if isinstance(query_embedding, dict):  # Dict of NamedEmbeddings
            for key, embedding_item in query_embedding.items():
                if isinstance(embedding_item, EmbeddingItem):
                    vector_name = embedding_item.name if embedding_item.name is not None else key
                    embedding = embedding_item.embedding
                    vector_kind = embedding_item.kind
                else:  # If dict values are not EmbeddingItems, assume they are the embedding
                    vector_name = key
                    embedding = embedding_item
                    vector_kind = "dense" if isinstance(embedding, list) else "sparse"

                # Create Dense Search Request
                if search_dense and vector_kind == "dense":
                    if dense_vector_name is None:
                        dense_vector_name = vector_name
                    search_request = rest.SearchRequest(
                        vector=rest.NamedVector(name=dense_vector_name, vector=embedding),
                        limit=dense_top_k,
                        filter=query_filter,
                        with_payload=True,
                    )
                    search_requests[dense_vector_name] = search_request
                # Create Sparse Search Request
                if search_sparse and vector_kind == "sparse":
                    if sparse_vector_name is None:
                        sparse_vector_name = vector_name
                    sparse_indices, sparse_vectors = sparse_embedding_dict_to_index_vector_lists(
                        embedding
                    )
                    search_request = rest.SearchRequest(
                        vector=rest.NamedSparseVector(
                            name=sparse_vector_name,
                            vector=rest.SparseVector(
                                indices=sparse_indices,
                                values=sparse_vectors,
                            ),
                        ),
                        limit=sparse_top_k,
                        filter=query_filter,
                        with_payload=True,
                    )
                    search_requests[sparse_vector_name] = search_request
        else:
            # If only one embedding, assume Single Dense Embedding (default in LlamaIndex)
            search_request = rest.SearchRequest(
                vector=rest.NamedVector(name=dense_vector_name, vector=query_embedding),
                limit=query.similarity_top_k,
                filter=query_filter,
                with_payload=True,
            )
            search_requests[vector_name] = search_request
        return search_requests

    @qdrant_retry_decorator
    def _execute_search_request(
        self, search_requests: dict[str, rest.SearchRequest]
    ) -> dict[str, Any]:
        """Execute search requests and return the response.
        Each search request is a dict key-value pair where key is the vector name.
        Each search result is the value which is a list of Qdrant Points.
        """
        responses: list[Any] = self._client.search_batch(
            collection_name=self.collection_name, requests=search_requests.values()
        )
        return dict(zip(search_requests.keys(), responses, strict=True))

    @qdrant_retry_decorator
    async def _aexecute_search_request(
        self,
        search_requests: dict[str, rest.SearchRequest],
    ) -> Awaitable[dict[str, Any]]:
        """Execute search requests and return the responses.
        Each search request is a dict key-value pair where key is the vector name.
        Each search result is the value which is a list of Qdrant Points.
        """
        responses: list[Any] = await self._aclient.search_batch(
            collection_name=self.collection_name, requests=search_requests.values()
        )
        return dict(zip(search_requests.keys(), responses, strict=True))

    def _parse_search_results_to_nws(
        self, search_results: list[rest.ScoredPoint | rest.Record]
    ) -> dict[str, list[NodeWithScore]]:
        """Convert each Qdrant search result list to list of NodeWithScore."""
        return {k: qdrant_query_response_to_node_with_score(v) for k, v in search_results.items()}

    def _fuse_nws_results(
        self, search_results_nws: dict[str, list[NodeWithScore]]
    ) -> VectorStoreQueryResult:
        """If multiple search results, fuse them. Return as VectorStoreQueryResult.

        This method uses self._fusion_fn which is defined upon the Qdrant Vectorstore
        initialization, which is Distribution-Based Fusion Score by default. This default
        fusion implementation can fuse an arbitrary number of search result lists.

        The fusion function operates on a dictionary of search results where the key is
        the vector name and the value is a list of NodeWithScore objects that has been
        retrieved from the Vectorstore."""
        # Only One Query Embedding Search Request & Response, No Fusion Needed
        if len(search_results_nws) == 0:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])
        if len(search_results_nws) == 1:
            node_with_scores: list[NodeWithScore] = list(search_results_nws.values())[0]
            return VectorStoreQueryResult(
                nodes=[nws.node for nws in node_with_scores],
                similarities=[nws.score for nws in node_with_scores],
                ids=[nws.node_id for nws in node_with_scores],
            )
        # Multiple Query Embedding Search Requests & Responses, Apply Fusion
        else:
            # Fusion
            fused_node_with_scores: list[NodeWithScore] = self._fusion_fn(
                results=search_results_nws
            )
            # Create Final VectorStoreQueryResult
            return VectorStoreQueryResult(
                nodes=[nws.node for nws in fused_node_with_scores],
                similarities=[nws.score for nws in fused_node_with_scores],
                ids=[nws.node_id for nws in fused_node_with_scores],
            )

    def _parse_and_fuse_search_results(
        self, search_results: dict[str, list[rest.ScoredPoint | rest.Record]]
    ) -> VectorStoreQueryResult:
        search_results_nws = self._parse_search_results_to_nws(search_results)
        return self._fuse_nws_results(search_results_nws)

    def scroll(
        self,
        query: VectorStoreQuery | str = "",
        qdrant_filters: Filter | None = None,
        num_items: int = 50,
        offset: int = 0,
        order_by: str | None = None,
        ascending: str = "asc",
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs: Any,
    ) -> list[NodeWithScore]:
        """Scroll through all the results for a given query. This method will start
        at the specified offset and continue to scroll through the results until there
        are no more results to return.

        Args:
            query (VectorStoreQuery | str): query or query string. This is applied
                as a text search filter in the `text` payload field. If empty string
                then no text search is applied.
            qdrant_filters (Filter, optional): Qdrant Filter object to apply to the query.
            num_items (int, optional): number of results to return per scroll. Default is 50.
            offset (int, optional): offset to start the scroll from. Default is 0.
            order_by (str, optional): field to order the results by. Default is None.
            ascending (str, optional): order direction. Default is "asc".
            with_payload (bool, optional): whether to return payload with results. Default is True.
            with_vectors (bool, optional): whether to return vectors with results. Default is False.

        Returns:
            list[NodeWithScore]: list of NodeWithScore objects
        """
        all_record_list = []
        while offset is not None:
            record_list, offset = self._scroll(
                query=query,
                qdrant_filters=qdrant_filters,
                num_items=num_items,
                offset=offset,
                order_by=order_by,
                ascending=ascending,
                with_payload=with_payload,
                with_vectors=with_vectors,
                **kwargs,
            )
            all_record_list.extend(record_list)
        return qdrant_query_response_to_node_with_score(all_record_list)

    def _scroll(
        self,
        query: VectorStoreQuery | str = "",
        qdrant_filters: Filter | None = None,
        num_items: int = 50,
        offset: int = 0,
        order_by: str | None = None,
        ascending: str = "asc",
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs,
    ) -> tuple[list[rest.Record], int]:
        """Full-Text Search without vector search.

        This uses the Scroll Points API instead of vector search. In pure text search,
        we do not use vector similarity, but instead use a filter to text search directly on
        the `text` field on nodes.

        NOTE: The payload field specified by `order_by` requires a payload index to be created
        during collection creation, otherwise, `order_by` will fail.
        (https://qdrant.tech/documentation/concepts/points/#order-points-by-payload-key)
        """
        if isinstance(query, str):
            query_str = query
            query = VectorStoreQuery(query_str=query_str)
        else:
            query_str = query.query_str

        num_items = num_items or query.similarity_top_k

        # Merge native qdrant_filters & LlamaIndex metadata filters.
        metadata_filters: Filter = cast(Filter, self._build_query_filter(query))
        if metadata_filters and qdrant_filters:
            query_filter = self._merge_filters([metadata_filters, qdrant_filters])
        elif metadata_filters:
            query_filter = metadata_filters
        elif qdrant_filters:
            query_filter = qdrant_filters
        else:
            query_filter = rest.Filter()

        # Ensure query filter has `must` field
        if query_filter.must is None:
            query_filter.must = []
        # Add Full-Text Search Filter to Query Filter
        if query_str:
            query_filter.must.append(
                rest.FieldCondition(key="text", match=rest.MatchText(text=query_str))
            )

        # Order By Field
        if order_by:
            order_by = rest.OrderBy(key=order_by, direction=ascending)

        # Create Scroll Request for Full-Text Search
        record_list, next_page_offset = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=num_items,
            offset=offset,
            order_by=order_by,
            with_payload=with_payload,
            with_vectors=with_vectors,
            **kwargs,
        )
        return record_list, next_page_offset

    def _build_query_filter(self, query: VectorStoreQuery) -> Any | None:
        if not query.doc_ids and not query.query_str:
            return None

        from qdrant_client.http.models import (
            FieldCondition,
            Filter,
            MatchAny,
            MatchExcept,
            MatchText,
            MatchValue,
            Range,
        )

        must_conditions = []

        if query.doc_ids:
            must_conditions.append(
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(any=query.doc_ids),
                )
            )

        if query.node_ids:
            must_conditions.append(
                FieldCondition(
                    key="id",
                    match=MatchAny(any=query.node_ids),
                )
            )

        # Qdrant does not use the query.query_str property for the filtering. Full-text
        # filtering cannot handle longer queries and can effectively filter our all the
        # nodes. See: https://github.com/jerryjliu/llama_index/pull/1181

        if query.filters is None:
            return Filter(must=must_conditions)

        for subfilter in query.filters.filters:
            # only for exact match
            if not subfilter.operator or subfilter.operator == "==":
                if isinstance(subfilter.value, float):
                    must_conditions.append(
                        FieldCondition(
                            key=subfilter.key,
                            range=Range(
                                gte=subfilter.value,
                                lte=subfilter.value,
                            ),
                        )
                    )
                else:
                    must_conditions.append(
                        FieldCondition(
                            key=subfilter.key,
                            match=MatchValue(value=subfilter.value),
                        )
                    )
            elif subfilter.operator == "<":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(lt=subfilter.value),
                    )
                )
            elif subfilter.operator == ">":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(gt=subfilter.value),
                    )
                )
            elif subfilter.operator == ">=":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(gte=subfilter.value),
                    )
                )
            elif subfilter.operator == "<=":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        range=Range(lte=subfilter.value),
                    )
                )
            elif subfilter.operator == "text_match":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        match=MatchText(text=subfilter.value),
                    )
                )
            elif subfilter.operator == "!=":
                must_conditions.append(
                    FieldCondition(
                        key=subfilter.key,
                        match=MatchExcept(**{"except": [subfilter.value]}),
                    )
                )

        return Filter(must=must_conditions)

    def _merge_filters(self, filters: list[Filter]) -> Filter:
        """Merge multiple filters into a single filter."""
        should_conditions = []
        min_should_conditions = []
        must_conditions = []
        must_not_conditions = []
        for f in filters:
            if f.should is not None:
                should_conditions.extend(f.should)
            if f.min_should is not None:
                min_should_conditions.extend(f.min_should)
            if f.must is not None:
                must_conditions.extend(f.must)
            if f.must_not is not None:
                must_not_conditions.extend(f.must_not)
        return Filter(
            should=should_conditions if should_conditions else None,
            min_should=min_should_conditions if min_should_conditions else None,
            must=must_conditions if must_conditions else None,
            must_not=must_not_conditions if must_not_conditions else None,
        )
