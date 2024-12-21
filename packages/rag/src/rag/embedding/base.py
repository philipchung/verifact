"""Modified version of LlamaIndex BaseEmbedding which supports Multivector Embeddings

Main changes include:
1. Custom type definitions in EmbeddingEndEvent and EmbeddingStartEvent to support
any data type in embedding field (necessary for multivector embeddings)
2. alias Embedding type to NamedEmbedding (MultivectorEmbedding type)
3. Handle multivector embedding aggregation with multiple named embeddings

Modified from: llama_index.core.base.embeddings.base.BaseEmbedding
"""

import asyncio
from abc import abstractmethod
from collections.abc import Callable, Coroutine, Sequence
from typing import Any

import llama_index.core.instrumentation as instrument
from llama_index.core.async_utils import run_jobs
from llama_index.core.base.embeddings.base import BaseEmbedding as _BaseEmbedding
from llama_index.core.bridge.pydantic import ConfigDict, Field, field_validator
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.constants import DEFAULT_EMBED_BATCH_SIZE
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.core.utils import get_tqdm_iterable

from rag.embedding.instrumentation import EmbeddingEndEvent, EmbeddingStartEvent
from rag.embedding.utils import NamedEmbedding, mean_agg

Embedding = list[float] | NamedEmbedding
dispatcher = instrument.get_dispatcher(__name__)


class BaseEmbedding(_BaseEmbedding):
    """Base class for embedding models."""

    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )
    model_name: str = Field(default="unknown", description="The name of the embedding model.")
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        lte=2048,
    )
    callback_manager: CallbackManager = Field(
        default_factory=lambda: CallbackManager([]), exclude=True
    )
    num_workers: int | None = Field(
        default=None,
        description="The number of workers to use for async embedding calls.",
    )

    @field_validator("callback_manager")
    @classmethod
    def check_callback_manager(cls, v: CallbackManager) -> CallbackManager:
        if v is None:
            return CallbackManager([])
        return v

    @abstractmethod
    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query synchronously.

        Subclasses should implement this method. Reference get_query_embedding's
        docstring for more information.
        """

    @abstractmethod
    async def _aget_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query asynchronously.

        Subclasses should implement this method. Reference get_query_embedding's
        docstring for more information.
        """

    @dispatcher.span
    def get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query.

        When embedding a query, depending on the model, a special instruction
        can be prepended to the raw query string. For example, "Represent the
        question for retrieving supporting documents: ". If you're curious,
        other examples of predefined instructions can be found in
        embeddings/huggingface_utils.py.
        """
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            query_embedding = self._get_query_embedding(query)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [query],
                    EventPayload.EMBEDDINGS: [query_embedding],
                },
            )
        dispatcher.event(
            EmbeddingEndEvent(
                chunks=[query],
                embeddings=[query_embedding],
            )
        )
        return query_embedding

    @dispatcher.span
    async def aget_query_embedding(self, query: str) -> Embedding:
        """Get query embedding."""
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            query_embedding = await self._aget_query_embedding(query)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [query],
                    EventPayload.EMBEDDINGS: [query_embedding],
                },
            )
        dispatcher.event(
            EmbeddingEndEvent(
                chunks=[query],
                embeddings=[query_embedding],
            )
        )
        return query_embedding

    def get_agg_embedding_from_queries(
        self,
        queries: list[str],
        agg_fn: Callable[..., Embedding] | None = None,
    ) -> Embedding:
        """Get aggregated embedding from multiple queries."""
        query_embeddings = [self.get_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    async def aget_agg_embedding_from_queries(
        self,
        queries: list[str],
        agg_fn: Callable[..., Embedding] | None = None,
    ) -> Embedding:
        """Async get aggregated embedding from multiple queries."""
        query_embeddings = [await self.aget_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    @abstractmethod
    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text synchronously.

        Subclasses should implement this method. Reference get_text_embedding's
        docstring for more information.
        """

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text asynchronously.

        Subclasses can implement this method if there is a true async
        implementation. Reference get_text_embedding's docstring for more
        information.
        """
        # Default implementation just falls back on _get_text_embedding
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """
        Embed the input sequence of text synchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        # Default implementation just loops over _get_text_embedding
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """
        Embed the input sequence of text asynchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        return await asyncio.gather(*[self._aget_text_embedding(text) for text in texts])

    @dispatcher.span
    def get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text.

        When embedding text, depending on the model, a special instruction
        can be prepended to the raw text string. For example, "Represent the
        document for retrieval: ". If you're curious, other examples of
        predefined instructions can be found in embeddings/huggingface_utils.py.
        """
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            text_embedding = self._get_text_embedding(text)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [text],
                    EventPayload.EMBEDDINGS: [text_embedding],
                }
            )
        dispatcher.event(
            EmbeddingEndEvent(
                chunks=[text],
                embeddings=[text_embedding],
            )
        )
        return text_embedding

    @dispatcher.span
    async def aget_text_embedding(self, text: str) -> Embedding:
        """Async get text embedding."""
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        dispatcher.event(
            EmbeddingStartEvent(
                model_dict=model_dict,
            )
        )
        with self.callback_manager.event(
            CBEventType.EMBEDDING, payload={EventPayload.SERIALIZED: self.to_dict()}
        ) as event:
            text_embedding = await self._aget_text_embedding(text)

            event.on_end(
                payload={
                    EventPayload.CHUNKS: [text],
                    EventPayload.EMBEDDINGS: [text_embedding],
                }
            )
        dispatcher.event(
            EmbeddingEndEvent(
                chunks=[text],
                embeddings=[text_embedding],
            )
        )
        return text_embedding

    @dispatcher.span
    def get_text_embedding_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[Embedding]:
        """Get a list of text embeddings, with batching."""
        dispatch_event = dispatcher.get_dispatch_event()

        cur_batch: list[str] = []
        result_embeddings: list[Embedding] = []

        queue_with_progress = enumerate(
            get_tqdm_iterable(texts, show_progress, "Generating embeddings")
        )

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        for idx, text in queue_with_progress:
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                dispatch_event(
                    EmbeddingStartEvent(
                        model_dict=self.to_dict(),
                    )
                )
                with self.callback_manager.event(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                ) as event:
                    embeddings = self._get_text_embeddings(cur_batch)
                    result_embeddings.extend(embeddings)
                    event.on_end(
                        payload={
                            EventPayload.CHUNKS: cur_batch,
                            EventPayload.EMBEDDINGS: embeddings,
                        },
                    )
                dispatch_event(
                    EmbeddingEndEvent(
                        chunks=cur_batch,
                        embeddings=embeddings,
                    )
                )
                cur_batch = []

        return result_embeddings

    @dispatcher.span
    async def aget_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False
    ) -> list[Embedding]:
        """Asynchronously get a list of text embeddings, with batching."""
        num_workers = self.num_workers

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)

        cur_batch: list[str] = []
        callback_payloads: list[tuple[str, list[str]]] = []
        result_embeddings: list[Embedding] = []
        embeddings_coroutines: list[Coroutine] = []
        for idx, text in enumerate(texts):
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                dispatcher.event(
                    EmbeddingStartEvent(
                        model_dict=model_dict,
                    )
                )
                event_id = self.callback_manager.on_event_start(
                    CBEventType.EMBEDDING,
                    payload={EventPayload.SERIALIZED: self.to_dict()},
                )
                callback_payloads.append((event_id, cur_batch))
                embeddings_coroutines.append(self._aget_text_embeddings(cur_batch))
                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        nested_embeddings = []

        if num_workers and num_workers > 1:
            nested_embeddings = await run_jobs(
                embeddings_coroutines,
                show_progress=show_progress,
                workers=self.num_workers,
                desc="Generating embeddings",
            )
        else:
            if show_progress:
                try:
                    from tqdm.asyncio import tqdm_asyncio

                    nested_embeddings = await tqdm_asyncio.gather(
                        *embeddings_coroutines,
                        total=len(embeddings_coroutines),
                        desc="Generating embeddings",
                    )
                except ImportError:
                    nested_embeddings = await asyncio.gather(*embeddings_coroutines)
            else:
                nested_embeddings = await asyncio.gather(*embeddings_coroutines)

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]

        for (event_id, text_batch), embeddings in zip(
            callback_payloads, nested_embeddings, strict=False
        ):
            dispatcher.event(
                EmbeddingEndEvent(
                    chunks=text_batch,
                    embeddings=embeddings,
                )
            )
            self.callback_manager.on_event_end(
                CBEventType.EMBEDDING,
                payload={
                    EventPayload.CHUNKS: text_batch,
                    EventPayload.EMBEDDINGS: embeddings,
                },
                event_id=event_id,
            )

        return result_embeddings

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        embeddings = self.get_text_embedding_batch(
            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
            **kwargs,
        )

        for node, embedding in zip(nodes, embeddings, strict=False):
            node.embedding = embedding

        return nodes

    async def acall(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        embeddings = await self.aget_text_embedding_batch(
            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
            **kwargs,
        )

        for node, embedding in zip(nodes, embeddings, strict=False):
            node.embedding = embedding

        return nodes
