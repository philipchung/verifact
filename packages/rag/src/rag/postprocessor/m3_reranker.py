"""LlamaIndex wrapper around FlagEmbeddingReranker API service."""

import httpx
from llama_index.core.bridge.pydantic import ConfigDict, Field
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle


class M3Reranker(BaseNodePostprocessor):
    """LlamaIndex wrapper around API to M3 Reranker model.

    This implementation makes API calls to a reranking API endpoint to generate scores
    and is appropriate for async pipelines.  The embedding API endpoint should be configured
    and launched prior to calling methods in this class.

    NOTE `model` init arguments have no effect since they
    are configured on the server side.  The context length accepted by the inference model
    is also configured on the server side.
    """

    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_name: str = Field(
        default="BAAI/bge-reranker-v2-m3", description="BAAI Reranker model name."
    )
    api_base: str = Field(default="http://rerank.localhost/v1", description="API base URL.")
    top_n: int | None = Field(
        default=None,
        description="Number of nodes to return sorted by score. "
        "If `None`, all reranked nodes are returned.",
    )
    timeout: float = Field(default=600.0, description="Timeout for API calls.")
    num_retries: int = Field(default=5, description="Number of retries for API calls.")
    metadata_mode: MetadataMode | str = Field(
        default=MetadataMode.NONE,
        description=(
            "Metadata mode to format document with prior to computing score."
            "'none' means to only use text. "
            "'embed' and 'llm' uses text and metadata based on embed and llm metadata filters. "
            "'all' uses all text and metadata."
        ),
    )

    @classmethod
    def class_name(cls) -> str:
        return "M3Reranker"

    def _compute_score(self, query: str, documents: list[str], **kwargs) -> list[float]:
        """Score documents based on query using the reranker model.
        Returns list of scores in same order as documents."""
        num_retries = kwargs.pop("num_retries", self.num_retries)
        if num_retries == 0:
            raise ConnectionError(
                f"Failed to generate rerank scores after {self.num_retries} retries."
            )

        response = httpx.request(
            method="POST",
            url=self.api_base.rstrip("/") + "/rerank",
            json={
                "query": query,
                "documents": documents,
            },
            timeout=30,
            # timeout=self.timeout,
        )
        if response.is_success:
            # Sort scores by index to match order of documents
            sorted_scores = sorted(response.json()["results"], key=lambda d: d["index"])
            scores = [x["relevance_score"] for x in sorted_scores]
            return scores
        else:
            return self._compute_score(
                query=query,
                documents=documents,
                num_retries=num_retries - 1,
            )

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        query = query_bundle.query_str
        documents = [node.node.get_content(metadata_mode=self.metadata_mode) for node in nodes]

        with self.callback_manager.event(
            CBEventType.RERANKING,
            payload={
                EventPayload.NODES: nodes,
                EventPayload.MODEL_NAME: self.model_name,
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.TOP_K: self.top_n,
            },
        ) as event:
            scores = self._compute_score(query=query, documents=documents)

            # a single node passed into compute_score returns a float
            if isinstance(scores, float):
                scores = [scores]

            assert len(scores) == len(nodes)

            for node, score in zip(nodes, scores, strict=False):
                node.score = score

            new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)
            if self.top_n is not None and len(new_nodes) > self.top_n:
                new_nodes = new_nodes[: self.top_n]
            event.on_end(payload={EventPayload.NODES: new_nodes})

        return new_nodes
