"""Modified version of LlamaIndex definitions to support Multivector Embeddings

Modified from: llama_index.core.instrumentation.events.embedding
"""

from typing import Any, Literal

from llama_index.core.bridge.pydantic import ConfigDict
from llama_index.core.instrumentation.events.base import BaseEvent

from rag.embedding.utils import NamedEmbedding


class EmbeddingStartEvent(BaseEvent):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    model_dict: dict

    @classmethod
    def class_name(cls) -> Literal["EmbeddingStartEvent"]:
        """Class name."""
        return "EmbeddingStartEvent"


class EmbeddingEndEvent(BaseEvent):
    chunks: list[str]
    embeddings: list[NamedEmbedding | Any]

    @classmethod
    def class_name(cls) -> Literal["EmbeddingEndEvent"]:
        """Class name."""
        return "EmbeddingEndEvent"
