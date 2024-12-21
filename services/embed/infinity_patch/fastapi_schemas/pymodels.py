"""Patch for Infinity Embeddings v0.0.53

To support BGE-M3 model with multiple embeddings generated in a single inference forward
pass, the main change that needs to be made is to relax _EmbeddingObject.embedding to
allow not only `list[float]` (or EmbeddingReturnType), but also allow `Any` type.
This is because in the SentenceTransformerPatched class in
infinity/libs/infinity_emb/infinty_emb/transformer/embedder/sentence_transformer,
we will modify the `encode_post` method to return a dictionary of embeddings instead of
the original `EmbeddingReturnType` type which is an alias for
`npt.NDArray[Union[np.float32, np.float32]]` which is coerced to `list[float]` by pydantic.

We will modify `encode_post` for BGE-M3 model to return a dictionary of embeddings
`dict[str, Any]` with keys "dense", "sparse", and "colbert", and values corresponding
to those respective embeddings.

Original: infinity/libs/infinity_emb/infinity_emb/fastapi_schemas.py
(https://github.com/michaelfeil/infinity/blob/409f3a28d981354aa77ef7912b418033deb8ed67/libs/infinity_emb/infinity_emb/fastapi_schemas/pymodels.py)
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Annotated, Any, Literal
from uuid import uuid4

if TYPE_CHECKING:
    from infinity_emb.primitives import ClassifyReturnType, EmbeddingReturnType  # type: ignore


from infinity_emb._optional_imports import CHECK_PYDANTIC  # type: ignore

# potential backwards compatibility to pydantic 1.X
# pydantic 2.x is preferred by not strictly needed
if CHECK_PYDANTIC.is_available:
    from pydantic import BaseModel, Field, conlist

    try:
        from pydantic import AnyUrl, HttpUrl, StringConstraints

        # Note: adding artificial limit, this might reveal splitting
        # issues on the client side
        #      and is not a hard limit on the server side.
        INPUT_STRING = StringConstraints(max_length=8192 * 15, strip_whitespace=True)
        ITEMS_LIMIT = {
            "min_length": 1,
            "max_length": 2048,
        }
    except ImportError:
        from pydantic import constr

        INPUT_STRING = constr(max_length=8192 * 15, strip_whitespace=True)  # type: ignore
        ITEMS_LIMIT = {
            "min_items": 1,
            "max_items": 2048,
        }
        HttpUrl, AnyUrl = str, str  # type: ignore


else:

    class BaseModel:  # type: ignore[no-redef]
        pass

    def Field(*args, **kwargs):  # type: ignore
        pass

    def conlist():  # type: ignore
        pass


class _Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class OpenAIEmbeddingInput(BaseModel):
    input: conlist(Annotated[str, INPUT_STRING], **ITEMS_LIMIT) | Annotated[str, INPUT_STRING]  # type: ignore
    model: str | None = "default/not-specified"
    user: str | None = None


class ImageEmbeddingInput(BaseModel):
    input: conlist(Annotated[AnyUrl, HttpUrl], **ITEMS_LIMIT) | Annotated[AnyUrl, HttpUrl]  # type: ignore
    model: str = "default/not-specified"
    user: str | None = None


class _EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float] | Any  # <-- Custom change required for BGE-M3 model
    index: int


class OpenAIEmbeddingResult(BaseModel):
    object: Literal["embedding"] = "embedding"
    data: list[_EmbeddingObject]
    model: str
    usage: _Usage
    id: str = Field(default_factory=lambda: f"infinity-{uuid4()}")
    created: int = Field(default_factory=lambda: int(time.time()))

    @staticmethod
    def to_embeddings_response(
        embeddings: Iterable[EmbeddingReturnType],
        model: str,
        usage: int,
    ) -> dict[str, str | list[dict] | dict]:
        return dict(
            model=model,
            data=[
                dict(
                    object="embedding",
                    embedding=emb,
                    index=count,
                )
                for count, emb in enumerate(embeddings)
            ],
            usage=dict(prompt_tokens=usage, total_tokens=usage),
        )


class ClassifyInput(BaseModel):
    input: conlist(  # type: ignore
        Annotated[str, INPUT_STRING],
        **ITEMS_LIMIT,
    )
    model: str = "default/not-specified"
    raw_scores: bool = False


class _ClassifyObject(BaseModel):
    score: float
    label: str


class ClassifyResult(BaseModel):
    object: Literal["classify"] = "classify"
    data: list[list[_ClassifyObject]]
    model: str
    usage: _Usage
    id: str = Field(default_factory=lambda: f"infinity-{uuid4()}")
    created: int = Field(default_factory=lambda: int(time.time()))

    @staticmethod
    def to_classify_response(
        scores_labels: list[ClassifyReturnType],
        model: str,
        usage: int,
    ) -> dict[str, str | list[ClassifyReturnType] | dict]:
        return dict(
            model=model,
            data=scores_labels,
            usage=dict(prompt_tokens=usage, total_tokens=usage),
        )


class RerankInput(BaseModel):
    query: Annotated[str, INPUT_STRING]
    documents: conlist(  # type: ignore
        Annotated[str, INPUT_STRING],
        **ITEMS_LIMIT,
    )
    return_documents: bool = False
    model: str = "default/not-specified"


class _ReRankObject(BaseModel):
    relevance_score: float
    index: int
    document: str | None = None


class ReRankResult(BaseModel):
    object: Literal["rerank"] = "rerank"
    data: list[_ReRankObject]
    model: str
    usage: _Usage
    id: str = Field(default_factory=lambda: f"infinity-{uuid4()}")
    created: int = Field(default_factory=lambda: int(time.time()))

    @staticmethod
    def to_rerank_response(
        scores: list[float],
        model=str,
        usage=int,
        documents: list[str] | None = None,
    ) -> dict:
        if documents is None:
            return dict(
                model=model,
                results=[
                    dict(relevance_score=score, index=count) for count, score in enumerate(scores)
                ],
                usage=dict(prompt_tokens=usage, total_tokens=usage),
            )
        else:
            return dict(
                model=model,
                results=[
                    dict(relevance_score=score, index=count, document=doc)
                    for count, (score, doc) in enumerate(zip(scores, documents, strict=False))
                ],
                usage=dict(prompt_tokens=usage, total_tokens=usage),
            )


class ModelInfo(BaseModel):
    id: str
    stats: dict[str, Any]
    object: Literal["model"] = "model"
    owned_by: Literal["infinity"] = "infinity"
    created: int = Field(default_factory=lambda: int(time.time()))
    backend: str = ""
    capabilities: set[str] = set()


class OpenAIModelInfo(BaseModel):
    data: list[ModelInfo]
    object: str = "list"
