"""LlamaIndex Node schemas.

Modified from llama_index.core.schema to inherit from modified BaseNode class
to support multiple embeddings per node.

Note that we also inherit from the original LlamaIndex nodes at the end of the MRO
so that pydantic detects that our custom classes are subclasses of
the original LlamaIndex classes."""

import json
import textwrap
import uuid
from dataclasses import dataclass
from io import BytesIO
from typing import TYPE_CHECKING, Any

from dataclasses_json import DataClassJsonMixin
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.schema import (
    TRUNCATE_LENGTH,
    WRAP_WIDTH,
    ImageType,
    ObjectType,
)
from llama_index.core.schema import Document as _Document
from llama_index.core.schema import ImageDocument as _ImageDocument
from llama_index.core.schema import ImageNode as _ImageNode
from llama_index.core.schema import IndexNode as _IndexNode
from llama_index.core.utils import SAMPLE_TEXT, truncate_text

from .base_node import BaseNode, TextNode

if TYPE_CHECKING:
    from haystack.dataclasses.document import Document as HaystackDocument
    from llama_index.core.bridge.langchain import Document as LCDocument
    from semantic_kernel.memory.memory_record import MemoryRecord


# TODO: legacy backport of old Node class
Node = TextNode


class ImageNode(TextNode, _ImageNode):
    """Node with image."""

    # TODO: store reference instead of actual image
    # base64 encoded image str
    image: str | None = None
    image_path: str | None = None
    image_url: str | None = None
    image_mimetype: str | None = None
    text_embedding: list[float] | None = Field(
        default=None,
        description="Text embedding of image node, if text field is filled out",
    )

    @classmethod
    def get_type(cls) -> str:
        return ObjectType.IMAGE

    @classmethod
    def class_name(cls) -> str:
        return "ImageNode"

    def resolve_image(self) -> ImageType:
        """Resolve an image such that PIL can read it."""
        if self.image is not None:
            import base64

            return BytesIO(base64.b64decode(self.image))
        elif self.image_path is not None:
            return self.image_path
        elif self.image_url is not None:
            # load image from URL
            import requests

            response = requests.get(self.image_url)
            return BytesIO(response.content)
        else:
            raise ValueError("No image found in node.")


class IndexNode(TextNode, _IndexNode):
    """Node with reference to any object.

    This can include other indices, query engines, retrievers.

    This can also include other nodes (though this is overlapping with `relationships`
    on the Node class).
    """

    index_id: str
    obj: Any = None

    def dict(self, **kwargs: Any) -> dict[str, Any]:
        from llama_index.core.storage.docstore.utils import doc_to_json

        data = super().dict(**kwargs)

        try:
            if self.obj is None:
                data["obj"] = None
            elif isinstance(self.obj, BaseNode):
                data["obj"] = doc_to_json(self.obj)
            elif isinstance(self.obj, BaseModel):
                data["obj"] = self.obj.dict()
            else:
                data["obj"] = json.dumps(self.obj)
        except Exception as ex:
            raise ValueError("IndexNode obj is not serializable: " + str(self.obj)) from ex

        return data

    @classmethod
    def from_text_node(
        cls,
        node: TextNode,
        index_id: str,
    ) -> "IndexNode":
        """Create index node from text node."""
        # copy all attributes from text node, add index id
        return cls(
            **node.dict(),
            index_id=index_id,
        )

    # # TODO: return type here not supported by current mypy version
    # @classmethod
    # def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> "IndexNode":  # type: ignore
    #     output = super().from_dict(data, **kwargs)

    #     obj = data.get("obj")
    #     parsed_obj = None

    #     if isinstance(obj, str):
    #         parsed_obj = TextNode(text=obj)
    #     elif isinstance(obj, dict):
    #         from llama_index.core.storage.docstore.utils import json_to_doc

    #         # check if its a node, else assume stringable
    #         try:
    #             parsed_obj = json_to_doc(obj)
    #         except Exception:
    #             parsed_obj = TextNode(text=str(obj))

    #     output.obj = parsed_obj

    #     return output

    @classmethod
    def get_type(cls) -> str:
        return ObjectType.INDEX

    @classmethod
    def class_name(cls) -> str:
        return "IndexNode"


# Document Classes for Readers


class Document(TextNode, _Document):
    """Generic interface for a data document.

    This document connects to data sources.

    """

    # TODO: A lot of backwards compatibility logic here, clean up
    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique ID of the node.",
        alias="doc_id",
    )

    _compat_fields = {"doc_id": "id_", "extra_info": "metadata"}

    @classmethod
    def get_type(cls) -> str:
        """Get Document type."""
        return ObjectType.DOCUMENT

    @property
    def doc_id(self) -> str:
        """Get document ID."""
        return self.id_

    def __str__(self) -> str:
        source_text_truncated = truncate_text(self.get_content().strip(), TRUNCATE_LENGTH)
        source_text_wrapped = textwrap.fill(f"Text: {source_text_truncated}\n", width=WRAP_WIDTH)
        return f"Doc ID: {self.doc_id}\n{source_text_wrapped}"

    def get_doc_id(self) -> str:
        """TODO: Deprecated: Get document ID."""
        return self.id_

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._compat_fields:
            name = self._compat_fields[name]
        super().__setattr__(name, value)

    def to_langchain_format(self) -> "LCDocument":
        """Convert struct to LangChain document format."""
        from llama_index.core.bridge.langchain import Document as LCDocument

        metadata = self.metadata or {}
        return LCDocument(page_content=self.text, metadata=metadata)

    @classmethod
    def from_langchain_format(cls, doc: "LCDocument") -> "Document":
        """Convert struct from LangChain document format."""
        return cls(text=doc.page_content, metadata=doc.metadata)

    def to_haystack_format(self) -> "HaystackDocument":
        """Convert struct to Haystack document format."""
        from haystack.dataclasses.document import Document as HaystackDocument

        return HaystackDocument(
            content=self.text, meta=self.metadata, embedding=self.embedding, id=self.id_
        )

    @classmethod
    def from_haystack_format(cls, doc: "HaystackDocument") -> "Document":
        """Convert struct from Haystack document format."""
        return cls(text=doc.content, metadata=doc.meta, embedding=doc.embedding, id_=doc.id)

    def to_embedchain_format(self) -> dict[str, Any]:
        """Convert struct to EmbedChain document format."""
        return {
            "doc_id": self.id_,
            "data": {"content": self.text, "meta_data": self.metadata},
        }

    @classmethod
    def from_embedchain_format(cls, doc: dict[str, Any]) -> "Document":
        """Convert struct from EmbedChain document format."""
        return cls(
            text=doc["data"]["content"],
            metadata=doc["data"]["meta_data"],
            id_=doc["doc_id"],
        )

    def to_semantic_kernel_format(self) -> "MemoryRecord":
        """Convert struct to Semantic Kernel document format."""
        import numpy as np
        from semantic_kernel.memory.memory_record import MemoryRecord

        return MemoryRecord(
            id=self.id_,
            text=self.text,
            additional_metadata=self.get_metadata_str(),
            embedding=np.array(self.embedding) if self.embedding else None,
        )

    @classmethod
    def from_semantic_kernel_format(cls, doc: "MemoryRecord") -> "Document":
        """Convert struct from Semantic Kernel document format."""
        return cls(
            text=doc._text,
            metadata={"additional_metadata": doc._additional_metadata},
            embedding=doc._embedding.tolist() if doc._embedding is not None else None,
            id_=doc._id,
        )

    def to_vectorflow(self, client: Any) -> None:
        """Send a document to vectorflow, since they don't have a document object."""
        # write document to temp file
        import tempfile

        with tempfile.NamedTemporaryFile() as f:
            f.write(self.text.encode("utf-8"))
            f.flush()
            client.embed(f.name)

    @classmethod
    def example(cls) -> "Document":
        return Document(
            text=SAMPLE_TEXT,
            metadata={"filename": "README.md", "category": "codebase"},
        )

    @classmethod
    def class_name(cls) -> str:
        return "Document"


class ImageDocument(Document, ImageNode, _ImageDocument):
    """Data document containing an image."""

    @classmethod
    def class_name(cls) -> str:
        return "ImageDocument"


@dataclass
class QueryBundle(DataClassJsonMixin):
    """
    Query bundle.

    This dataclass contains the original query string and associated transformations.

    Args:
        query_str (str): the original user-specified query string.
            This is currently used by all non embedding-based queries.
        custom_embedding_strs (list[str]): list of strings used for embedding the query.
            This is currently used by all embedding-based queries.
        embedding (list[float]): the stored embedding for the query.
    """

    query_str: str
    # using single image path as query input
    image_path: str | None = None
    custom_embedding_strs: list[str] | None = None
    embedding: list[float] | None = None

    @property
    def embedding_strs(self) -> list[str]:
        """Use custom embedding strs if specified, otherwise use query str."""
        if self.custom_embedding_strs is None:
            if len(self.query_str) == 0:
                return []
            return [self.query_str]
        else:
            return self.custom_embedding_strs

    @property
    def embedding_image(self) -> list[ImageType]:
        """Use image path for image retrieval."""
        if self.image_path is None:
            return []
        return [self.image_path]

    def __str__(self) -> str:
        """Convert to string representation."""
        return self.query_str


QueryType = str | QueryBundle
