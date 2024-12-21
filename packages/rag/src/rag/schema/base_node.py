"""Custom BaseNode and NodeWithScore schema for LlamaIndex
to support multiple embedding vectors per node.

Modified from llama_index.core.schema.

Note that we also inherit from the original LlamaIndex nodes at the end of the MRO
so that pydantic detects that our custom clasess are subclasses of
the original LlamaIndex classes.
"""

import textwrap
import uuid
from abc import abstractmethod
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from llama_index.core.bridge.pydantic import ConfigDict, Field
from llama_index.core.schema import (
    DEFAULT_METADATA_TMPL,
    DEFAULT_TEXT_NODE_TMPL,
    TRUNCATE_LENGTH,
    WRAP_WIDTH,
    MetadataMode,
    NodeRelationship,
    ObjectType,
    RelatedNodeInfo,
    RelatedNodeType,
)
from llama_index.core.schema import BaseNode as _BaseNode
from llama_index.core.schema import NodeWithScore as _NodeWithScore
from llama_index.core.schema import TextNode as _TextNode
from llama_index.core.utils import truncate_text

if TYPE_CHECKING:
    pass

import warnings

from .base import BaseComponent


class EmbeddingItem(BaseComponent):
    """Container for a single embedding."""

    name: str | None = Field(default=None, description="Name of embedding.")
    embedding: list[float] | Any | None = Field(default=None, description="Embedding vector.")
    kind: str = Field(default="dense", description="Kind of embedding (dense, sparse, etc.)")

    @classmethod
    def class_name(cls) -> str:
        return "Embedding"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EmbeddingItem):
            return False
        return self.name == other.name

    def __hash__(self) -> int:
        return hash(f"{self.class_name} {self.name}")


# Node classes for indexes
class BaseNode(BaseComponent, _BaseNode):
    """Modified version of llama_index.core.schema.BaseNode to allow for multiple
    embeddings in a single node. However, most of LlamaIndex assumes each node
    has a single embedding representation, so we need to maintain backward compatibility.

    In addition to the default `embedding` Field from typical LlamaIndex BaseNode,
    there is also an `embeddings` Field which allows storing multiple embeddings
    in a single node.  The `embedding` Field is kept for compatibility with LlamaIndex
    and should always be the same as  `embeddings[default_embedding_name]`.
    """

    # hash is computed on local field, during the validation process
    model_config = ConfigDict(populate_by_name=True, validate_assignment=True)

    id_: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Unique ID of the node."
    )
    """
    Default dense vector embedding, is stored in both `embedding` and `embeddings` Field.
    The `embeddings` Field is a dict of EmbeddingItem objects.
    The `embedding` Field mirrors the embedding value of the default embedding.
    To properly set embeddings, use the defined setter and getters, which will mirror
    the default dense vector embedding to both `embedding` and `embeddings` Fields.
    """
    default_embedding_name: str = Field(
        default="dense", description="Key for `embedding` in `embeddings` dictionary."
    )
    embedding: list[float] | None = Field(
        default=None, description="Default embedding of the node."
    )
    embeddings: dict[str, EmbeddingItem] = Field(
        default_factory=dict,
        description="All embeddings for the node as a dict of EmbeddingItem",
    )

    def set_embedding(
        self, name: str | None, embedding: Any, kind: str | None = None, default: str | None = None
    ) -> None:
        """Set the embedding of the node."""
        # Set default embedding name if provided (as a convenience)
        if default:
            self.default_embedding_name = default
        # If no explicit key provided, use default embedding key
        if name is None:
            name = self.default_embedding_name
        # Set the default embedding only if embedding matches default_embedding_name
        if name == self.default_embedding_name:
            self.embedding = embedding
        # Add embedding to the embeddings dictionary
        self.embeddings |= {name: EmbeddingItem(name=name, embedding=embedding, kind=kind)}

    def get_embedding(self, name: str | None = None) -> Any | EmbeddingItem:
        """Get the embedding of the node."""
        # Get default embedding if no name specified
        if name is None:
            try:
                return self.embedding
            except Exception:
                try:
                    name = self.default_embedding_name
                    return self.embeddings[name]
                except Exception as ex2:
                    raise ValueError("Default embedding not set on node.") from ex2
        # Get named embedding in the embeddings dictionary if name specified
        else:
            return self.embeddings[name]

    def set_embeddings(
        self,
        embeddings: dict[str, Any] | None = None,
        default: str | None = None,
        **kwargs,
    ) -> None:
        """Set multiple embeddings of the node."""
        # Set default embedding name if provided (as a convenience)
        if default:
            self.default_embedding_name = default
        # If embedding k-v pairs provided as keyword arguments
        if kwargs:
            self.set_embeddings(embeddings=kwargs, default=default)
        else:
            # embeddings provided as dict
            if isinstance(embeddings, dict):
                for key, value in embeddings.items():
                    if isinstance(value, EmbeddingItem):
                        self.set_embedding(
                            name=value.name,
                            embedding=value.embedding,
                            kind=value.kind,
                            default=default,
                        )
                        # Set default embedding value
                        if key == self.default_embedding_name:
                            self.embedding = value.embedding
                    else:
                        try:
                            warnings.warn(
                                "Embeddings value set without using EmbeddingItem object. "
                                "`kind` is not set. Use EmbeddingItem object as dict value "
                                "when setting to explicitly specify embedding kind.",
                                stacklevel=2,
                            )
                            self.set_embedding(
                                name=key, embedding=value, kind=None, default=default
                            )
                            # Set default embedding value
                            if key == self.default_embedding_name:
                                self.embedding = value
                        except Exception as ex:
                            raise ValueError("Error setting embeddings: {ex}.") from ex
            else:
                raise ValueError(
                    "Invalid embeddings type. Must be key-value dict with "
                    "Embedding or EmbeddingItem value."
                )

    def get_embeddings(self, names: list[str] | None = None) -> Any | dict[str, Any]:
        """Get multiple embeddings of the node."""
        # If no names specified, return all embeddings as a dictionary
        if names is None:
            return self.embeddings
        # If embedding names specified, return only those embeddings as a dictionary
        else:
            if not isinstance(names, list):
                names = [names]
            return {k: v for k, v in self.embeddings.items() if k in names}

    """"
    metadata fields
    - injected as part of the text shown to LLMs as context
    - injected as part of the text for generating embeddings
    - used by vector DBs for metadata filtering

    """
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="A flat dictionary of metadata fields",
        alias="extra_info",
    )
    excluded_embed_metadata_keys: list[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the embed model.",
    )
    excluded_llm_metadata_keys: list[str] = Field(
        default_factory=list,
        description="Metadata keys that are excluded from text for the LLM.",
    )
    relationships: dict[NodeRelationship, RelatedNodeType] = Field(
        default_factory=dict,
        description="A mapping of relationships to other node information.",
    )

    @classmethod
    @abstractmethod
    def get_type(cls) -> str:
        """Get Object type."""

    @abstractmethod
    def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
        """Get object content."""

    @abstractmethod
    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata string."""

    @abstractmethod
    def set_content(self, value: Any) -> None:
        """Set the content of the node."""

    @property
    @abstractmethod
    def hash(self) -> str:
        """Get hash of node."""

    @property
    def node_id(self) -> str:
        return self.id_

    @node_id.setter
    def node_id(self, value: str) -> None:
        self.id_ = value

    @property
    def source_node(self) -> RelatedNodeInfo | None:
        """Source object node.

        Extracted from the relationships field.

        """
        if NodeRelationship.SOURCE not in self.relationships:
            return None

        relation = self.relationships[NodeRelationship.SOURCE]
        if isinstance(relation, list):
            raise ValueError("Source object must be a single RelatedNodeInfo object")
        return relation

    @property
    def prev_node(self) -> RelatedNodeInfo | None:
        """Prev node."""
        if NodeRelationship.PREVIOUS not in self.relationships:
            return None

        relation = self.relationships[NodeRelationship.PREVIOUS]
        if not isinstance(relation, RelatedNodeInfo):
            raise ValueError("Previous object must be a single RelatedNodeInfo object")
        return relation

    @property
    def next_node(self) -> RelatedNodeInfo | None:
        """Next node."""
        if NodeRelationship.NEXT not in self.relationships:
            return None

        relation = self.relationships[NodeRelationship.NEXT]
        if not isinstance(relation, RelatedNodeInfo):
            raise ValueError("Next object must be a single RelatedNodeInfo object")
        return relation

    @property
    def parent_node(self) -> RelatedNodeInfo | None:
        """Parent node."""
        if NodeRelationship.PARENT not in self.relationships:
            return None

        relation = self.relationships[NodeRelationship.PARENT]
        if not isinstance(relation, RelatedNodeInfo):
            raise ValueError("Parent object must be a single RelatedNodeInfo object")
        return relation

    @property
    def child_nodes(self) -> list[RelatedNodeInfo] | None:
        """Child nodes."""
        if NodeRelationship.CHILD not in self.relationships:
            return None

        relation = self.relationships[NodeRelationship.CHILD]
        if not isinstance(relation, list):
            raise ValueError("Child objects must be a list of RelatedNodeInfo objects.")
        return relation

    @property
    def ref_doc_id(self) -> str | None:
        """Deprecated: Get ref doc id."""
        source_node = self.source_node
        if source_node is None:
            return None
        return source_node.node_id

    @property
    def extra_info(self) -> dict[str, Any]:
        """TODO: DEPRECATED: Extra info."""
        return self.metadata

    def __str__(self) -> str:
        source_text_truncated = truncate_text(self.get_content().strip(), TRUNCATE_LENGTH)
        source_text_wrapped = textwrap.fill(f"Text: {source_text_truncated}\n", width=WRAP_WIDTH)
        return f"Node ID: {self.node_id}\n{source_text_wrapped}"

    def as_related_node_info(self) -> RelatedNodeInfo:
        """Get node as RelatedNodeInfo."""
        return RelatedNodeInfo(
            node_id=self.node_id,
            node_type=self.get_type(),
            metadata=self.metadata,
            hash=self.hash,
        )


class TextNode(BaseNode, _TextNode):
    text: str = Field(default="", description="Text content of the node.")
    mimetype: str = Field(default="text/plain", description="MIME type of the node content.")
    start_char_idx: int | None = Field(default=None, description="Start char index of the node.")
    end_char_idx: int | None = Field(default=None, description="End char index of the node.")
    text_template: str = Field(
        default=DEFAULT_TEXT_NODE_TMPL,
        description=(
            "Template for how text is formatted, with {content} and " "{metadata_str} placeholders."
        ),
    )
    metadata_template: str = Field(
        default=DEFAULT_METADATA_TMPL,
        description=(
            "Template for how metadata is formatted, with {key} and " "{value} placeholders."
        ),
    )
    metadata_seperator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "TextNode"

    @property
    def hash(self) -> str:
        doc_identity = str(self.text) + str(self.metadata)
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectType.TEXT

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        """Get object content."""
        metadata_str = self.get_metadata_str(mode=metadata_mode).strip()
        if not metadata_str:
            return self.text

        return self.text_template.format(content=self.text, metadata_str=metadata_str).strip()

    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        """Metadata info string."""
        if mode == MetadataMode.NONE:
            return ""

        usable_metadata_keys = set(self.metadata.keys())
        if mode == MetadataMode.LLM:
            for key in self.excluded_llm_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)
        elif mode == MetadataMode.EMBED:
            for key in self.excluded_embed_metadata_keys:
                if key in usable_metadata_keys:
                    usable_metadata_keys.remove(key)

        return self.metadata_seperator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.metadata.items()
                if key in usable_metadata_keys
            ]
        )

    def set_content(self, value: str) -> None:
        """Set the content of the node."""
        self.text = value

    def get_node_info(self) -> dict[str, Any]:
        """Get node info."""
        return {"start": self.start_char_idx, "end": self.end_char_idx}

    def get_text(self) -> str:
        return self.get_content(metadata_mode=MetadataMode.NONE)

    @property
    def node_info(self) -> dict[str, Any]:
        """Deprecated: Get node info."""
        return self.get_node_info()


class NodeWithScore(BaseComponent, _NodeWithScore):
    node: BaseNode
    score: float | None = None

    def __str__(self) -> str:
        score_str = "None" if self.score is None else f"{self.score: 0.3f}"
        return f"{self.node}\nScore: {score_str}\n"

    def get_score(self, raise_error: bool = False) -> float:
        """Get score."""
        if self.score is None:
            if raise_error:
                raise ValueError("Score not set.")
            else:
                return 0.0
        else:
            return self.score

    @classmethod
    def class_name(cls) -> str:
        return "NodeWithScore"

    ##### pass through methods to BaseNode #####
    @property
    def node_id(self) -> str:
        return self.node.node_id

    @property
    def id_(self) -> str:
        return self.node.id_

    @property
    def text(self) -> str:
        if isinstance(self.node, TextNode):
            return self.node.text
        else:
            raise ValueError("Node must be a TextNode to get text.")

    @property
    def metadata(self) -> dict[str, Any]:
        return self.node.metadata

    @property
    def embedding(self) -> list[float] | None:
        return self.node.embedding

    def get_text(self) -> str:
        if isinstance(self.node, TextNode):
            return self.node.get_text()
        else:
            raise ValueError("Node must be a TextNode to get text.")

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.NONE) -> str:
        return self.node.get_content(metadata_mode=metadata_mode)

    def get_embedding(self) -> list[float]:
        return self.node.get_embedding()
