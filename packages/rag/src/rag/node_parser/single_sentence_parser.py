"""Node Parser that splits text into individual sentences and returns a node for each sentence.

This NOT the same as the SentenceSplitter NodeParser in the LlamaIndex package
which can be found at llama_index.core.node_parser.text.sentence.SentenceSplitter.
The LlamaIndex SentenceSplitter NodeParser is used to split text into chunks using
paragraph and sentence boundaries, but does not create a node for each sentence.
"""

import logging
from collections.abc import Callable
from typing import Any

from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import MetadataMode, NodeRelationship
from llama_index.core.utils import get_tqdm_iterable
from utils import convert_nan_nat_to_none, convert_timestamps_to_iso, get_utc_time

from rag import MIMIC_NODE, SENTENCE_NODE
from rag.node_parser.node_utils import (
    count_sentences,
    count_tokens,
    mimic_node_id_fn,
    nltk_split_sentences,
)
from rag.schema import BaseNode, Document, TextNode

logger = logging.getLogger(__name__)


class SingleSentenceNodeParser(NodeParser):
    """Sentence node parser that splits text by sentences and returns a node for
    each sentence. This splitter uses the NLTK PunktSentenceTokenizer to identify
    sentence boundaries and split text.
    """

    sentence_splitter: Callable[[str], list[str]] | None = Field(
        default_factory=nltk_split_sentences,
        description="A function that defines how to split text into sentences. "
        "The function should accept a string argument and return a list of string.",
        exclude=True,
    )
    include_parent_text: bool = Field(
        default=False, description="Flag for whether to include parent text in metadata."
    )
    id_func: Callable[[int, Document], str] = Field(
        default=mimic_node_id_fn,
        description="A function that generates a unique ID for each node."
        "Function must take 3 arguments `i` (int), `node` (BaseNode), and `text` (str)",
        exclude=True,
    )
    node_kind: str = Field(default=SENTENCE_NODE, description="Node category.")

    @classmethod
    def class_name(cls) -> str:
        return "SingleSentenceNodeParser"

    @classmethod
    def from_defaults(
        cls,
        sentence_splitter: Callable[[str], list[str]] | None = None,
        include_parent_text: bool = True,
        node_kind: str = SENTENCE_NODE,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: CallbackManager | None = None,
        id_func: Callable[[int, Document], str] | None = None,
    ) -> "SingleSentenceNodeParser":
        sentence_splitter = sentence_splitter or nltk_split_sentences
        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or mimic_node_id_fn
        return cls(
            sentence_splitter=sentence_splitter,
            include_parent_text=include_parent_text,
            node_kind=node_kind,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )

    def _count_tokens(self, text: str) -> int:
        """Count number of tokens in a text string."""
        return count_tokens(text)

    def _count_sentences(self, text: str) -> int:
        """Count number of sentences in a text string."""
        return count_sentences(text)

    def _build_nodes_from_sentences_list(
        self, sentences: list[str], original_node: BaseNode, **kwargs
    ) -> list[BaseNode]:
        """Builds nodes from claims list."""
        # Create New Node for Each Sentence
        output_nodes: list[TextNode] = []
        for i, sentence in enumerate(sentences):
            node_text = sentence
            parent_text = original_node.get_content(MetadataMode.NONE)
            # Create Unique ID for New Node - This will be unique for each patient note,
            # author_type, proposition_type/node_kind, and extracted text
            node_id = self.id_func(
                text=node_text,
                subject_id=kwargs.get("subject_id", ""),
                row_id=kwargs.get("row_id", ""),
                author_type=kwargs.get("author_type", ""),
                node_kind=getattr(self, "node_kind", ""),
            )
            metadata = {
                # Monotonically increasing index (relative position in original node)
                "node_monotonic_idx": i,
                # Node Kind
                "node_kind": self.node_kind,
                # Node Creation Time
                "created_at": get_utc_time(),
                # Metadata from Original Node
                "parent_text": parent_text if self.include_parent_text else None,
                "parent_chunk_id": original_node.node_id,
                "source_document_id": original_node.source_node.node_id
                if original_node.source_node
                else original_node.node_id,
            }
            # Set Parent Node Relationship
            relationships = {NodeRelationship.PARENT: original_node.as_related_node_info()}
            # Set Source Node Relationship
            if isinstance(original_node, Document):
                relationships |= {NodeRelationship.SOURCE: original_node.as_related_node_info()}
            elif original_node.relationships.get(NodeRelationship.SOURCE) is not None:
                relationships |= {
                    NodeRelationship.SOURCE: original_node.relationships[NodeRelationship.SOURCE]
                }

            new_node = TextNode(
                id=node_id,
                text=node_text,
                metadata=metadata,
                relationships=relationships,
                # Carry over properties from original node
                embeddings=original_node.embeddings,
                embedding=original_node.embedding,
                excluded_embed_metadata_keys=original_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=original_node.excluded_llm_metadata_keys,
                metadata_seperator=original_node.metadata_seperator,
                metadata_template=original_node.metadata_template,
                text_template=original_node.text_template,
            )
            output_nodes.append(new_node)
        return output_nodes

    def _parse_nodes(
        self, nodes: list[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> list[BaseNode]:
        """Parses input nodes into sentences and creates a new node for each sentence."""
        queue_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing Node to Sentences")

        output_nodes = []
        for node in queue_with_progress:
            # Extract Sentences from Node Text
            node_text = node.get_content(MetadataMode.NONE)
            sentences: list[str] = self.sentence_splitter(node_text)
            # Build a New Node for each Sentence
            sentence_nodes = self._build_nodes_from_sentences_list(sentences, node, **kwargs)
            output_nodes.extend(sentence_nodes)
        return output_nodes

    def _postprocess_parsed_nodes(
        self, nodes: list[BaseNode], parent_doc_map: dict[str, Document]
    ) -> list[BaseNode]:
        """Update metadata and relationships for each node.

        Args:
            nodes (list[BaseNode]): List of nodes that have been created by node parser.
            parent_doc_map (dict[str, Document]): List of original documents.

        Returns:
            list[BaseNode]: List of nodes with updated metadata and relationships.
        """
        for i, node in enumerate(nodes):
            parent_doc = parent_doc_map.get(node.parent_node.node_id, None)

            if parent_doc is not None:
                self._update_node_start_end_char_idx(node, parent_doc)

                if self.include_metadata:
                    node = self._update_node_metadata(
                        node,
                        parent_doc,
                        start_char_idx=node.start_char_idx,
                        end_char_idx=node.end_char_idx,
                    )

            if self.include_prev_next_rel:
                self._update_prev_next_relationships(i, node, nodes)
        return nodes

    def _update_node_start_end_char_idx(
        self, node: BaseNode, parent_doc: Document, **kwargs: Any
    ) -> BaseNode:
        """Update the start and end character index for a node based on the parent document."""
        node_text = node.get_content(MetadataMode.NONE)
        start_char_idx = parent_doc.text.find(node_text)
        if start_char_idx >= 0:
            node.start_char_idx = start_char_idx
            node.end_char_idx = start_char_idx + len(node_text)
        return node

    def _update_node_metadata(
        self, node: BaseNode, parent_doc: Document, **kwargs: Any
    ) -> BaseNode:
        """Update the metadata of a node, including text length, token count, sentence count,
        and character positions.

        If the original document is a PDF, also include in the metadata the page number(s)
        in the document.
        """
        node_text = node.get_content(MetadataMode.NONE)
        node.metadata |= {
            "num_characters": len(node_text),
            "num_tokens": self._count_tokens(node_text),
            "num_sentences": self._count_sentences(node_text),
            "start_char_idx": node.start_char_idx,
            "end_char_idx": node.end_char_idx,
        }

        # If original document is MIMIC-III note, include some original metadata
        source_node_relationship = node.relationships.get(NodeRelationship.SOURCE)
        if (
            source_node_relationship
            and source_node_relationship.metadata.get("node_kind") == MIMIC_NODE
        ):
            node.metadata |= {
                "ROW_ID": source_node_relationship.metadata.get("ROW_ID"),
                "SUBJECT_ID": source_node_relationship.metadata.get("SUBJECT_ID"),
                "HADM_ID": source_node_relationship.metadata.get("HADM_ID"),
                "CHARTDATE": source_node_relationship.metadata.get("CHARTDATE"),
                "CHARTTIME": source_node_relationship.metadata.get("CHARTTIME"),
                "STORETIME": source_node_relationship.metadata.get("STORETIME"),
                "CATEGORY": source_node_relationship.metadata.get("CATEGORY"),
                "DESCRIPTION": source_node_relationship.metadata.get("DESCRIPTION"),
                "CGID": source_node_relationship.metadata.get("CGID"),
            }
            node.metadata = {k: convert_nan_nat_to_none(v) for k, v in node.metadata.items()}
            node.metadata = {k: convert_timestamps_to_iso(v) for k, v in node.metadata.items()}
        return node

    def _update_prev_next_relationships(
        self, i: int, node: BaseNode, nodes: list[BaseNode], **kwargs: Any
    ) -> BaseNode:
        """Update prev/next relationships for a node."""
        # establish prev/next relationships if nodes share the same source_node
        if (
            i > 0
            and node.source_node
            and nodes[i - 1].source_node
            and nodes[i - 1].source_node.node_id == node.source_node.node_id
        ):
            node.relationships[NodeRelationship.PREVIOUS] = nodes[i - 1].as_related_node_info()
        if (
            i < len(nodes) - 1
            and node.source_node
            and nodes[i + 1].source_node
            and nodes[i + 1].source_node.node_id == node.source_node.node_id
        ):
            node.relationships[NodeRelationship.NEXT] = nodes[i + 1].as_related_node_info()
        return node
