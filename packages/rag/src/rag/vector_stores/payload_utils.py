"""Utilities for converting node to metadata dict and vice versa when saving to vector store.

These have been modified to customize the payload fields on Qdrant Points.
"""

import json
from typing import Any

from llama_index.core.vector_stores.utils import _validate_is_flat_dict

from rag.schema import BaseNode, ImageNode, IndexNode, TextNode

DEFAULT_TEXT_KEY = "text"
DEFAULT_EMBEDDING_KEY = "embedding"
DEFAULT_DOC_ID_KEY = "doc_id"


def node_to_qdrant_payload_dict(
    node: BaseNode,
) -> dict[str, Any]:
    """Custom logic for converting LlamaIndex Node into Qdrant Payload dict."""
    node_dict = node.dict()
    metadata: dict[str, Any] = node_dict.get("metadata", {})

    ## Construct Payload Dict
    # Since we pop items from node_dict and metadata, the items are removed
    # from those original dicts in the process
    payload = {
        "node_id": node_dict.pop("id_", None),
        "text": node_dict.pop("text", None),
        "relationships": node_dict.pop("relationships", {}),
        "node_monotonic_idx": metadata.pop("node_monotonic_idx", None),
        "node_kind": metadata.pop("node_kind", None),
        "created_at": metadata.pop("created_at", None),
        "parent_chunk_id": metadata.pop("parent_chunk_id", None),
        "source_document_id": metadata.pop("source_document_id", None),
        "parent_text": metadata.pop("parent_text", None),
        # MIMIC-III specific fields
        "ROW_ID": metadata.pop("ROW_ID", None),
        "SUBJECT_ID": metadata.pop("SUBJECT_ID", None),
        "HADM_ID": metadata.pop("HADM_ID", None),
        "CHARTDATE": metadata.pop("CHARTDATE", None),
        "CHARTTIME": metadata.pop("CHARTTIME", None),
        "STORETIME": metadata.pop("STORETIME", None),
        "CATEGORY": metadata.pop("CATEGORY", None),
        "DESCRIPTION": metadata.pop("DESCRIPTION", None),
        "CGID": metadata.pop("CGID", None),
        "metadata": metadata,
    }

    # remove embeddings from node_dict (they are separately represented as Qdrant Vector)
    node_dict["embedding"] = None
    node_dict["embeddings"] = {}

    # dump remainder of node_dict to json string and add to payload
    payload |= {
        "_node_content": json.dumps(node_dict),
        "_node_type": node.class_name(),
    }
    return payload


def qdrant_payload_dict_to_node(payload: dict) -> dict[str, Any]:
    """Custom logic for convering Qdrant Payload dict into LlamaIndex Node."""
    # Rebuild node from _node_content & _node_type field in payload
    node_json = payload.get("_node_content")
    node_type = payload.get("_node_type")
    if node_json is None:
        raise ValueError("Node content not found in metadata dict.")

    node: BaseNode
    if node_type == IndexNode.class_name():
        node = IndexNode.parse_raw(node_json)
    elif node_type == ImageNode.class_name():
        node = ImageNode.parse_raw(node_json)
    else:
        node = TextNode.parse_raw(node_json)

    # Populate node with payload data stored at top level
    node.node_id = payload.get("node_id")
    node.text = payload.get("text")
    node.relationships = payload.get("relationships", {})
    node.metadata = payload.get("metadata", {}) | {
        "node_monotonic_idx": payload.get("node_monotonic_idx"),
        "node_kind": payload.get("node_kind"),
        "created_at": payload.get("created_at"),
        "parent_chunk_id": payload.get("parent_chunk_id"),
        "source_document_id": payload.get("source_document_id"),
        "parent_text": payload.get("parent_text"),
        "ROW_ID": payload.get("ROW_ID"),
        "SUBJECT_ID": payload.get("SUBJECT_ID"),
        "HADM_ID": payload.get("HADM_ID"),
        "CHARTDATE": payload.get("CHARTDATE"),
        "CHARTTIME": payload.get("CHARTTIME"),
        "STORETIME": payload.get("STORETIME"),
        "CATEGORY": payload.get("CATEGORY"),
        "DESCRIPTION": payload.get("DESCRIPTION"),
        "mimic_row_id": payload.get("mimic_row_id"),
        "CGID": payload.get("CGID"),
    }
    return node


def node_to_metadata_dict(
    node: BaseNode,
    remove_text: bool = False,
    text_field: str = DEFAULT_TEXT_KEY,
    flat_metadata: bool = False,
) -> dict[str, Any]:
    """Common logic for saving Node data into metadata dict."""
    node_dict = node.dict()
    metadata: dict[str, Any] = node_dict.get("metadata", {})

    if flat_metadata:
        _validate_is_flat_dict(metadata)

    # store entire node as json string - some minor text duplication
    if remove_text:
        node_dict[text_field] = ""

    # remove embeddings from node_dict
    node_dict["embedding"] = None
    node_dict["embeddings"] = None

    # dump remainder of node_dict to json string
    metadata["_node_content"] = json.dumps(node_dict)
    metadata["_node_type"] = node.class_name()

    # store ref doc id at top level to allow metadata filtering
    # kept for backwards compatibility, will consolidate in future
    metadata["document_id"] = node.ref_doc_id or "None"  # for Chroma
    metadata["doc_id"] = node.ref_doc_id or "None"  # for Pinecone, Qdrant, Redis
    metadata["ref_doc_id"] = node.ref_doc_id or "None"  # for Weaviate

    # add text to easily search only original text
    metadata["text"] = node.text
    return metadata


def metadata_dict_to_node(metadata: dict, text: str | None = None) -> BaseNode:
    """Common logic for loading Node data from metadata dict."""
    node_json = metadata.get("_node_content")
    node_type = metadata.get("_node_type")
    if node_json is None:
        raise ValueError("Node content not found in metadata dict.")

    node: BaseNode
    if node_type == IndexNode.class_name():
        node = IndexNode.parse_raw(node_json)
    elif node_type == ImageNode.class_name():
        node = ImageNode.parse_raw(node_json)
    else:
        node = TextNode.parse_raw(node_json)

    if text is not None:
        node.set_content(text)

    return node
