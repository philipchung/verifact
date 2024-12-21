from llama_index.core.constants import DATA_KEY, TYPE_KEY

from .base_node import BaseNode
from .node_schema import Document, ImageDocument, ImageNode, IndexNode, TextNode


def doc_to_json(doc: BaseNode) -> dict:
    return {
        DATA_KEY: doc.dict(),
        TYPE_KEY: doc.get_type(),
    }


def json_to_doc(doc_dict: dict) -> BaseNode:
    """Converts JSON representation of node back to object.

    Llama-index does not support Multivector/Named Embeddings in nodes. To add support
    for this functionality in the docstore and ingestion cache, we need to make some slight
    modifications so Llama-index is aware of this new node type.

    Modified From: llama_index.storage.docstore.utils
    """
    doc_type = doc_dict[TYPE_KEY]
    data_dict = doc_dict[DATA_KEY]
    doc: BaseNode

    if doc_type == Document.get_type():
        doc = Document.parse_obj(data_dict)
    elif doc_type == ImageDocument.get_type():
        doc = ImageDocument.parse_obj(data_dict)
    elif doc_type == TextNode.get_type():
        doc = TextNode.parse_obj(data_dict)
    elif doc_type == ImageNode.get_type():
        doc = ImageNode.parse_obj(data_dict)
    elif doc_type == IndexNode.get_type():
        doc = IndexNode.parse_obj(data_dict)
    else:
        raise ValueError(f"Unknown doc type: {doc_type}")

    return doc
