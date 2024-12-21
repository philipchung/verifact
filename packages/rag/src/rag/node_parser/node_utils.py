import logging
from collections.abc import Callable
from typing import Any

import nltk
import tiktoken
from llama_index.core.node_parser.node_utils import (
    IdFuncCallable,
    default_id_func,
)
from llama_index.core.schema import NodeRelationship
from llama_index.core.utils import truncate_text
from transformers import AutoTokenizer
from utils import create_uuid_from_string

from rag.schema.node_schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    TextNode,
)

logger = logging.getLogger(__name__)


def node_id_from_string(text: str) -> str:
    return str(create_uuid_from_string(text))


def build_nodes_from_splits(
    text_splits: list[str],
    document: BaseNode,
    ref_doc: BaseNode | None = None,
    id_func: IdFuncCallable | None = None,
) -> list[BaseNode]:
    """Build nodes from splits. Modified implementation to support nodes with `embeddings`
    and `embedding` fields (which allow multiple vectors per node).

    This method does not compute embeddings for the text splits. It simply copies the
    existing document embeddings to the new nodes for the text splits if document
    embeddings exist.

    Original Implementation in:
    https://github.com/run-llama/llama_index/blob/main/llama_index/node_parser/node_utils.py
    """
    ref_doc = ref_doc or document
    id_func = id_func or default_id_func
    nodes: list[BaseNode] = []
    for i, text_chunk in enumerate(text_splits):
        logger.debug(f"> Adding chunk: {truncate_text(text_chunk, 50)}")

        if isinstance(document, ImageDocument):
            image_node = ImageNode(
                id_=id_func(i, document),
                text=text_chunk,
                embeddings=document.embeddings,
                embedding=document.embedding,
                image=document.image,
                image_path=document.image_path,
                image_url=document.image_url,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
                relationships={NodeRelationship.SOURCE: ref_doc.as_related_node_info()},
            )
            nodes.append(image_node)  # type: ignore
        elif isinstance(document, Document | TextNode):
            node = TextNode(
                id_=id_func(i, document),
                text=text_chunk,
                embeddings=document.embeddings,
                embedding=document.embedding,
                excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                metadata_seperator=document.metadata_seperator,
                metadata_template=document.metadata_template,
                text_template=document.text_template,
                relationships={NodeRelationship.SOURCE: ref_doc.as_related_node_info()},
            )
            nodes.append(node)
        else:
            raise ValueError(f"Unknown document type: {type(document)}")

    return nodes


nltk_tokenizer = nltk.tokenize.PunktSentenceTokenizer()


def nltk_split_sentences(text: str) -> list[str]:
    """Tokenize text using NLTK PunktSentenceTokenizer."""
    spans = list(nltk_tokenizer.span_tokenize(text))
    sentences = []
    for i, span in enumerate(spans):
        # Get start and end indices of each sentence
        # For end index, use the start of the next span if available
        start = span[0]
        end = spans[i + 1][0] if i < len(spans) - 1 else len(text)
        sentences.append(text[start:end])
    return sentences


def count_sentences(
    text: str, sentence_splitter: Callable[[str], list[str]] = nltk_split_sentences
) -> int:
    return len(sentence_splitter(text))


def count_tokens(text: str, tokenizer: Any = "openai") -> int:
    """Count the number of tokens in a text string."""
    # User-defined tokenizer function
    if isinstance(tokenizer, Callable):
        return tokenizer(text)
    # OpenAI tiktoken tokenizer
    elif tokenizer.lower() in ("openai", "tiktoken", "o200k_base"):
        encoding = tiktoken.get_encoding(encoding_name="o200k_base")
        return len(encoding.encode(text=text))
    elif tokenizer.lower() in ("cl100k_base"):
        encoding = tiktoken.get_encoding(encoding_name="cl100k_base")
        return len(encoding.encode(text=text))
    # Llama tokenizer
    elif tokenizer.lower() in ("llama", "llama3", "llama3.1"):
        # Disable tranfsformers verbosity
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        # Count Tokens
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
        return len(tokenizer.encode(text=text, max_length=None, truncation=False, padding=False))
    # Otherwise assume Huggingface tokenizer
    elif len(tokenizer.split("/")) == 2:
        # Disable tranfsformers verbosity
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()

        # Count Tokens
        # NOTE: limitation here is that some tokenizers impose a length limit
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, max_length=None)
        print("tokenizer is fast: ", tokenizer.is_fast)
        return len(tokenizer.encode(text=text, max_length=None, truncation=False, padding=False))
    else:
        raise ValueError("Invalid tokenizer method for Node Parser.")
