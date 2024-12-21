"""Node Parser Components used in ingestion and query scripts and pipelines."""

import logging
from collections.abc import Callable

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI

from rag.components.models import get_embed_model, get_llm
from rag.llms.openai_like import OpenAILike
from rag.node_parser import (
    AtomicClaimNodeParser,
    SemanticSplitterNodeParser,
    SingleSentenceNodeParser,
)
from rag.node_parser.atomic_claims.prompts import (
    prompt_contains_atomic_claims,
    prompt_extract_atomic_claims,
    system_prompt_extract_atomic_claims,
)

logger = logging.getLogger()


def get_semantic_node_parser(
    buffer_size: int = 3,
    breakpoint_percentile_threshold: int = 90,
    backup_node_parser_threshold: float = 75.0,
    max_chunk_size: int = 256,
    embed_model: HuggingFaceEmbedding | BaseEmbedding | None = None,
    embed_batch_size: int = 20,
    tokenizer_method: str = "openai",
) -> SemanticSplitterNodeParser:
    embed_model = embed_model or get_embed_model(embed_batch_size=embed_batch_size)
    return SemanticSplitterNodeParser.from_defaults(
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        backup_node_parser_threshold=backup_node_parser_threshold,
        max_chunk_size=max_chunk_size,
        embed_model=embed_model,
        tokenizer_method=tokenizer_method,
    )


def get_single_sentence_node_parser(
    sentence_splitter: Callable[[str], list[str]] | None = None,
) -> SingleSentenceNodeParser:
    return SingleSentenceNodeParser.from_defaults(sentence_splitter=sentence_splitter)


def get_atomic_claim_node_parser(
    llm: OpenAILike | OpenAI | None = None,
    system_prompt_template: Callable | None = system_prompt_extract_atomic_claims,
    detect_claims_template: Callable | None = prompt_contains_atomic_claims,
    extract_claims_template: Callable | None = prompt_extract_atomic_claims,
    num_workers: int = 24,
) -> AtomicClaimNodeParser:
    llm = llm or get_llm(temperature=0.1, top_p=1.0)
    return AtomicClaimNodeParser.from_defaults(
        llm=llm,
        system_prompt_template=system_prompt_template,
        detect_claims_template=detect_claims_template,
        extract_claims_template=extract_claims_template,
        num_workers=num_workers,
    )
