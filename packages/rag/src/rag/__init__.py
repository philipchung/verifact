"""Commonly used static variables (constants) in the RAG package."""

from typing import Final

# Node Categories representing different units of text information
CLAIM_NODE: Final[str] = "claim"
SENTENCE_NODE: Final[str] = "sentence"
SEMANTIC_NODE: Final[str] = "semantic"
MIMIC_NODE: Final[str] = "mimic-iii note"

# Brief Hospital Course (BHC) Author Type
LLM_AUTHOR: Final[str] = "llm"
HUMAN_AUTHOR: Final[str] = "human"

# RAG Retrieval Method
DENSE: Final[str] = "dense"
HYBRID: Final[str] = "hybrid"
RERANK: Final[str] = "rerank"

# Reference Context Format
SCORE: Final[str] = "score"
ABSOLUTE_TIME: Final[str] = "absolute_time"
RELATIVE_TIME: Final[str] = "relative_time"


def hello() -> str:
    return "Hello from rag!"
