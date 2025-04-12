"""Commonly used static variables (constants) in the IRR Metrics package."""

from typing import Final

# Verdict Labels
SUPPORTED: Final[str] = "Supported"
NOT_SUPPORTED: Final[str] = "Not Supported"
NOT_ADDRESSED: Final[str] = "Not Addressed"
NOT_SUPPORTED_OR_ADDRESSED: Final[str] = "Not Supported or Addressed"
VERDICT_LABELS: tuple = (SUPPORTED, NOT_SUPPORTED, NOT_ADDRESSED)
BINARIZED_VERDICT_LABELS: tuple = (SUPPORTED, NOT_SUPPORTED_OR_ADDRESSED)
ALL_VERDICT_LABELS: tuple = (SUPPORTED, NOT_SUPPORTED, NOT_ADDRESSED, NOT_SUPPORTED_OR_ADDRESSED)

# Input Text Types (Author Type x Node Kind)
LLM_CLAIM: Final[str] = "llm_claim"
LLM_SENTENCE: Final[str] = "llm_sentence"
HUMAN_CLAIM: Final[str] = "human_claim"
HUMAN_SENTENCE: Final[str] = "human_sentence"

# Different Stratifications of Input Text Types
ALL_SAMPLES: Final[str] = "all_samples"
STRATA: tuple = (LLM_CLAIM, LLM_SENTENCE, HUMAN_CLAIM, HUMAN_SENTENCE)
ALL_STRATA: tuple = (ALL_SAMPLES,) + STRATA
