"""Pydantic models with built-in LLM generation methods using
vLLM-compatible and OpenAILike models."""

# ruff: noqa: F403, F405
from .base import *
from .claim import *
from .llm_base_model import LLMBaseModel
from .prompts import *

__all__ = [
    "CustomBaseModel",
    "BaseListModel",
    "SimpleClaimList",
    "LLMBaseModel",
    "prompts",
]


def hello() -> str:
    return "Hello from pydantic-utils!"
