# ruff: noqa: F403, F405
from .hospital_course import *
from .prompts import *

__all__ = ["hospital_course", "prompts"]


def hello() -> str:
    return "Hello from llm-writer!"
