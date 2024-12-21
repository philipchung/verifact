# ruff: noqa: F403, F405
from .judge import *
from .prompts import *

__all__ = ["judge", "prompts"]


def hello() -> str:
    return "Hello from proposition-validity!"
