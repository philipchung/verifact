"""Helper functions for rq (Redis Queue)."""

# ruff: noqa: F403, F405
from .rq import *

__all__ = ["rq"]


def hello() -> str:
    return "Hello from rq-utils!"
