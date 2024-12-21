"""Interrater analysis."""

# ruff: noqa: F403, F405
from .constants import *
from .interrater import *
from .metric_bunch import *

__all__ = ["constants", "interrater", "metric_bunch"]


def hello() -> str:
    return "Hello from irr_metrics!"
