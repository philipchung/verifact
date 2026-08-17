"""Interrater analysis."""

# ruff: noqa: F403, F405
from .binarize import *
from .classification_metrics import *
from .constants import *
from .interrater import *
from .melt_wide_to_long import *
from .metric_bunch import *
from .type_utils import *

__all__ = [
    "binarize",
    "constants",
    "interrater",
    "melt_wide_to_long",
    "metric_bunch",
    "classification_metrics",
    "type_utils",
]


def hello() -> str:
    return "Hello from irr_metrics!"
