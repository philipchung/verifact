from typing import Any

import pandas as pd
from pandas.api.types import CategoricalDtype

from irr_metrics.constants import (
    NOT_ADDRESSED,
    NOT_SUPPORTED,
    NOT_SUPPORTED_OR_ADDRESSED,
    SUPPORTED,
)

COL_DTYPE_MAPPING: dict = {
    # Data Types for Human Clinician Verdicts
    "proposition_id": "string",
    "text": "string",
    "author_type": "string",
    "proposition_type": "string",
    "fact_type": "string",
    "subject_id": "Int64",
    "rater1": "string",
    "rater2": "string",
    "rater3": "string",
    "verdict1": "string",
    "verdict2": "string",
    "verdict3": "string",
    "uncertain_flag1": "boolean",
    "uncertain_flag2": "boolean",
    "uncertain_flag3": "boolean",
    "comment1": "string",
    "comment2": "string",
    "comment3": "string",
    "round1_num_raters_agree": "Int64",
    "round1_majority_vote": "string",
    "rater4": "string",
    "rater5": "string",
    "verdict4": "string",
    "verdict5": "string",
    "comment4": "string",
    "comment5": "string",
    "round2_decision": "string",
    "adjudicated_verdict": "string",
    "adjudicated_comment": "string",
    "human_gt": "string",
    # Additional Data Types for VeriFact AI System Verdicts
    "filename": "string",
    "rater_name": "string",
    "rater_alias": "string",
    "verdict": "string",
    "reason": "string",
    "reasoning_chain": "string",
    "reasoning_final_answer": "string",
    "reference": "string",
    "retrieval_method": "string",
    "top_n": "Int64",
    "reference_format": "string",
    "reference_only_admission": "boolean",
    "deduplicate_text": "boolean",
    # Metrics Data Types
    "metric_name": "string",
    "value": "Float64",
    "confidence_level": "Float64",
    "ci_lower": "Float64",
    "ci_upper": "Float64",
    "p_value": "Float64",
    "bootstrap_iterations": "Int64",
}

RETRIEVAL_METHOD_CATEGORICAL_DTYPE = CategoricalDtype(
    ["Dense", "Sparse", "Hybrid", "Rerank"], ordered=True
)
REFERENCE_CONTEXT_CATEGORICAL_DTYPE = CategoricalDtype(
    ["Score", "Absolute Time", "Relative Time"], ordered=True
)
RETRIEVAL_SCOPE_CATEGORICAL_DTYPE = CategoricalDtype(["Yes", "No"], ordered=True)


def coerce_types(
    df: pd.DataFrame, cols: dict[str, Any] | list[str] | str | None = None
) -> pd.DataFrame:
    """Coerce DataFrame Column Types."""
    # Select Column-Data Type Mapping
    match type(cols):
        case dict():
            col_mapping = cols | COL_DTYPE_MAPPING
        case list():
            col_mapping = {col: COL_DTYPE_MAPPING[col] for col in cols}
        case str():
            col_mapping = {cols: COL_DTYPE_MAPPING[cols]}
        case None | _:
            col_mapping = COL_DTYPE_MAPPING
    # Apply Column-Data Type Mapping
    dtype_mapping = {k: v for k, v in col_mapping.items() if k in df.columns}
    df = df.astype(dtype_mapping)
    return df


def coerce_categorical_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce and Reformat DataFrame Categorical Column Types."""
    # Coerce Categorical Columns and Adjust Values
    if "retrieval_method" in df.columns:

        def _coerce_retrieval_method(x: str) -> str:
            if x in ("rerank", "dense", "sparse", "hybrid"):
                return x.title()
            else:
                return x

        df = df.assign(
            retrieval_method=df.retrieval_method.apply(_coerce_retrieval_method).astype(
                RETRIEVAL_METHOD_CATEGORICAL_DTYPE
            )
        )
    if "reference_format" in df.columns:

        def _coerce_reference_format(x: str) -> str:
            if x in ("score", "absolute time", "relative time", "absolute_time", "relative_time"):
                return x.replace("_", " ").title()
            else:
                return x

        df = df.assign(
            reference_format=df.reference_format.apply(_coerce_reference_format).astype(
                REFERENCE_CONTEXT_CATEGORICAL_DTYPE
            )
        )

    if "reference_only_admission" in df.columns:

        def _coerce_reference_only_admission(x: Any) -> str:
            if isinstance(x, bool):
                return "Yes" if x else "No"
            else:
                return x

        df = df.assign(
            reference_only_admission=df.reference_only_admission.apply(
                _coerce_reference_only_admission
            ).astype(RETRIEVAL_SCOPE_CATEGORICAL_DTYPE)
        )

    return df


verdict_label_dtype = CategoricalDtype(
    categories=[SUPPORTED, NOT_SUPPORTED, NOT_ADDRESSED, NOT_SUPPORTED_OR_ADDRESSED],
    ordered=True,
)

metric_category_dtype = CategoricalDtype(
    categories=["TPR", "TNR", "PPV", "NPV", "TP", "TN", "FP", "FN", "Support"], ordered=True
)
