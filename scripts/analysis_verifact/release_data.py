"""Read the normalized PhysioNet v1.1.0 release for historical analyses."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from utils import load_pandas

_VERDICT_COLUMNS = [
    "model",
    "proposition_id",
    "rater_alias",
    "verdict",
    "reason",
    "reasoning_chain",
    "reasoning_final_answer",
    "reference_id",
]
_RATER_COLUMNS = [
    "model",
    "rater_alias",
    "rater_name",
    "fact_type",
    "retrieval_method",
    "top_n",
    "reference_format",
    "reference_only_admission",
    "deduplicate_text",
]
_PROPOSITION_COLUMNS = [
    "proposition_id",
    "text",
    "subject_id",
    "author_type",
    "proposition_type",
]
_ANNOTATION_COLUMNS = [
    "model",
    "proposition_id",
    "rater_alias",
    "rater_name",
    "subject_id",
    "author_type",
    "proposition_type",
    "fact_type",
    "retrieval_method",
    "top_n",
    "reference_format",
    "reference_only_admission",
    "deduplicate_text",
    "verdict",
    "reason",
    "reasoning_chain",
    "reasoning_final_answer",
    "text",
    "reference_id",
]
_STRING_COLUMNS = {
    "model",
    "proposition_id",
    "rater_alias",
    "rater_name",
    "author_type",
    "proposition_type",
    "fact_type",
    "retrieval_method",
    "reference_format",
    "verdict",
    "reason",
    "reasoning_chain",
    "reasoning_final_answer",
    "text",
    "reference_id",
    "reference",
}
_BOOLEAN_COLUMNS = {"reference_only_admission", "deduplicate_text"}
_INTEGER_COLUMNS = {"subject_id", "top_n", "reference_word_count", "reference_char_count"}
_RELEASE_TO_ANALYSIS_MODEL = {
    "Gemma-3-12B": "Gemma3-12B",
    "Gemma-3-27B": "Gemma3-27B",
    "Qwen-3-32B": "Qwen3-32B",
    "Qwen-3-30B-A3B-Instruct": "Qwen3-30B-A3B-Instruct",
    "Qwen-3-30B-A3B-Thinking": "Qwen3-30B-A3B-Thinking",
    "Llama-8B": "Llama-8B",
    "Llama-70B": "Llama-70B",
    "R1-8B": "R1-8B",
    "R1-70B": "R1-70B",
}
_ANALYSIS_TO_RELEASE_MODEL = {
    analysis: release for release, analysis in _RELEASE_TO_ANALYSIS_MODEL.items()
}
_PARQUET_BATCH_SIZE = 2_048


class ReleaseDataError(ValueError):
    """Raised when release data does not satisfy the analysis adapter contract."""


def _duplicate_values(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        else:
            seen.add(value)
    return sorted(duplicates)


def _release_path(release_dir: Path | str, relative_path: str) -> Path:
    path = Path(release_dir).expanduser() / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"PhysioNet release file does not exist: {path}")
    return path


def _require_columns(df: pd.DataFrame, columns: Sequence[str], table: str) -> None:
    missing = set(columns) - set(df.columns)
    if missing:
        raise ReleaseDataError(f"{table} is missing required columns: {sorted(missing)}")


def _load_required_columns(
    path: Path,
    columns: Sequence[str],
    table: str,
    **kwargs: Any,
) -> pd.DataFrame:
    selector = "columns" if path.suffix == ".parquet" else "usecols"
    try:
        frame = load_pandas(path, **{selector: list(columns)}, **kwargs)
    except (KeyError, ValueError) as exc:
        raise ReleaseDataError(f"{table} is missing one or more required columns") from exc
    _require_columns(frame, columns, table)
    return frame


def _coerce_analysis_types(df: pd.DataFrame) -> pd.DataFrame:
    dtypes: dict[str, str] = {}
    for column in df.columns:
        if column in _STRING_COLUMNS:
            dtypes[column] = "string"
        elif column in _BOOLEAN_COLUMNS:
            dtypes[column] = "boolean"
        elif column in _INTEGER_COLUMNS:
            dtypes[column] = "Int64"
    return df.astype(dtypes)


def _normalize_models(models: str | Iterable[str] | None) -> list[str] | None:
    if models is None:
        return None
    requested = [models] if isinstance(models, str) else list(models)
    known_names = set(_RELEASE_TO_ANALYSIS_MODEL) | set(_ANALYSIS_TO_RELEASE_MODEL)
    unknown = sorted({str(model) for model in requested if model not in known_names})
    if unknown:
        raise ReleaseDataError(
            f"Unknown model name(s): {unknown}. Expected release or historical analysis names."
        )

    release_names = [
        model if model in _RELEASE_TO_ANALYSIS_MODEL else _ANALYSIS_TO_RELEASE_MODEL[model]
        for model in requested
    ]
    duplicates = _duplicate_values(release_names)
    if duplicates:
        raise ReleaseDataError(
            f"Model selection contains duplicate aliases for release model(s): {duplicates}"
        )
    return release_names


def _rewrite_rater_name(rater_name: Any, release_model: Any) -> str:
    expected_prefix = f"model={release_model},"
    if not isinstance(rater_name, str) or not rater_name.startswith(expected_prefix):
        raise ReleaseDataError(
            "rater_configurations.csv contains a rater_name whose model prefix does not "
            f"match its model: {rater_name!r}"
        )
    analysis_model = _RELEASE_TO_ANALYSIS_MODEL[str(release_model)]
    return f"model={analysis_model},{rater_name.removeprefix(expected_prefix)}"


def load_release_annotations(
    release_dir: Path | str,
    *,
    models: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Reconstruct the historical annotation view without reference payload text."""
    selected_models = _normalize_models(models)
    verdict_path = _release_path(release_dir, "verifact/verdicts.parquet")
    rater_path = _release_path(release_dir, "verifact/rater_configurations.csv")
    proposition_path = _release_path(release_dir, "propositions/propositions.csv.gz")

    filters = None
    if selected_models is not None:
        filters = [("model", "in", selected_models)]
    verdicts = _load_required_columns(
        verdict_path,
        _VERDICT_COLUMNS,
        "verdicts.parquet",
        filters=filters,
    )
    if verdicts[["model", "rater_alias", "proposition_id"]].isna().any().any():
        raise ReleaseDataError("verdicts.parquet contains null foreign keys")
    verdict_key = ["proposition_id", "rater_alias"]
    if verdicts.duplicated(verdict_key).any():
        raise ReleaseDataError(
            "verdicts.parquet contains duplicate proposition_id/rater_alias keys"
        )
    unknown_release_models = sorted(set(verdicts["model"]) - set(_RELEASE_TO_ANALYSIS_MODEL))
    if unknown_release_models:
        raise ReleaseDataError(
            f"verdicts.parquet contains unknown model name(s): {unknown_release_models}"
        )

    raters = _load_required_columns(
        rater_path,
        _RATER_COLUMNS,
        "rater_configurations.csv",
        keep_default_na=False,
    )
    if raters[["model", "rater_alias"]].isna().any().any():
        raise ReleaseDataError("rater_configurations.csv contains null key values")
    if raters["rater_alias"].duplicated().any():
        raise ReleaseDataError("rater_configurations.csv contains duplicate rater_alias values")
    unknown_rater_models = sorted(set(raters["model"]) - set(_RELEASE_TO_ANALYSIS_MODEL))
    if unknown_rater_models:
        raise ReleaseDataError(
            f"rater_configurations.csv contains unknown model name(s): {unknown_rater_models}"
        )

    verdict_count = len(verdicts)
    annotations = verdicts.merge(
        raters,
        on=["model", "rater_alias"],
        how="left",
        sort=False,
        validate="many_to_one",
        indicator="_rater_join",
    )
    if len(annotations) != verdict_count:
        raise ReleaseDataError("Rater join changed the annotation row count")
    missing_raters = annotations["_rater_join"].ne("both")
    if missing_raters.any():
        keys = annotations.loc[missing_raters, ["model", "rater_alias"]].drop_duplicates()
        raise ReleaseDataError(
            f"{len(keys)} model/rater_alias key(s) in verdicts.parquet are unresolved"
        )
    annotations = annotations.drop(columns="_rater_join")

    propositions = _load_required_columns(
        proposition_path,
        _PROPOSITION_COLUMNS,
        "propositions.csv.gz",
    )
    if propositions["proposition_id"].isna().any():
        raise ReleaseDataError("propositions.csv.gz contains null proposition_id values")
    if propositions["proposition_id"].duplicated().any():
        raise ReleaseDataError("propositions.csv.gz contains duplicate proposition_id values")

    annotations = annotations.merge(
        propositions,
        on="proposition_id",
        how="left",
        sort=False,
        validate="many_to_one",
        indicator="_proposition_join",
    )
    if len(annotations) != verdict_count:
        raise ReleaseDataError("Proposition join changed the annotation row count")
    missing_propositions = annotations["_proposition_join"].ne("both")
    if missing_propositions.any():
        count = annotations.loc[missing_propositions, "proposition_id"].nunique()
        raise ReleaseDataError(
            f"{count} proposition_id value(s) in verdicts.parquet are unresolved"
        )
    annotations = annotations.drop(columns="_proposition_join")

    annotations["rater_name"] = [
        _rewrite_rater_name(rater_name, model)
        for rater_name, model in annotations[["rater_name", "model"]].itertuples(
            index=False, name=None
        )
    ]
    annotations["model"] = annotations["model"].map(_RELEASE_TO_ANALYSIS_MODEL)
    annotations["reference_format"] = annotations["reference_format"].str.replace(
        "_", " ", regex=False
    )
    annotations = annotations.loc[:, _ANNOTATION_COLUMNS]
    return _coerce_analysis_types(annotations)


def load_release_ground_truth(release_dir: Path | str) -> pd.DataFrame:
    """Load the analysis-ready human ground-truth labels from the release."""
    path = _release_path(release_dir, "propositions/human_verdicts.csv.gz")
    source_columns = [
        "proposition_id",
        "text",
        "author_type",
        "proposition_type",
        "human_gt",
    ]
    ground_truth = _load_required_columns(
        path,
        source_columns,
        "human_verdicts.csv.gz",
    )
    if ground_truth["proposition_id"].isna().any():
        raise ReleaseDataError("human_verdicts.csv.gz contains null proposition_id values")
    if ground_truth["proposition_id"].duplicated().any():
        raise ReleaseDataError("human_verdicts.csv.gz contains duplicate proposition_id values")
    if ground_truth["human_gt"].isna().any():
        raise ReleaseDataError("human_verdicts.csv.gz contains null human_gt values")
    ground_truth = ground_truth.rename(columns={"human_gt": "verdict"}).assign(
        rater_name="human_gt"
    )
    columns = [
        "proposition_id",
        "text",
        "author_type",
        "proposition_type",
        "rater_name",
        "verdict",
    ]
    return _coerce_analysis_types(ground_truth.loc[:, columns])


def _normalize_reference_ids(reference_ids: str | Iterable[str] | None) -> list[str] | None:
    if reference_ids is None:
        return None
    requested = [reference_ids] if isinstance(reference_ids, str) else list(reference_ids)
    if any(pd.isna(reference_id) for reference_id in requested):
        raise ReleaseDataError("reference_ids contains a null value")
    normalized = [str(reference_id) for reference_id in requested]
    duplicates = _duplicate_values(normalized)
    if duplicates:
        raise ReleaseDataError(f"reference_ids contains duplicate values: {duplicates}")
    return normalized


def _scan_reference_payloads(
    release_dir: Path | str,
    reference_ids: Sequence[str] | None,
) -> Iterable[dict[str, list[Any]]]:
    path = _release_path(release_dir, "verifact/reference_payloads.parquet")
    parquet_file = pq.ParquetFile(path)
    required = {"reference_id", "reference"}
    missing = required - set(parquet_file.schema_arrow.names)
    if missing:
        raise ReleaseDataError(
            f"reference_payloads.parquet is missing required columns: {sorted(missing)}"
        )
    selected = set(reference_ids) if reference_ids is not None else None
    for batch in parquet_file.iter_batches(
        columns=["reference_id", "reference"],
        batch_size=_PARQUET_BATCH_SIZE,
        use_threads=False,
    ):
        values = batch.to_pydict()
        if selected is None:
            yield values
            continue
        matches = [
            (reference_id, reference)
            for reference_id, reference in zip(
                values["reference_id"], values["reference"], strict=True
            )
            if reference_id in selected
        ]
        if matches:
            yield {
                "reference_id": [reference_id for reference_id, _ in matches],
                "reference": [reference for _, reference in matches],
            }


def load_reference_lengths(
    release_dir: Path | str,
    *,
    reference_ids: str | Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute reference lengths in bounded-memory Parquet batches."""
    requested = _normalize_reference_ids(reference_ids)
    records: list[tuple[str, int, int]] = []
    found: set[str] = set()
    for batch in _scan_reference_payloads(release_dir, requested):
        for reference_id, reference in zip(batch["reference_id"], batch["reference"], strict=True):
            if reference_id is None or reference is None:
                raise ReleaseDataError("reference_payloads.parquet contains null values")
            reference_id = str(reference_id)
            if reference_id in found:
                raise ReleaseDataError(
                    f"reference_payloads.parquet contains duplicate reference_id: {reference_id}"
                )
            found.add(reference_id)
            text = str(reference)
            records.append((reference_id, len(text.split()), len(text)))

    if requested is not None:
        missing = sorted(set(requested) - found)
        if missing:
            raise ReleaseDataError(
                f"reference_payloads.parquet is missing requested reference_id value(s): {missing}"
            )
    lengths = pd.DataFrame.from_records(
        records,
        columns=["reference_id", "reference_word_count", "reference_char_count"],
    )
    return _coerce_analysis_types(lengths)


def _load_selected_reference_payloads(
    release_dir: Path | str, reference_ids: Sequence[str]
) -> pd.DataFrame:
    records: list[tuple[str, str]] = []
    found: set[str] = set()
    for batch in _scan_reference_payloads(release_dir, reference_ids):
        for reference_id, reference in zip(batch["reference_id"], batch["reference"], strict=True):
            if reference_id is None or reference is None:
                raise ReleaseDataError("reference_payloads.parquet contains null values")
            reference_id = str(reference_id)
            if reference_id in found:
                raise ReleaseDataError(
                    f"reference_payloads.parquet contains duplicate reference_id: {reference_id}"
                )
            found.add(reference_id)
            records.append((reference_id, str(reference)))
    missing = sorted(set(reference_ids) - found)
    if missing:
        raise ReleaseDataError(
            f"reference_payloads.parquet is missing requested reference_id value(s): {missing}"
        )
    payloads = pd.DataFrame.from_records(records, columns=["reference_id", "reference"])
    return _coerce_analysis_types(payloads)


def attach_reference_payloads(
    annotations: pd.DataFrame,
    release_dir: Path | str,
) -> pd.DataFrame:
    """Attach only the reference payloads needed by the supplied annotation rows."""
    if "reference_id" not in annotations.columns:
        raise ReleaseDataError("annotations must contain a reference_id column")
    if "reference" in annotations.columns:
        raise ReleaseDataError("annotations already contains a reference column")
    if annotations["reference_id"].isna().any():
        raise ReleaseDataError("annotations contains null reference_id values")
    if annotations.empty:
        result = annotations.copy()
        result["reference"] = pd.Series(index=result.index, dtype="string")
        return result

    reference_ids = annotations["reference_id"].astype("string").drop_duplicates().tolist()
    payloads = _load_selected_reference_payloads(release_dir, reference_ids)
    original_index = annotations.index.copy()
    result = annotations.merge(
        payloads,
        on="reference_id",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(result) != len(annotations):
        raise ReleaseDataError("Reference-payload join changed the annotation row count")
    if result["reference"].isna().any():
        raise ReleaseDataError("Reference-payload join left unresolved reference_id values")
    result.index = original_index
    return result
