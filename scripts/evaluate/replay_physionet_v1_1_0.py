"""Replay manuscript judge inference with exact PhysioNet v1.1.0 inputs."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import pyarrow.parquet as pq
import typer
from dotenv import dotenv_values, load_dotenv
from llm_judge import Judge
from rag.components import get_aux_llm, get_llm
from utils import load_environment

MODEL_CONFIGURATION_COLUMNS = [
    "model",
    "author_type",
    "proposition_type",
    "fact_type",
    "retrieval_method",
    "top_n",
    "reference_format",
    "reference_only_admission",
    "deduplicate_text",
]
REQUIRED_PROFILE_KEYS = {
    "VERIFACT_MODEL",
    "VERIFACT_OUTPUT_SUBDIR",
    "IS_REASONING_MODEL",
    "LLM_MODEL_NAME",
    "TOKENIZER_MODEL_NAME",
    "LLM_MAX_MODEL_LEN",
    "MAIN_VLLM_USE_V1",
    "EXTRA_ARGS",
}
REFERENCE_BATCH_SIZE = 2_048
LARGE_RUN_THRESHOLD = 1_000


class ReplayConfigurationError(ValueError):
    """Raised when a release, model profile, or replay selection is inconsistent."""


@dataclass(frozen=True)
class ModelProfile:
    path: Path
    model: str
    output_subdir: str
    is_reasoning_model: bool
    judge_model_name: str
    tokenizer_model_name: str
    max_model_len: int
    vllm_use_v1: bool
    extra_args: str
    enable_thinking: bool | None


def parse_bool(value: Any, *, name: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ReplayConfigurationError(f"{name} must be a boolean value, received {value!r}")


def parse_optional_bool(value: Any, *, name: str) -> bool | None:
    if value is None or not str(value).strip():
        return None
    return parse_bool(value, name=name)


def load_model_profile(path: Path | str) -> ModelProfile:
    profile_path = Path(path).expanduser().resolve()
    if not profile_path.is_file():
        raise FileNotFoundError(f"Model profile does not exist: {profile_path}")
    values = {key: value for key, value in dotenv_values(profile_path).items() if value is not None}
    missing = sorted(REQUIRED_PROFILE_KEYS - set(values))
    if missing:
        raise ReplayConfigurationError(
            f"Model profile {profile_path} is missing required keys: {missing}"
        )
    try:
        max_model_len = int(values["LLM_MAX_MODEL_LEN"])
    except ValueError as exc:
        raise ReplayConfigurationError("LLM_MAX_MODEL_LEN must be an integer") from exc
    if max_model_len <= 0:
        raise ReplayConfigurationError("LLM_MAX_MODEL_LEN must be positive")
    return ModelProfile(
        path=profile_path,
        model=values["VERIFACT_MODEL"],
        output_subdir=values["VERIFACT_OUTPUT_SUBDIR"],
        is_reasoning_model=parse_bool(values["IS_REASONING_MODEL"], name="IS_REASONING_MODEL"),
        judge_model_name=values["LLM_MODEL_NAME"],
        tokenizer_model_name=values["TOKENIZER_MODEL_NAME"],
        max_model_len=max_model_len,
        vllm_use_v1=parse_bool(values["MAIN_VLLM_USE_V1"], name="MAIN_VLLM_USE_V1"),
        extra_args=values["EXTRA_ARGS"],
        enable_thinking=parse_optional_bool(
            values.get("LLM_CHAT_TEMPLATE_ENABLE_THINKING"),
            name="LLM_CHAT_TEMPLATE_ENABLE_THINKING",
        ),
    )


def _release_file(release_dir: Path | str, relative_path: str) -> Path:
    path = Path(release_dir).expanduser().resolve() / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required PhysioNet v1.1.0 file does not exist: {path}")
    return path


def load_model_metadata(release_dir: Path | str, model: str) -> dict[str, Any]:
    path = _release_file(release_dir, "verifact/model_metadata.csv")
    metadata = pd.read_csv(path, dtype="string", keep_default_na=False)
    required = {
        "model",
        "judge_model_name",
        "tokenizer_model_name",
        "is_reasoning_model",
        "structured_output_model_name",
        "structured_output_tokenizer_name",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ReplayConfigurationError(f"model_metadata.csv is missing columns: {missing}")
    selected = metadata.loc[metadata["model"].eq(model)]
    if len(selected) != 1:
        known = sorted(metadata["model"].tolist())
        raise ReplayConfigurationError(
            f"Expected one metadata row for {model!r}; found {len(selected)}. Known models: {known}"
        )
    row = selected.iloc[0].to_dict()
    row["is_reasoning_model"] = parse_bool(
        row["is_reasoning_model"], name="model_metadata.is_reasoning_model"
    )
    return row


def validate_profile_against_metadata(profile: ModelProfile, metadata: dict[str, Any]) -> None:
    comparisons = {
        "model": (profile.model, metadata["model"]),
        "judge model": (profile.judge_model_name, metadata["judge_model_name"]),
        "tokenizer": (profile.tokenizer_model_name, metadata["tokenizer_model_name"]),
        "reasoning flag": (profile.is_reasoning_model, metadata["is_reasoning_model"]),
    }
    mismatches = [
        f"{name}: profile={actual!r}, release={expected!r}"
        for name, (actual, expected) in comparisons.items()
        if actual != expected
    ]
    if profile.model == "Qwen-3-32B" and profile.enable_thinking is not False:
        mismatches.append("Qwen-3-32B must set LLM_CHAT_TEMPLATE_ENABLE_THINKING=false")
    if profile.model == "Qwen-3-30B-A3B-Thinking" and profile.enable_thinking is not True:
        mismatches.append("Qwen-3-30B-A3B-Thinking must set LLM_CHAT_TEMPLATE_ENABLE_THINKING=true")
    if mismatches:
        raise ReplayConfigurationError(
            "Model profile does not match release metadata: " + "; ".join(mismatches)
        )


def activate_model_profile(profile: ModelProfile) -> None:
    load_environment()
    load_dotenv(profile.path, override=True)


def _filter_values(frame: pd.DataFrame, column: str, values: list[Any] | None) -> pd.DataFrame:
    if not values:
        return frame
    return frame.loc[frame[column].isin(values)]


def load_replay_manifest(
    release_dir: Path | str,
    *,
    model: str,
    rater_aliases: list[str] | None = None,
    subject_ids: list[int] | None = None,
    author_types: list[str] | None = None,
    proposition_types: list[str] | None = None,
    fact_types: list[str] | None = None,
    top_n: list[int] | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    verdicts_path = _release_file(release_dir, "verifact/verdicts.parquet")
    verdicts = pd.read_parquet(
        verdicts_path,
        columns=["model", "proposition_id", "rater_alias", "reference_id", "verdict"],
        filters=[("model", "==", model)],
    ).rename(columns={"verdict": "expected_verdict"})
    if verdicts.empty:
        raise ReplayConfigurationError(f"No released verdict rows found for model {model!r}")
    if verdicts.duplicated(["model", "proposition_id", "rater_alias"]).any():
        raise ReplayConfigurationError("verdicts.parquet contains duplicate replay keys")

    raters = pd.read_csv(
        _release_file(release_dir, "verifact/rater_configurations.csv"),
        keep_default_na=False,
    )
    raters = raters.loc[raters["model"].eq(model)]
    if raters.duplicated(["model", "rater_alias"]).any():
        raise ReplayConfigurationError("rater_configurations.csv contains duplicate keys")
    manifest = verdicts.merge(
        raters,
        on=["model", "rater_alias"],
        how="left",
        validate="many_to_one",
        indicator="_rater_join",
    )
    if manifest["_rater_join"].ne("both").any():
        raise ReplayConfigurationError("Released verdict rows contain unresolved rater aliases")
    manifest = manifest.drop(columns="_rater_join")

    propositions = pd.read_csv(
        _release_file(release_dir, "propositions/propositions.csv.gz"),
        usecols=["proposition_id", "text", "subject_id", "author_type", "proposition_type"],
    )
    if propositions["proposition_id"].duplicated().any():
        raise ReplayConfigurationError("propositions.csv.gz contains duplicate proposition IDs")
    manifest = manifest.merge(
        propositions,
        on="proposition_id",
        how="left",
        validate="many_to_one",
        indicator="_proposition_join",
    )
    if manifest["_proposition_join"].ne("both").any():
        raise ReplayConfigurationError("Released verdict rows contain unresolved proposition IDs")
    manifest = manifest.drop(columns="_proposition_join")

    configuration_matrix = pd.read_csv(
        _release_file(release_dir, "verifact/model_configuration_matrix.csv"),
        keep_default_na=False,
    )
    if configuration_matrix.duplicated(MODEL_CONFIGURATION_COLUMNS).any():
        raise ReplayConfigurationError("model_configuration_matrix.csv contains duplicate keys")
    manifest = manifest.merge(
        configuration_matrix[MODEL_CONFIGURATION_COLUMNS + ["publication_role"]],
        on=MODEL_CONFIGURATION_COLUMNS,
        how="left",
        validate="many_to_one",
        indicator="_configuration_join",
    )
    if manifest["_configuration_join"].ne("both").any():
        raise ReplayConfigurationError(
            "Released verdict rows contain configurations outside model_configuration_matrix.csv"
        )
    manifest = manifest.drop(columns="_configuration_join")

    manifest = _filter_values(manifest, "rater_alias", rater_aliases)
    manifest = _filter_values(manifest, "subject_id", subject_ids)
    manifest = _filter_values(manifest, "author_type", author_types)
    manifest = _filter_values(manifest, "proposition_type", proposition_types)
    manifest = _filter_values(manifest, "fact_type", fact_types)
    manifest = _filter_values(manifest, "top_n", top_n)
    manifest = manifest.sort_values(
        ["rater_alias", "subject_id", "author_type", "proposition_type", "proposition_id"],
        kind="stable",
    ).reset_index(drop=True)
    if limit is not None:
        if limit <= 0:
            raise ReplayConfigurationError("limit must be positive")
        manifest = manifest.head(limit).copy()
    if manifest.empty:
        raise ReplayConfigurationError("Replay selection contains no rows")
    return manifest


def summarize_manifest(manifest: pd.DataFrame, profile: ModelProfile) -> dict[str, Any]:
    return {
        "profile": str(profile.path),
        "model": profile.model,
        "judge_model_name": profile.judge_model_name,
        "is_reasoning_model": profile.is_reasoning_model,
        "rows": len(manifest),
        "subjects": int(manifest["subject_id"].nunique()),
        "rater_aliases": int(manifest["rater_alias"].nunique()),
        "publication_roles": sorted(manifest["publication_role"].unique().tolist()),
        "groups": int(
            manifest[["rater_alias", "subject_id", "author_type", "proposition_type"]]
            .drop_duplicates()
            .shape[0]
        ),
    }


def attach_reference_payloads(manifest: pd.DataFrame, release_dir: Path | str) -> pd.DataFrame:
    requested = set(manifest["reference_id"].astype(str))
    path = _release_file(release_dir, "verifact/reference_payloads.parquet")
    parquet_file = pq.ParquetFile(path)
    required = {"reference_id", "reference"}
    missing = required - set(parquet_file.schema_arrow.names)
    if missing:
        raise ReplayConfigurationError(
            f"reference_payloads.parquet is missing columns: {sorted(missing)}"
        )
    records: list[tuple[str, str]] = []
    found: set[str] = set()
    for batch in parquet_file.iter_batches(
        columns=["reference_id", "reference"],
        batch_size=REFERENCE_BATCH_SIZE,
        use_threads=False,
    ):
        values = batch.to_pydict()
        for reference_id, reference in zip(
            values["reference_id"], values["reference"], strict=True
        ):
            reference_id = str(reference_id)
            if reference_id in requested:
                if reference_id in found:
                    raise ReplayConfigurationError(
                        f"Duplicate reference payload for {reference_id}"
                    )
                if reference is None:
                    raise ReplayConfigurationError(f"Reference payload {reference_id} is null")
                records.append((reference_id, str(reference)))
                found.add(reference_id)
        if found == requested:
            break
    missing_ids = sorted(requested - found)
    if missing_ids:
        raise ReplayConfigurationError(
            f"Missing {len(missing_ids)} reference payload(s); first values: {missing_ids[:5]}"
        )
    references = pd.DataFrame(records, columns=["reference_id", "reference"])
    result = manifest.merge(
        references,
        on="reference_id",
        how="left",
        validate="many_to_one",
    )
    if result["reference"].isna().any() or len(result) != len(manifest):
        raise ReplayConfigurationError("Reference join did not preserve the replay manifest")
    return result


def validate_runtime_environment(profile: ModelProfile, metadata: dict[str, Any]) -> None:
    required = ["LLM_URL_BASE"]
    if profile.is_reasoning_model:
        required.extend(
            [
                "AUX_LLM_URL_BASE",
                "AUX_LLM_MODEL_NAME",
                "AUX_TOKENIZER_MODEL_NAME",
                "AUX_LLM_MAX_MODEL_LEN",
            ]
        )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise ReplayConfigurationError(
            f"Runtime environment is missing required variables: {missing}. "
            "Copy .env.example to .env before running inference."
        )
    if profile.is_reasoning_model:
        comparisons = {
            "AUX_LLM_MODEL_NAME": metadata["structured_output_model_name"],
            "AUX_TOKENIZER_MODEL_NAME": metadata["structured_output_tokenizer_name"],
        }
        mismatches = [
            f"{name}: configured={os.environ.get(name)!r}, release={expected!r}"
            for name, expected in comparisons.items()
            if os.environ.get(name) != expected
        ]
        if mismatches:
            raise ReplayConfigurationError(
                "Auxiliary structured-output configuration does not match release metadata: "
                + "; ".join(mismatches)
            )


def build_judge(
    profile: ModelProfile,
    *,
    workers: int,
    temperature: float,
    top_p: float,
) -> Judge:
    llm = get_llm(temperature=temperature, top_p=top_p)
    aux_llm = (
        get_aux_llm(temperature=temperature, top_p=top_p) if profile.is_reasoning_model else None
    )
    return Judge.from_defaults(
        llm=llm,
        aux_llm=aux_llm,
        is_reasoning_model=profile.is_reasoning_model,
        num_workers=workers,
        num_invalid_output_retries=5,
    )


def _safe_slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def group_output_path(output_dir: Path, group: pd.DataFrame) -> Path:
    first = group.iloc[0]
    filename = (
        f"subject-{int(first['subject_id'])}-{_safe_slug(first['author_type'])}-"
        f"{_safe_slug(first['proposition_type'])}.parquet"
    )
    return output_dir / _safe_slug(first["rater_alias"]) / filename


def _validate_existing_output(path: Path, group: pd.DataFrame) -> None:
    existing = pd.read_parquet(path, columns=["proposition_id", "rater_alias"])
    expected = group[["proposition_id", "rater_alias"]].reset_index(drop=True)
    if not existing.reset_index(drop=True).equals(expected):
        raise ReplayConfigurationError(
            f"Existing replay output does not match the selected source rows: {path}"
        )


async def replay_manifest(
    manifest: pd.DataFrame,
    *,
    judge: Judge,
    output_dir: Path,
    resume: bool,
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_groups = 0
    skipped_groups = 0
    completed_rows = 0
    group_columns = ["rater_alias", "subject_id", "author_type", "proposition_type"]
    for _, group in manifest.groupby(group_columns, sort=False, observed=True):
        group = group.reset_index(drop=True)
        path = group_output_path(output_dir, group)
        if resume and path.is_file():
            _validate_existing_output(path, group)
            skipped_groups += 1
            continue
        first = group.iloc[0]
        report = await judge.a_evaluate(
            texts=group["text"].astype(str).tolist(),
            references=group["reference"].astype(str).tolist(),
            proposition_ids=group["proposition_id"].astype(str).tolist(),
            proposition_type=str(first["proposition_type"]),
            fact_type=str(first["fact_type"]),
            include_explanations=False,
            show_progress=False,
        )
        if report is None or len(report.verdicts) != len(group):
            raise RuntimeError("Judge output row count does not match replay input row count")
        generated = pd.DataFrame(
            [
                {
                    "proposition_id": verdict.proposition_id,
                    "replay_verdict": verdict.verdict,
                    "replay_reason": verdict.reason,
                    "replay_reasoning_chain": verdict.reasoning_chain,
                    "replay_reasoning_final_answer": verdict.reasoning_final_answer,
                }
                for verdict in report.verdicts
            ]
        )
        expected_ids = group["proposition_id"].astype(str).tolist()
        generated_ids = generated["proposition_id"].astype(str).tolist()
        if generated_ids != expected_ids:
            raise RuntimeError("Judge output proposition IDs do not match replay input order")
        source_columns = [
            "model",
            "proposition_id",
            "rater_alias",
            "subject_id",
            "author_type",
            "proposition_type",
            "fact_type",
            "retrieval_method",
            "top_n",
            "reference_format",
            "reference_only_admission",
            "deduplicate_text",
            "publication_role",
            "reference_id",
            "expected_verdict",
        ]
        output = group[source_columns].merge(
            generated,
            on="proposition_id",
            how="left",
            validate="one_to_one",
        )
        output["verdict_matches_release"] = output["replay_verdict"].eq(output["expected_verdict"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp.parquet")
        output.to_parquet(temporary, index=False)
        temporary.replace(path)
        completed_groups += 1
        completed_rows += len(output)
    return {
        "completed_groups": completed_groups,
        "skipped_groups": skipped_groups,
        "completed_rows": completed_rows,
    }


def main(
    model_profile: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            help="Model profile from configs/inference.",
        ),
    ],
    release_dir: Annotated[
        Path | None,
        typer.Option(help="Root of the PhysioNet VeriFact-BHC v1.1.0 release."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(help="Directory for partitioned replay verdict outputs."),
    ] = None,
    rater_alias: Annotated[
        list[str], typer.Option(help="Rater alias to replay; repeat to select multiple aliases.")
    ] = [],
    subject_id: Annotated[
        list[int], typer.Option(help="Subject ID to replay; repeat to select multiple subjects.")
    ] = [],
    author_type: Annotated[
        list[str], typer.Option(help="Author type filter; repeat to select multiple values.")
    ] = [],
    proposition_type: Annotated[
        list[str], typer.Option(help="Proposition type filter; repeat for multiple values.")
    ] = [],
    fact_type: Annotated[
        list[str], typer.Option(help="EHR fact type filter; repeat for multiple values.")
    ] = [],
    top_n: Annotated[
        list[int], typer.Option(help="Retrieved-fact count filter; repeat for multiple values.")
    ] = [],
    limit: Annotated[
        int | None, typer.Option(help="Limit selected rows, primarily for smoke tests.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Validate and summarize exact release rows without inference.")
    ] = False,
    allow_large_run: Annotated[
        bool,
        typer.Option(
            help=f"Allow more than {LARGE_RUN_THRESHOLD} inference rows in one invocation."
        ),
    ] = False,
    resume: Annotated[
        bool, typer.Option(help="Skip complete output partitions after validating their keys.")
    ] = True,
    workers: Annotated[
        int, typer.Option(help="Concurrent requests dispatched within each replay group.")
    ] = 8,
    temperature: Annotated[float, typer.Option(help="Judge sampling temperature.")] = 0.1,
    top_p: Annotated[float, typer.Option(help="Judge nucleus-sampling value.")] = 1.0,
) -> None:
    load_environment()
    profile = load_model_profile(model_profile)
    metadata = load_model_metadata(
        release_dir
        or os.environ.get("VERIFACTBHC_DATASET_DIR", "")
        or Path("data/physionet.org/files/mimic-iii-ext-verifact-bhc/1.1.0"),
        profile.model,
    )
    validate_profile_against_metadata(profile, metadata)
    resolved_release_dir = Path(
        release_dir
        or os.environ.get("VERIFACTBHC_DATASET_DIR", "")
        or "data/physionet.org/files/mimic-iii-ext-verifact-bhc/1.1.0"
    )
    manifest = load_replay_manifest(
        resolved_release_dir,
        model=profile.model,
        rater_aliases=rater_alias,
        subject_ids=subject_id,
        author_types=author_type,
        proposition_types=proposition_type,
        fact_types=fact_type,
        top_n=top_n,
        limit=limit,
    )
    summary = summarize_manifest(manifest, profile)
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))
    if dry_run:
        return
    if len(manifest) > LARGE_RUN_THRESHOLD and not allow_large_run:
        raise typer.BadParameter(
            f"Selection contains {len(manifest):,} rows. Re-run with --allow-large-run or "
            "narrow the selection with --rater-alias, --subject-id, or --limit."
        )
    activate_model_profile(profile)
    validate_runtime_environment(profile, metadata)
    manifest = attach_reference_payloads(manifest, resolved_release_dir)
    judge = build_judge(
        profile,
        workers=workers,
        temperature=temperature,
        top_p=top_p,
    )
    resolved_output_dir = Path(
        output_dir
        or Path(os.environ.get("VERIFACT_RESULTS_DIR", "scripts/evaluate/replay_outputs"))
        / "physionet_v1_1_0_replay"
        / profile.output_subdir
    )
    result = asyncio.run(
        replay_manifest(
            manifest,
            judge=judge,
            output_dir=resolved_output_dir,
            resume=resume,
        )
    )
    run_report = summary | result | {"output_dir": str(resolved_output_dir.resolve())}
    report_path = resolved_output_dir / "replay_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(run_report, indent=2, sort_keys=True) + "\n")
    typer.echo(json.dumps(run_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    typer.run(main)
