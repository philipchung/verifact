# %% [markdown]
# ## Check Completeness of Experiments
#
# This script checks for outputs generated by the VeriFact evaluation pipeline
# from `scripts/evaluate/run_verifact.py`.
# %%
import os
from pathlib import Path
from typing import Any

import pandas as pd
from irr_metrics import MetricBunch, coerce_types
from rag import CLAIM_NODE, HUMAN_AUTHOR, LLM_AUTHOR, SENTENCE_NODE
from tqdm.auto import tqdm
from utils import load_environment, load_pandas

load_environment()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 1000)
pd.set_option("display.min_rows", 60)

# Load Human Clinician Verdict Labels (One Row Per Proposition)
human_verdicts = load_pandas(
    Path(os.environ["VERIFACTBHC_PROPOSITIONS_DIR"]) / "human_verdicts.csv.gz"
)
human_verdicts = coerce_types(human_verdicts)
# Isolate Human Ground Truth Labels
human_gt = (
    human_verdicts.assign(rater_name="human_gt")
    .astype({"rater_name": "string"})
    .rename(columns={"human_gt": "verdict"})
    .loc[:, ["proposition_id", "text", "author_type", "proposition_type", "rater_name", "verdict"]]
)

# Select Experiment Groups to Check: Author Type-Proposition Type-Fact Type
exp_groups = (
    "LLM-Claim-Claim",
    "LLM-Claim-Sentence",
    "LLM-Sentence-Claim",
    "LLM-Sentence-Sentence",
    "Human-Claim-Claim",
    "Human-Claim-Sentence",
)

# Map of VeriFact Result Directories for Each Model
# NOTE: these correspond to the `output_dir` argument when invoking the `run_verifact.py` script
model_dir_map = {
    "Llama-8B": "verifact_llama3_1_8B",
    "Llama-70B": "verifact_llama3_1_70B",
    "R1-8B": "verifact_deepseek_r1_distill_llama_8B",
    "R1-70B": "verifact_deepseek_r1_distill_llama_70B",
}
# Enumerate Unique Models
models = list(model_dir_map.keys())
# Load VeriFact Labels for Different Models
model_data_dict: dict[str, pd.DataFrame] = {}
for model, model_dir in model_dir_map.items():
    model_data_dict[model] = coerce_types(
        load_pandas(
            Path(os.environ["VERIFACT_RESULTS_DIR"])
            / model_dir
            / "score_reports"
            / "verdicts.feather"
        )
    )

# Combine all VeriFact Labels into a Single DataFrame
ai_verdicts = coerce_types(
    pd.concat(
        model_data_dict,
        axis="index",
        names=("model", "proposition_id"),
    )
    .reset_index()
    .astype({"model": "string"})
)
ai_verdicts = ai_verdicts.assign(
    rater_name=ai_verdicts.apply(lambda row: f"model={row.model},{row.rater_name}", axis="columns")
)
# %%

# Check for Completeness of Each Experimental Condition (All Propositions Scored)


def get_expected_num_propositions(author_type: str, proposition_type: str) -> int:
    """Expected number of propositions for each author_type & proposition_type combination.
    Values are based on the number of propositions in the human ground truth dataset."""
    if author_type == LLM_AUTHOR and proposition_type == CLAIM_NODE:
        return 3447
    elif author_type == LLM_AUTHOR and proposition_type == SENTENCE_NODE:
        return 998
    elif author_type == HUMAN_AUTHOR and proposition_type == CLAIM_NODE:
        return 5648
    elif author_type == HUMAN_AUTHOR and proposition_type == SENTENCE_NODE:
        return 2977
    else:
        return None


# Generate all JudgeSingleSubject stratifications and count number of completed propositions
subset_dfs: dict[str, pd.DataFrame] = MetricBunch.stratify_by_columns(
    df=ai_verdicts,
    columns=[
        "model",
        "author_type",
        "proposition_type",
        "fact_type",
        "top_n",
        "retrieval_method",
        "reference_format",
        "reference_only_admission",
    ],
)

# Check for any partially completed runs
complete_check = {}
for strata_name, subset_df in tqdm(
    subset_dfs.items(),
    total=len(subset_dfs),
    desc="Checking Completeness",
):
    completed_items = subset_df.shape[0]
    if completed_items == 0:
        continue
    unique_propositions = subset_df["proposition_id"].nunique()
    expected_num_propositions = get_expected_num_propositions(
        author_type=subset_df["author_type"].iloc[0],
        proposition_type=subset_df["proposition_type"].iloc[0],
    )
    missing_propositions = expected_num_propositions - completed_items
    complete_check[strata_name] = pd.Series(
        {
            "completed_propositions": completed_items,
            "unique_propositions": unique_propositions,
            "expected_propositions": expected_num_propositions,
            "missing_propositions": missing_propositions,
            "model": subset_df["model"].iloc[0],
            "subject_id": subset_df["subject_id"].iloc[0],
            "author_type": subset_df["author_type"].iloc[0],
            "proposition_type": subset_df["proposition_type"].iloc[0],
            "fact_type": subset_df["fact_type"].iloc[0],
            "top_n": subset_df["top_n"].iloc[0],
            "retrieval_method": subset_df["retrieval_method"].iloc[0],
            "reference_format": subset_df["reference_format"].iloc[0],
            "reference_only_admission": subset_df["reference_only_admission"].iloc[0],
            "deduplicate_text": subset_df["deduplicate_text"].iloc[0],
        },
        name=strata_name,
    )

complete_check_df = pd.DataFrame(complete_check.values())

# Display Incomplete Conditions
complete_check_df.query("missing_propositions > 0")
# %%
## Check That Specific Experiments were initiated and we have at least one result value from
# that experiment


def assign_status(completed: int, expected: int) -> str:
    if completed == 0:
        return "Missing All"
    elif completed < expected:
        num_missing = expected - completed
        return f"Missing {num_missing} (Completed: {completed})"
    elif completed == expected:
        return f"All Completed (Completed: {completed})"
    else:
        return "Unexpected Condition."


def check_top_n_experiment(
    df: pd.DataFrame, model: str, author_type: str, proposition_type: str, fact_type: str
) -> dict[int, Any]:
    # Get Subset of Experiments to Check
    retrieval_method = "rerank"
    reference_format = "absolute time"
    reference_only_admission = True
    _df = df.query(
        f"model == '{model}' & author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & fact_type == '{fact_type}' "
        f"& retrieval_method == '{retrieval_method}' & reference_format == '{reference_format}' "
        f"& reference_only_admission == {reference_only_admission}"
    )
    # Get Expected Number of Propositions
    expected_num_propositions = get_expected_num_propositions(
        author_type=author_type,
        proposition_type=proposition_type,
    )
    # Check for Missing Top N Values
    top_n_status: dict[int, Any] = {}
    top_n_values: list[int] = [5, 10, 25, 50, 75, 100, 125, 150]
    for top_n in top_n_values:
        _df_top_n = _df.query(f"top_n == {top_n}")
        num_propositions_completed = _df_top_n.shape[0]
        top_n_status[top_n] = assign_status(num_propositions_completed, expected_num_propositions)
    return top_n_status


def check_retrieval_method_experiment(
    df: pd.DataFrame, model: str, author_type: str, proposition_type: str, fact_type: str
) -> dict[int, Any]:
    # Get Subset of Experiments to Check
    top_n = 50
    reference_format = "absolute time"
    reference_only_admission = True
    _df = df.query(
        f"model == '{model}' & author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & fact_type == '{fact_type}' "
        f"& top_n == {top_n} & reference_format == '{reference_format}' "
        f"& reference_only_admission == {reference_only_admission}"
    )
    # Get Expected Number of Propositions
    expected_num_propositions = get_expected_num_propositions(
        author_type=author_type,
        proposition_type=proposition_type,
    )
    # Check for Missing Retrieval Method Values
    retrieval_method_status: dict[str, Any] = {}
    retrieval_methods = ["dense", "sparse", "hybrid", "rerank"]
    for retrieval_method in retrieval_methods:
        _df_retrieval_method = _df.query(f"retrieval_method == '{retrieval_method}'")
        num_propositions_completed = _df_retrieval_method.shape[0]
        retrieval_method_status[retrieval_method] = assign_status(
            num_propositions_completed, expected_num_propositions
        )
    return retrieval_method_status


def check_reference_format_experiment(
    df: pd.DataFrame, model: str, author_type: str, proposition_type: str, fact_type: str
) -> dict[int, Any]:
    # Get Subset of Experiments to Check
    top_n = 50
    retrieval_method = "rerank"
    reference_only_admission = True
    _df = df.query(
        f"model == '{model}' & author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & fact_type == '{fact_type}' "
        f"& top_n == {top_n} & retrieval_method == '{retrieval_method}' "
        f"& reference_only_admission == {reference_only_admission}"
    )
    # Get Expected Number of Propositions
    expected_num_propositions = get_expected_num_propositions(
        author_type=author_type,
        proposition_type=proposition_type,
    )
    # Check for Missing Reference Format Values
    reference_format_status: dict[str, Any] = {}
    reference_formats = ["score", "absolute time", "relative time"]
    for reference_format in reference_formats:
        _df_reference_format = _df.query(f"reference_format == '{reference_format}'")
        num_propositions_completed = _df_reference_format.shape[0]
        reference_format_status[reference_format] = assign_status(
            num_propositions_completed, expected_num_propositions
        )
    return reference_format_status


def check_reference_only_admission_experiment(
    df: pd.DataFrame, model: str, author_type: str, proposition_type: str, fact_type: str
) -> dict[int, Any]:
    # Get Subset of Experiments to Check
    top_n = 50
    retrieval_method = "rerank"
    reference_format = "absolute time"
    _df = df.query(
        f"model == '{model}' & author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & fact_type == '{fact_type}' "
        f"& top_n == {top_n} & retrieval_method == '{retrieval_method}' "
        f"& reference_format == '{reference_format}'"
    )
    # Get Expected Number of Propositions
    expected_num_propositions = get_expected_num_propositions(
        author_type=author_type,
        proposition_type=proposition_type,
    )
    # Check for Missing Reference Only Admission Values
    reference_only_admission_status: dict[bool, Any] = {}
    reference_only_admission_values = [True, False]
    for reference_only_admission in reference_only_admission_values:
        _df_reference_only_admission = _df.query(
            f"reference_only_admission == {reference_only_admission}"
        )
        num_propositions_completed = _df_reference_only_admission.shape[0]
        reference_only_admission_status[reference_only_admission] = assign_status(
            num_propositions_completed, expected_num_propositions
        )
    return reference_only_admission_status


def match_experiment_group(exp_group: str) -> tuple[str, str, str]:
    match exp_group:
        case "LLM-Claim-Claim":
            return (LLM_AUTHOR, CLAIM_NODE, CLAIM_NODE)
        case "LLM-Claim-Sentence":
            return (LLM_AUTHOR, CLAIM_NODE, SENTENCE_NODE)
        case "LLM-Sentence-Claim":
            return (LLM_AUTHOR, SENTENCE_NODE, CLAIM_NODE)
        case "LLM-Sentence-Sentence":
            return (LLM_AUTHOR, SENTENCE_NODE, SENTENCE_NODE)
        case "Human-Claim-Claim":
            return (HUMAN_AUTHOR, CLAIM_NODE, CLAIM_NODE)
        case "Human-Claim-Sentence":
            return (HUMAN_AUTHOR, CLAIM_NODE, SENTENCE_NODE)
        case "Human-Sentence-Claim":
            return (HUMAN_AUTHOR, SENTENCE_NODE, CLAIM_NODE)
        case "Human-Sentence-Sentence":
            return (HUMAN_AUTHOR, SENTENCE_NODE, SENTENCE_NODE)
        case _:
            raise ValueError(f"Unexpected Experiment Group: {exp_group}")


# %%
# Top N Experiment Completeness
results = []
for exp_group in exp_groups:
    for model in models:
        # Set Author, Proposition, and Fact Types based on Experiment Group
        author_type, proposition_type, fact_type = match_experiment_group(exp_group)

        # Check Top N Experiment Completeness
        check_result = check_top_n_experiment(
            df=ai_verdicts,
            model=model,
            author_type=author_type,
            proposition_type=proposition_type,
            fact_type=fact_type,
        )
        # Convert Result to DataFrame
        _df = (
            pd.DataFrame.from_dict(check_result, orient="index", columns=["status"])
            .rename_axis(index="top_n")
            .reset_index()
        )
        _df = _df.assign(
            exp_group=exp_group,
            model=model,
        )
        results.append(_df)

# Combine Results
top_n_experiment_check_results = (
    pd.concat(results, axis="index")
    .loc[:, ["exp_group", "model", "top_n", "status"]]
    .reset_index(drop=True)
)
top_n_experiment_check_results

# %%
# Retrieval Method Experiment Completeness
results = []
for exp_group in exp_groups:
    for model in models:
        # Set Author, Proposition, and Fact Types based on Experiment Group
        author_type, proposition_type, fact_type = match_experiment_group(exp_group)

        # Check Top N Experiment Completeness
        check_result = check_retrieval_method_experiment(
            df=ai_verdicts,
            model=model,
            author_type=author_type,
            proposition_type=proposition_type,
            fact_type=fact_type,
        )
        # Convert Result to DataFrame
        _df = (
            pd.DataFrame.from_dict(check_result, orient="index", columns=["status"])
            .rename_axis(index="retrieval_method")
            .reset_index()
        )
        _df = _df.assign(
            exp_group=exp_group,
            model=model,
        )
        results.append(_df)

# Combine Results
retrieval_method_experiment_check_results = (
    pd.concat(results, axis="index")
    .loc[:, ["exp_group", "model", "retrieval_method", "status"]]
    .reset_index(drop=True)
)
retrieval_method_experiment_check_results

# %%
# Reference Format Experiment Completeness
results = []
for exp_group in exp_groups:
    for model in models:
        # Set Author, Proposition, and Fact Types based on Experiment Group
        author_type, proposition_type, fact_type = match_experiment_group(exp_group)

        # Check Top N Experiment Completeness
        check_result = check_reference_format_experiment(
            df=ai_verdicts,
            model=model,
            author_type=author_type,
            proposition_type=proposition_type,
            fact_type=fact_type,
        )
        # Convert Result to DataFrame
        _df = (
            pd.DataFrame.from_dict(check_result, orient="index", columns=["status"])
            .rename_axis(index="reference_format")
            .reset_index()
        )
        _df = _df.assign(
            exp_group=exp_group,
            model=model,
        )
        results.append(_df)

# Combine Results
reference_format_experiment_check_results = (
    pd.concat(results, axis="index")
    .loc[:, ["exp_group", "model", "reference_format", "status"]]
    .reset_index(drop=True)
)
reference_format_experiment_check_results
# %%
# Reference Only Admission Experiment
results = []
for exp_group in exp_groups:
    for model in models:
        # Set Author, Proposition, and Fact Types based on Experiment Group
        author_type, proposition_type, fact_type = match_experiment_group(exp_group)

        # Check Top N Experiment Completeness
        check_result = check_reference_only_admission_experiment(
            df=ai_verdicts,
            model=model,
            author_type=author_type,
            proposition_type=proposition_type,
            fact_type=fact_type,
        )
        # Convert Result to DataFrame
        _df = (
            pd.DataFrame.from_dict(check_result, orient="index", columns=["status"])
            .rename_axis(index="reference_only_admission")
            .reset_index()
        )
        _df = _df.assign(
            exp_group=exp_group,
            model=model,
        )
        results.append(_df)

# Combine Results
reference_only_admission_experiment_check_results = (
    pd.concat(results, axis="index")
    .loc[:, ["exp_group", "model", "reference_only_admission", "status"]]
    .reset_index(drop=True)
)
reference_only_admission_experiment_check_results

# %%
