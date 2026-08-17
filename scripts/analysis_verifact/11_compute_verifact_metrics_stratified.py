# %% [markdown]
# ## Compute VeriFact Metrics Stratified by Round 1 Annotator Agreement
# %%
import itertools
import os
from pathlib import Path

import pandas as pd
from irr_metrics import MetricBunch, coerce_types
from release_data import load_release_annotations, load_release_ground_truth
from tqdm.auto import tqdm
from utils import (
    get_function_status_string,
    get_local_time,
    get_utc_time,
    load_environment,
    load_pandas,
    send_notification,
)

load_environment()
analysis_dir = Path(os.environ["PROJECT_DIR"]) / "scripts" / "analysis_verifact"
release_dir = Path(os.environ["VERIFACTBHC_DATASET_DIR"])
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

# Load Human Clinician Verdict Labels (One Row Per Proposition)
human_verdicts = load_pandas(release_dir / "propositions" / "human_verdicts.csv.gz")
human_verdicts = coerce_types(human_verdicts)
human_gt = load_release_ground_truth(release_dir)
# Load Propositions
propositions = load_pandas(release_dir / "propositions" / "propositions.csv.gz")
# Load Proposition Validity Analysis Annotations
proposition_validity = load_pandas(release_dir / "propositions" / "proposition_validity.csv.gz")
# %%
# Create Dataframe for Examining Propositions
left_df = human_verdicts.loc[
    :,
    [
        "proposition_id",
        "text",
        "author_type",
        "proposition_type",
        "round1_num_raters_agree",
        "round1_majority_vote",
        # "adjudicated_verdict",
        # "adjudicated_comment",
        "human_gt",
    ],
]
right_df = proposition_validity.loc[
    :, ["proposition_id", "invalid", "imperative", "interrogative", "incomplete", "vague"]
]
df = pd.merge(
    left=left_df,
    right=right_df,
    on="proposition_id",
    how="left",
)

# Stratify Propositions by Round 1 annotation agreement
all_disagree = df.query("round1_num_raters_agree == 0")
majority_agree = df.query("round1_num_raters_agree == 2")
unanimous_agree = df.query("round1_num_raters_agree == 3")

# %%
## Load Best VeriFact Models to compare against Ground Truth Human Labels

models = ["Qwen3-30B-A3B-Thinking"]
ai_verdicts = load_release_annotations(release_dir, models=models)
# Narrow to top_n = 150 for claims, 100 for sentences
ai_verdicts_selected = ai_verdicts.query(
    "((fact_type == 'claim' & top_n == 150) | (fact_type == 'sentence' & top_n == 100))"
)
# List the selected VeriFact AI Hyperparameter Combinations
print("Selected VeriFact AI Hyperparameter Combinations:")
for i, r in enumerate(ai_verdicts_selected.rater_name.unique()):
    print(f"{i + 1}: {r}")


# %%
## Compute Interrater Agreement Metrics for Human Raters vs. VeriFact Models for Strata
agreement_strata = [
    "all_disagree",
    "majority_agree",
    "unanimous_agree",
    "all",
]
metrics = [
    "percent_agreement",
    "gwet",
    "mcc",
    # "s-tpr",
    # "s-tnr"s,
    # "s-ppv",
    # "s-npv",
    # "ns-tpr",
    # "ns-tnr",
    # "ns-ppv",
    # "ns-npv",
    # "na-tpr",
    # "na-tnr",
    # "na-ppv",
    # "na-npv",
]

author_types = ["llm", "human"]
proposition_types = ["claim", "sentence"]
workers = 20
num_parallel_raters = 4
bootstrap_iterations = 1000
force_recompute = False

# Set Run Name & Notification Timestamp Info
run_name = f"Models: {models}"
start_utc_time = get_utc_time(output_format="str")
start_local_time = get_local_time(output_format="str")


def select_strata_data(agreement_strata: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    match agreement_strata:
        case "all_disagree":
            rater_verdicts = ai_verdicts_selected.query(
                "proposition_id in @all_disagree.proposition_id"
            ).copy()
            ground_truth = human_gt.query("proposition_id in @all_disagree.proposition_id").copy()
        case "majority_agree":
            rater_verdicts = ai_verdicts_selected.query(
                "proposition_id in @majority_agree.proposition_id"
            ).copy()
            ground_truth = human_gt.query("proposition_id in @majority_agree.proposition_id").copy()
        case "unanimous_agree":
            rater_verdicts = ai_verdicts_selected.query(
                "proposition_id in @unanimous_agree.proposition_id"
            ).copy()
            ground_truth = human_gt.query(
                "proposition_id in @unanimous_agree.proposition_id"
            ).copy()
        case "all":
            rater_verdicts = ai_verdicts_selected.copy()
            ground_truth = human_gt.copy()
    return rater_verdicts, ground_truth  # type: ignore


# Sweep over models, metrics, agreement strata
for model, metric, agreement_stratum in (
    pbar := tqdm(list(itertools.product(models, metrics, agreement_strata)))
):
    pbar.set_description(f"Model: {model}, Metric: {metric}, Stratum: {agreement_stratum}")

    # Display name for metric
    metric_name = metric.replace(" ", "_")
    # Define Proposition Strata
    rater_verdicts, ground_truth = select_strata_data(agreement_stratum)

    # Define save directory for Metric Bunch
    name = f"strata_{agreement_stratum}-{metric_name}_ci"
    save_dir = (
        Path(os.environ["PROJECT_DIR"])
        / "scripts"
        / "analysis_verifact"
        / "11_compute_verifact_metrics_stratified"
    )

    # Compute Metric Bunch, reloading computed metrics from cache if available
    mb = MetricBunch.from_defaults(
        name=name,
        rater_verdicts=rater_verdicts,
        ground_truth=ground_truth,
        metric=metric,
        rater_type="ai",
        rater_id_col="stratum",
        stratify_cols=[
            "model",
            "author_type",
            "proposition_type",
            "fact_type",
            "top_n",
            "retrieval_method",
            "reference_format",
            "reference_only_admission",
        ],
        workers=workers,
        num_parallel_raters=num_parallel_raters,
        bootstrap_iterations=bootstrap_iterations,
        cache_dir=save_dir,
        force_recompute=force_recompute,
        show_progress=True,
    )

# Create Notification Message
msg = get_function_status_string(
    filename=__file__, start_utc_time=start_utc_time, start_local_time=start_local_time
)
send_notification(
    title=f"Completed Run: {run_name}", message=msg, url=os.environ["NOTIFY_WEBHOOK_URL"]
)
# %%
