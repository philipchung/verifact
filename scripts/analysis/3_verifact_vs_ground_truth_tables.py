# %% [markdown]
# ## Agreement for each VeriFact AI System Variation vs. Ground Truth Human Clinician Label
#
# Percent Agreement & Gwet's AC1 is computed for both original and binarized labels
# (treating Not Addressed labels as Not Supported). Then the results are displayed in table format.
# %%
import os
from pathlib import Path

import pandas as pd
from irr_metrics import MetricBunch
from utils import load_environment, load_pandas, save_pandas

load_environment()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

annotated_dataset_dir = Path(os.environ["ANNOTATED_DATASET_DIR"])
output_dir = Path.cwd() / "3_verifact_vs_ground_truth_tables"

# Groups: Cross Product of Author Type & Proposition Type
input_text_types = ["llm_claim", "llm_sentence", "human_claim", "human_sentence"]

# Load Human Clinician Verdict Labels (One Row Per Proposition)
human_verdicts = load_pandas(annotated_dataset_dir / "human_verdicts.csv.gz").astype(
    {
        "proposition_id": "string",
        "text": "string",
        "author_type": "string",
        "proposition_type": "string",
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
    }
)
# Isolate Human Ground Truth Labels
human_gt = human_verdicts[
    ["proposition_id", "text", "author_type", "proposition_type", "human_gt"]
].rename(columns={"human_gt": "verdict"})
human_gt = human_gt.assign(rater_name="human_gt").astype({"rater_name": "string"})

# Load VeriFact AI System Verdict Labels
ai_verdicts = load_pandas(annotated_dataset_dir / "verifact_verdicts.csv.gz").astype(
    {
        "proposition_id": "string",
        "text": "string",
        "author_type": "string",
        "proposition_type": "string",
        "rater_name": "string",
        "rater_alias": "string",
        "verdict": "string",
        "reason": "string",
        "reference": "string",
        "retrieval_method": "string",
        "top_n": "Int64",
        "reference_format": "string",
        "reference_only_admission": "boolean",
    }
)

# Binarize Human and VeriFact AI System Verdicts


def binarize_verdicts(verdict: str) -> str:
    """Convert the original 3-label verdicts to 2-labels (Supported, Not Supported)."""
    if pd.isna(verdict):
        return pd.NA
    elif verdict == "Supported":
        return "Supported"
    else:
        return "Not Supported"


def binarize_verdicts_in_df(df: pd.DataFrame, verdict_name: str = "verdict") -> pd.DataFrame:
    """Convert the original 3-label verdicts to 2-labels (Supported, Not Supported)."""
    return df.assign(**{verdict_name: df[verdict_name].map(binarize_verdicts)})


human_gt_binarized = binarize_verdicts_in_df(human_gt)
ai_verdicts_binarized = binarize_verdicts_in_df(ai_verdicts)

# Number of Bootstrap Iterations for Confidence Intervals Estimation
bootstrap_iterations = 1000
# Number of concurrent comparisons between VeriFact Raters vs. Human Ground Truth
num_parallel_raters = 16
# Number of concurrent workers computing bootstrap iterations for each comparison
workers_per_rater = 8
# %%
# Compute Percent Agreement - Original Labels
try:
    judge_pa_mb = MetricBunch.load(filepath=output_dir / "pa" / "mb")
except Exception:
    judge_pa_mb = MetricBunch.from_defaults(
        ground_truth_labels=human_gt,
        rater_verdicts=ai_verdicts,
        rater_type="ai",
        rater_id_col="rater_alias",
        strata=input_text_types,
        metric="percent_agreement",
        bootstrap_iterations=bootstrap_iterations,
        workers=workers_per_rater,
        num_parallel_raters=num_parallel_raters,
    )
    judge_pa_mb.save(filepath=output_dir / "pa" / "mb")
    judge_pa_mb.export_crosstabs(filepath=output_dir / "pa" / "xtab", strata=input_text_types)

# %%
# Compute Percent Agreement - Binarized Labels
try:
    judge_pa_binarized_mb = MetricBunch.load(filepath=output_dir / "pa_binarized" / "mb")
except Exception:
    judge_pa_binarized_mb = MetricBunch.from_defaults(
        ground_truth_labels=human_gt_binarized,
        rater_verdicts=ai_verdicts_binarized,
        rater_type="ai",
        rater_id_col="rater_alias",
        strata=input_text_types,
        metric="percent_agreement",
        bootstrap_iterations=bootstrap_iterations,
        workers=workers_per_rater,
        num_parallel_raters=num_parallel_raters,
    )
    judge_pa_binarized_mb.save(filepath=output_dir / "pa_binarized" / "mb")
    judge_pa_binarized_mb.export_crosstabs(
        filepath=output_dir / "pa_binarized" / "xtab", strata=input_text_types
    )

# %%
# Compute Gwet's AC1 - Original Labels
try:
    judge_gwet_mb = MetricBunch.load(filepath=output_dir / "gwet" / "mb")
except Exception:
    judge_gwet_mb = MetricBunch.from_defaults(
        ground_truth_labels=human_gt,
        rater_verdicts=ai_verdicts,
        rater_type="ai",
        rater_id_col="rater_alias",
        strata=input_text_types,
        metric="gwet",
        bootstrap_iterations=bootstrap_iterations,
        workers=workers_per_rater,
    )
    judge_gwet_mb.save(filepath=output_dir / "gwet" / "mb")
    judge_gwet_mb.export_crosstabs(filepath=output_dir / "gwet" / "xtab", strata=input_text_types)

# %%
# Compute Gwet's AC1 - Binarized Labels
try:
    judge_gwet_binarized_mb = MetricBunch.load(filepath=output_dir / "gwet_binarized" / "mb")
except Exception:
    judge_gwet_binarized_mb = MetricBunch.from_defaults(
        ground_truth_labels=human_gt_binarized,
        rater_verdicts=ai_verdicts_binarized,
        rater_type="ai",
        rater_id_col="rater_alias",
        strata=input_text_types,
        metric="gwet",
        bootstrap_iterations=bootstrap_iterations,
        workers=workers_per_rater,
        num_parallel_raters=num_parallel_raters,
    )
    judge_gwet_binarized_mb.save(filepath=output_dir / "gwet_binarized" / "mb")
    judge_gwet_binarized_mb.export_crosstabs(
        filepath=output_dir / "gwet_binarized" / "xtab", strata=input_text_types
    )

# %%
# Load Computed Agreement Metrics
pa_mb = MetricBunch.load(filepath=output_dir / "pa" / "mb")
pa_binarized_mb = MetricBunch.load(filepath=output_dir / "pa_binarized" / "mb")
gwet_mb = MetricBunch.load(filepath=output_dir / "gwet" / "mb")
gwet_binarized_mb = MetricBunch.load(filepath=output_dir / "gwet_binarized" / "mb")

# Generate Crosstab Tables
pa_mb.export_crosstabs(filepath=output_dir / "pa" / "xtab", strata=input_text_types)
pa_binarized_mb.export_crosstabs(
    filepath=output_dir / "pa_binarized" / "xtab", strata=input_text_types
)
gwet_mb.export_crosstabs(filepath=output_dir / "gwet" / "xtab", strata=input_text_types)
gwet_binarized_mb.export_crosstabs(
    filepath=output_dir / "gwet_binarized" / "xtab", strata=input_text_types
)
# %%
# ## Percent Agreement for LLM-written Summary Propositions

# Create Combined Crosstab Table with 95% Confidence Intervals
pa_original_llm_written = pd.concat(
    {
        "Claim": pa_mb.make_crosstab(stratum="llm_claim", values="display_str"),
        "Sentence": pa_mb.make_crosstab(stratum="llm_sentence", values="display_str"),
    },
    axis="index",
    names=["Proposition Type"],
)
pa_binarized_llm_written = pd.concat(
    {
        "Claim": pa_binarized_mb.make_crosstab(stratum="llm_claim", values="display_str"),
        "Sentence": pa_binarized_mb.make_crosstab(stratum="llm_sentence", values="display_str"),
    },
    axis="index",
    names=["Proposition Type"],
)
pa_llm_written = pd.concat(
    {
        "Original Labels": pa_original_llm_written,
        "Binarized Labels": pa_binarized_llm_written,
    },
    axis="columns",
    names=["Label Space"],
)

# Create Equivalent Table with only raw Percent Agreement Values
pa_original_llm_written_value = pd.concat(
    {
        "Claim": pa_mb.make_crosstab(stratum="llm_claim", values="value"),
        "Sentence": pa_mb.make_crosstab(stratum="llm_sentence", values="value"),
    },
    axis="index",
    names=["Proposition Type"],
)
pa_binarized_llm_written_value = pd.concat(
    {
        "Claim": pa_binarized_mb.make_crosstab(stratum="llm_claim", values="value"),
        "Sentence": pa_binarized_mb.make_crosstab(stratum="llm_sentence", values="value"),
    },
    axis="index",
    names=["Proposition Type"],
)
llm_written_value = pd.concat(
    {
        "Original Labels": pa_original_llm_written_value,
        "Binarized Labels": pa_binarized_llm_written_value,
    },
    axis="columns",
    names=["Label Space"],
)
# Style Table with Heatmap in Background
pa_llm_written_crosstab = pa_llm_written.style.background_gradient(
    cmap="Blues", axis=None, gmap=llm_written_value
)
save_pandas(
    df=pa_llm_written_crosstab, filepath=output_dir / "pa" / "combined_xtab" / "llm_written.xlsx"
)

# %%
# ## Percent Agreement for Human-written Summary Propositions

# Create Combined Crosstab Table with 95% Confidence Intervals
pa_original_human_written = pd.concat(
    {
        "Claim": pa_mb.make_crosstab(stratum="human_claim", values="display_str"),
        "Sentence": pa_mb.make_crosstab(stratum="human_sentence", values="display_str"),
    },
    axis="index",
    names=["Proposition Type"],
)
pa_binarized_human_written = pd.concat(
    {
        "Claim": pa_binarized_mb.make_crosstab(stratum="human_claim", values="display_str"),
        "Sentence": pa_binarized_mb.make_crosstab(stratum="human_sentence", values="display_str"),
    },
    axis="index",
    names=["Proposition Type"],
)
pa_human_written = pd.concat(
    {
        "Original Labels": pa_original_human_written,
        "Binarized Labels": pa_binarized_human_written,
    },
    axis="columns",
    names=["Label Space"],
)

# Create Equivalent Table with only raw Percent Agreement Values
pa_original_human_written_value = pd.concat(
    {
        "Claim": pa_mb.make_crosstab(stratum="human_claim", values="value"),
        "Sentence": pa_mb.make_crosstab(stratum="human_sentence", values="value"),
    },
    axis="index",
    names=["Proposition Type"],
)
pa_binarized_human_written_value = pd.concat(
    {
        "Claim": pa_binarized_mb.make_crosstab(stratum="human_claim", values="value"),
        "Sentence": pa_binarized_mb.make_crosstab(stratum="human_sentence", values="value"),
    },
    axis="index",
    names=["Proposition Type"],
)
human_written_value = pd.concat(
    {
        "Original Labels": pa_original_human_written_value,
        "Binarized Labels": pa_binarized_human_written_value,
    },
    axis="columns",
    names=["Label Space"],
)
# Style Table with Heatmap in Background
pa_human_written_crosstab = pa_human_written.style.background_gradient(
    cmap="Blues", axis=None, gmap=human_written_value
)
save_pandas(
    df=pa_human_written_crosstab,
    filepath=output_dir / "pa" / "combined_xtab" / "human_written.xlsx",
)
# %%
# ## Gwet's AC1 for LLM-written Summary Propositions

# Create Combined Crosstab Table with 95% Confidence Intervals
gwet_original_llm_written = pd.concat(
    {
        "Claim": gwet_mb.make_crosstab(stratum="llm_claim", values="display_str"),
        "Sentence": gwet_mb.make_crosstab(stratum="llm_sentence", values="display_str"),
    },
    axis="index",
    names=["Proposition Type"],
)
gwet_binarized_llm_written = pd.concat(
    {
        "Claim": gwet_binarized_mb.make_crosstab(stratum="llm_claim", values="display_str"),
        "Sentence": gwet_binarized_mb.make_crosstab(stratum="llm_sentence", values="display_str"),
    },
    axis="index",
    names=["Proposition Type"],
)
gwet_llm_written = pd.concat(
    {
        "Original Labels": gwet_original_llm_written,
        "Binarized Labels": gwet_binarized_llm_written,
    },
    axis="columns",
    names=["Label Space"],
)

# Create Equivalent Table with only raw Percent Agreement Values
gwet_original_llm_written_value = pd.concat(
    {
        "Claim": gwet_mb.make_crosstab(stratum="llm_claim", values="value"),
        "Sentence": gwet_mb.make_crosstab(stratum="llm_sentence", values="value"),
    },
    axis="index",
    names=["Proposition Type"],
)
gwet_binarized_llm_written_value = pd.concat(
    {
        "Claim": gwet_binarized_mb.make_crosstab(stratum="llm_claim", values="value"),
        "Sentence": gwet_binarized_mb.make_crosstab(stratum="llm_sentence", values="value"),
    },
    axis="index",
    names=["Proposition Type"],
)
llm_written_value = pd.concat(
    {
        "Original Labels": gwet_original_llm_written_value,
        "Binarized Labels": gwet_binarized_llm_written_value,
    },
    axis="columns",
    names=["Label Space"],
)
# Style Table with Heatmap in Background
gwet_llm_written_crosstab = gwet_llm_written.style.background_gradient(
    cmap="Blues", axis=None, gmap=llm_written_value
)
save_pandas(
    df=gwet_llm_written_crosstab,
    filepath=output_dir / "gwet" / "combined_xtab" / "llm_written.xlsx",
)

# %%
# Gwet's AC1 for Human-written Summary Propositions

# Create Combined Crosstab Table with 95% Confidence Intervals
gwet_original_human_written = pd.concat(
    {
        "Claim": gwet_mb.make_crosstab(stratum="human_claim", values="display_str"),
        "Sentence": gwet_mb.make_crosstab(stratum="human_sentence", values="display_str"),
    },
    axis="index",
    names=["Proposition Type"],
)
gwet_binarized_human_written = pd.concat(
    {
        "Claim": gwet_binarized_mb.make_crosstab(stratum="human_claim", values="display_str"),
        "Sentence": gwet_binarized_mb.make_crosstab(stratum="human_sentence", values="display_str"),
    },
    axis="index",
    names=["Proposition Type"],
)
gwet_human_written = pd.concat(
    {
        "Original Labels": gwet_original_human_written,
        "Binarized Labels": gwet_binarized_human_written,
    },
    axis="columns",
    names=["Label Space"],
)

# Create Equivalent Table with only raw Percent Agreement Values
gwet_original_human_written_value = pd.concat(
    {
        "Claim": gwet_mb.make_crosstab(stratum="human_claim", values="value"),
        "Sentence": gwet_mb.make_crosstab(stratum="human_sentence", values="value"),
    },
    axis="index",
    names=["Proposition Type"],
)
gwet_binarized_human_written_value = pd.concat(
    {
        "Claim": gwet_binarized_mb.make_crosstab(stratum="human_claim", values="value"),
        "Sentence": gwet_binarized_mb.make_crosstab(stratum="human_sentence", values="value"),
    },
    axis="index",
    names=["Proposition Type"],
)
human_written_value = pd.concat(
    {
        "Original Labels": gwet_original_human_written_value,
        "Binarized Labels": gwet_binarized_human_written_value,
    },
    axis="columns",
    names=["Label Space"],
)
# Style Table with Heatmap in Background
gwet_human_written_crosstab = gwet_human_written.style.background_gradient(
    cmap="Blues", axis=None, gmap=human_written_value
)
save_pandas(
    df=gwet_human_written_crosstab,
    filepath=output_dir / "gwet" / "combined_xtab" / "human_written.xlsx",
)
# %%
