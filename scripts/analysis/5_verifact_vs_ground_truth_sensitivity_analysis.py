# %% [markdown]
# ## Effect of Hyperparameters on VeriFact Performance
# %%
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from irr_metrics import MetricBunch
from utils import load_environment, load_pandas

load_environment()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

annotated_dataset_dir = Path(os.environ["ANNOTATED_DATASET_DIR"])
mb_dir = Path.cwd() / "3_verifact_vs_ground_truth_tables"
output_dir = Path.cwd() / "5_verifact_vs_ground_truth_sensitivity_analysis"
output_dir.mkdir(exist_ok=True, parents=True)

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

# Load Computed Agreement Metrics (from 3_verifact_vs_ground_truth_agreement.py)
pa_mb = MetricBunch.load(filepath=mb_dir / "pa" / "mb")
pa_binarized_mb = MetricBunch.load(filepath=mb_dir / "pa_binarized" / "mb")
gwet_mb = MetricBunch.load(filepath=mb_dir / "gwet" / "mb")
gwet_binarized_mb = MetricBunch.load(filepath=mb_dir / "gwet_binarized" / "mb")

# %% [markdown]
# ## Sensitivity Analysis Plots on each of the VeriFact Hyperparameter Variables
# %%


def select_rater_for_sensitivity_analysis(
    metric_bunch: MetricBunch,
    analysis_variable: str = "top_n",
    default_retrieval_method: str = "Dense",
    default_top_n: int = 10,
    default_reference_format: str = "Score",
    default_reference_only_admission: str = "Yes",
) -> pd.DataFrame:
    """Select a subset of the data for a given analysis variable and
    defaults for other variables."""
    # Select Data
    selected_ehr_judges = []
    for input_text_type in input_text_types:
        df = getattr(metric_bunch, input_text_type)
        # Default Values
        if analysis_variable != "top_n":
            df = df.query(f"top_n == {default_top_n}")
        if analysis_variable != "retrieval_method":
            df = df.query(f"retrieval_method == '{default_retrieval_method}'")
        if analysis_variable != "reference_format":
            df = df.query(f"reference_format == '{default_reference_format}'")
        if analysis_variable != "reference_only_admission":
            df = df.query(f"reference_only_admission == '{default_reference_only_admission}'")
        df = df.assign(input_text_type=input_text_type)
        selected_ehr_judges.append(df)
    data: pd.DataFrame = pd.concat(selected_ehr_judges, axis="index")
    return data


def make_sensitivity_analysis_plot(
    data: pd.DataFrame,
    analysis_variable: str = "top_n",
    ax: plt.Axes | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    ymin: int | None = None,
    ymax: int | None = None,
    format_y_as_percent: bool = True,
    palette: str | None = "tab10",
    legend: bool = True,
    **kwargs: Any,
) -> plt.Axes:
    """Make a single sensitivity analysis plot for a given analysis variable."""
    if ax is None:
        ax = plt.axes()
    # Make Line Plot
    ax: plt.Axes = sns.lineplot(
        ax=ax,
        data=data,
        x=analysis_variable,
        y="value",
        hue="input_text_type",
        marker="o",
        legend=legend,
        palette=palette,
        **kwargs,
    )
    # Confidence Interval as Background Area Plot
    for input_text_type in input_text_types:
        data_subset = data.query(f"input_text_type == '{input_text_type}'").sort_values(
            by=analysis_variable
        )
        y_lower = data_subset.loc[:, "ci_lower"].tolist()
        y_upper = data_subset.loc[:, "ci_upper"].tolist()
        x = data_subset.loc[:, analysis_variable].tolist()
        ax.fill_between(x=x, y1=y_lower, y2=y_upper, alpha=0.2)
    # Create Title
    variable_name = analysis_variable.replace("_", " ").title()
    ax.set_title(f"{variable_name}" if title is None else title)
    # Y-Axis Formatting
    ax.set_ylabel("Percent Agreement" if ylabel is None else ylabel)
    if format_y_as_percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:.0%}"))
    if ymin or ymax:
        bottom, top = ax.get_ylim()  # Get current y-axis limits
        ax.set_ylim(bottom=ymin or bottom, top=ymax or top)  # Set new y-axis limits
    # X-Axis Formatting
    ax.set_xlabel(
        f"{variable_name}" if xlabel is None else xlabel,
        fontdict={"fontsize": 9, "fontweight": "bold"},
    )
    if analysis_variable == "top_n":
        ax.set_xticks(ticks=[5, 10, 25, 50], labels=[5, 10, 25, 50])
    if analysis_variable == "reference_format":
        ax.set_xticks(
            ticks=[0, 1, 2], labels=["Relevance\nScore", "Absolute\nTime", "Relative\nTime"]
        )
    # ax.grid(which="both", axis="both", linestyle="--", linewidth=0.5)
    ax.grid(which="minor", alpha=0.2)
    ax.grid(which="major", alpha=0.5)
    ax.set_axisbelow(True)
    return ax


# Get Data Subsets for Plotting
data_subsets: dict[str, list[pd.DataFrame]] = {
    "top_n": [
        select_rater_for_sensitivity_analysis(
            metric_bunch=pa_mb,
            analysis_variable="top_n",
        ),
        select_rater_for_sensitivity_analysis(
            metric_bunch=pa_binarized_mb,
            analysis_variable="top_n",
        ),
    ],
    "retrieval_method": [
        select_rater_for_sensitivity_analysis(
            metric_bunch=pa_mb,
            analysis_variable="retrieval_method",
        ),
        select_rater_for_sensitivity_analysis(
            metric_bunch=pa_binarized_mb,
            analysis_variable="retrieval_method",
        ),
    ],
    "reference_format": [
        select_rater_for_sensitivity_analysis(
            metric_bunch=pa_mb,
            analysis_variable="reference_format",
        ),
        select_rater_for_sensitivity_analysis(
            metric_bunch=pa_binarized_mb,
            analysis_variable="reference_format",
        ),
    ],
    "reference_only_admission": [
        select_rater_for_sensitivity_analysis(
            metric_bunch=pa_mb,
            analysis_variable="reference_only_admission",
        ),
        select_rater_for_sensitivity_analysis(
            metric_bunch=pa_binarized_mb,
            analysis_variable="reference_only_admission",
        ),
    ],
}
# %%
# Make Plot Figure with Original Labels Only
fig = plt.figure(figsize=(8, 7), layout="constrained")
gs = plt.GridSpec(
    nrows=3,
    ncols=2,
    figure=fig,
    height_ratios=[1, 1, 0.1],
    width_ratios=[1, 1],
)
# Figure Titles & Labels
fig.suptitle("Hyperparameter Importance", fontsize=16)
# Create Subplots
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
ax2 = fig.add_subplot(gs[1, 0], sharey=ax0)
ax3 = fig.add_subplot(gs[1, 1], sharey=ax0)
axes = np.asarray([[ax0, ax1], [ax2, ax3]])

# Create Actual Plots for Each Subplot
analysis_variable = "top_n"
make_sensitivity_analysis_plot(
    data=data_subsets[analysis_variable][0],
    analysis_variable=analysis_variable,
    ax=ax0,
    title="Top N",
    xlabel="",
    ylabel="Percent Agreement",
)
analysis_variable = "retrieval_method"
make_sensitivity_analysis_plot(
    data=data_subsets[analysis_variable][0],
    analysis_variable=analysis_variable,
    ax=ax1,
    title="Retrieval Method",
    xlabel="",
    ylabel="Percent Agreement",
)
analysis_variable = "reference_format"
make_sensitivity_analysis_plot(
    data=data_subsets[analysis_variable][0],
    analysis_variable=analysis_variable,
    ax=ax2,
    title="Reference Context Format",
    xlabel="",
    ylabel="Percent Agreement",
)
analysis_variable = "reference_only_admission"
make_sensitivity_analysis_plot(
    data=data_subsets[analysis_variable][0],
    analysis_variable=analysis_variable,
    ax=ax3,
    title="Retrieve Facts Only From Current Admission",
    xlabel="",
    ylabel="Percent Agreement",
)
# Fill Bottom Grid Space with Legend
handles, labels = ax0.get_legend_handles_labels()
labels = [
    "LLM-written Text,\nAtomic Claim Proposition",
    "LLM-written Text,\nSentence Proposition",
    "Human-written Text,\nAtomic Claim Proposition",
    "Human-written Text,\nSentence Proposition",
]

legend_p = fig.add_subplot(gs[2, 0:2])
legend_p.set_frame_on(False)
legend_p.get_xaxis().set_visible(False)
legend_p.get_yaxis().set_visible(False)
legend_p.legend(
    handles,
    labels,
    loc="center",
    ncol=4,
    fontsize=9,
)
# Remove Legends on All Plots
for ax in axes.flat:
    ax.get_legend().remove()
# Save Figure
fig.savefig(output_dir / "verifact_hyperparameters.png", dpi=300, bbox_inches="tight")

# %%
