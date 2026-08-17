# %% [markdown]
# ## Error Analysis
# %%
import inspect
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from irr_metrics import MetricBunch
from pandas.api.types import CategoricalDtype
from release_data import (
    attach_reference_payloads,
    load_reference_lengths,
    load_release_annotations,
    load_release_ground_truth,
)
from sklearn.metrics import confusion_matrix
from utils import load_environment

load_environment()
analysis_dir = Path(os.environ["PROJECT_DIR"]) / "scripts" / "analysis_verifact"
release_dir = Path(os.environ["VERIFACTBHC_DATASET_DIR"])
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

human_gt = load_release_ground_truth(release_dir)
models = ["Llama-8B", "Llama-70B", "R1-8B", "R1-70B"]
ai_verdicts = load_release_annotations(release_dir, models=models)

# Load MetricBunch with computed metrics from save_dir cache
metric = "percent_agreement"
name = f"ai_rater_{metric}_ci"
mb_save_dir = analysis_dir / "2_compute_verifact_metrics"
mb = MetricBunch.load(save_dir=mb_save_dir, name=name)
# %%
## Add Reference Context Lengths to Metric Bunch
# Compute Mean Reference Context Lengths without loading reference text
# for each Metric Bunch evaluation category
reference_lengths = load_reference_lengths(
    release_dir, reference_ids=ai_verdicts["reference_id"].unique()
)
ai_verdicts = ai_verdicts.merge(
    reference_lengths,
    on="reference_id",
    how="left",
    validate="many_to_one",
)
mean_reference_context_lengths = (
    ai_verdicts.groupby(
        [
            "author_type",
            "proposition_type",
            "fact_type",
            "model",
            "top_n",
            "retrieval_method",
            "reference_format",
            "reference_only_admission",
        ],
        observed=True,
    )
    .agg(
        mean_word_length=("reference_word_count", "mean"),
        mean_char_length=("reference_char_count", "mean"),
    )
    .reset_index()
)
# Join Mean Reference Context Lengths with Metric Bunch Data
mb.metrics = pd.merge(
    mb.metrics,
    mean_reference_context_lengths,
    on=[
        "author_type",
        "proposition_type",
        "fact_type",
        "model",
        "top_n",
        "retrieval_method",
        "reference_format",
        "reference_only_admission",
    ],
    how="left",
)

# %% [markdown]
# ## Helper Functions for Error Analysis
# %%
## Error Analysis when Non-Reasoning Models Fail but Reasoning Models Succeed

author_type = "llm"
proposition_type = "claim"
fact_type = "claim"
n = 150
llama_8B = (
    ai_verdicts.query(
        "model == 'Llama-8B' & "
        f"author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & "
        f"fact_type == '{fact_type}' & "
        f"top_n == {n} & "
        f"retrieval_method == 'rerank' & "
        f"reference_format == 'absolute time' & "
        f"reference_only_admission == True"
    )
    .set_index("proposition_id")
    .sort_index()
)
llama_70B = (
    ai_verdicts.query(
        "model == 'Llama-70B' & "
        f"author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & "
        f"fact_type == '{fact_type}' & "
        f"top_n == {n} & "
        f"retrieval_method == 'rerank' & "
        f"reference_format == 'absolute time' & "
        f"reference_only_admission == True"
    )
    .set_index("proposition_id")
    .sort_index()
)
r1_8B = (
    ai_verdicts.query(
        "model == 'R1-8B' & "
        f"author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & "
        f"fact_type == '{fact_type}' & "
        f"top_n == {n} & "
        f"retrieval_method == 'rerank' & "
        f"reference_format == 'absolute time' & "
        f"reference_only_admission == True"
    )
    .set_index("proposition_id")
    .sort_index()
)
r1_70B = (
    ai_verdicts.query(
        "model == 'R1-70B' & "
        f"author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & "
        f"fact_type == '{fact_type}' & "
        f"top_n == {n} & "
        f"retrieval_method == 'rerank' & "
        f"reference_format == 'absolute time' & "
        f"reference_only_admission == True"
    )
    .set_index("proposition_id")
    .sort_index()
)

# Ground Truth Labels
gt_labels = (
    human_gt.query(f"author_type == '{author_type}' & proposition_type == '{proposition_type}'")
    .set_index("proposition_id")
    .sort_index()
)

# Get Mask for Correct and Incorrect Verdicts when Compared to Ground Truth
llama_8B_correct = llama_8B.verdict == gt_labels.verdict
llama_8B_incorrect = llama_8B.verdict != gt_labels.verdict
llama_70B_correct = llama_70B.verdict == gt_labels.verdict
llama_70B_incorrect = llama_70B.verdict != gt_labels.verdict
r1_8B_correct = r1_8B.verdict == gt_labels.verdict
r1_8B_incorrect = r1_8B.verdict != gt_labels.verdict
r1_70B_correct = r1_70B.verdict == gt_labels.verdict
r1_70B_incorrect = r1_70B.verdict != gt_labels.verdict

# %%
# 2x2 Confusion Matrix Plot


def make_conf_mat_plot(
    df: pd.DataFrame,
    x_var: str,
    y_var: str,
    labels: list,
    vmax: int = 200,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Create a Confusion Matrix Plot for Two Models."""
    # Select data
    x_data = df[x_var].values
    y_data = df[y_var].values
    # Compute Confusion Matrix
    cm = confusion_matrix(y_true=y_data, y_pred=x_data, labels=labels)
    cm_normalize = confusion_matrix(y_true=y_data, y_pred=x_data, labels=labels, normalize="all")
    # Create Annotations
    annotations = []
    for i in range(cm.shape[0]):
        row_list = []
        for j in range(cm.shape[1]):
            row_list.append(f"{cm[i, j]:d}\n({cm_normalize[i, j]:.1%})")
        annotations.append(row_list)

    # Display Heatmap of Confusion Matrix
    if fig is None:
        fig, ax = plt.subplots(figsize=(5, 4), layout="tight")
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 14},
        vmin=0,  # Minimum colormap value
        vmax=vmax,  # Maximum colormap value (values above this are clipped to this value)
        cmap="Blues",
        linewidths=0.5,
        linecolor="white",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"{x_var} vs. {y_var}", size=14)
    ax.set_ylabel(y_var, size=14)
    ax.set_yticklabels(labels=labels, rotation=0, size=12)
    ax.set_xlabel(x_var, size=14)
    ax.set_xticklabels(labels=labels, rotation=0, size=12)
    ax.tick_params(axis="both", which="both", left=False, bottom=False)
    return fig, ax


CORRECT_DTYPE = CategoricalDtype(["Correct", "Incorrect"], ordered=True)
labels = CORRECT_DTYPE.categories

# %%
# Figure for 2x2 Confusion Matrix Comparing Non-Reasoning vs. Reasoning Models
# (Llama-8B vs. R1-8B) and (Llama-70B vs. R1-70B)
data = (
    pd.concat(
        {
            "Llama-8B": llama_8B_correct,
            "R1-8B": r1_8B_correct,
            "Llama-70B": llama_70B_correct,
            "R1-70B": r1_70B_correct,
        },
        axis="columns",
    )
    .map(lambda x: "Correct" if x else "Incorrect")
    .astype(CORRECT_DTYPE)
)

fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), layout="constrained")
ax_left = make_conf_mat_plot(
    df=data, x_var="Llama-8B", y_var="R1-8B", labels=labels, vmax=750, fig=fig, ax=ax_left
)
ax_right = make_conf_mat_plot(
    df=data, x_var="Llama-70B", y_var="R1-70B", labels=labels, vmax=750, fig=fig, ax=ax_right
)
fig.suptitle("Performance Advantage From Reasoning vs. Non-Reasoning Model", size=18)
# Save Figure
save_dir = analysis_dir / "7_error_analysis"
save_dir.mkdir(exist_ok=True, parents=True)
fig.savefig(
    save_dir / "confusion_matrix_reasoning_vs_nonreasoning.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
print(
    "Confusion Matrix for Performance of Reasoning vs. Non-Reasoning Models."
    "Each model's predictions are compared to the human ground truth labels "
    "which are treated as the gold standard. Based on this comparison, each model's "
    "decision can be classified as either 'Correct' or 'Incorrect'. "
    "Counts and percentages in each cell represent the "
    "number and percentage of propositions, respectively."
    "Any deviation from the diagonal indicates a difference in performance between the models."
    ""
)

# %%
# Figure for 2x2 Confusion Matrix Comparing 8B vs. 70B Models
# (Llama-8B vs. Llama-70B) and (R1-8B vs. R1-70B)
fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), layout="constrained")
ax_left = make_conf_mat_plot(
    df=data, x_var="Llama-8B", y_var="Llama-70B", labels=labels, vmax=750, fig=fig, ax=ax_left
)
ax_right = make_conf_mat_plot(
    df=data, x_var="R1-8B", y_var="R1-70B", labels=labels, vmax=750, fig=fig, ax=ax_right
)
fig.suptitle("Performance Advantage From Increasing Model Size", size=18)
# Save Figure
save_dir = analysis_dir / "7_error_analysis"
save_dir.mkdir(exist_ok=True, parents=True)
fig.savefig(
    save_dir / "confusion_matrix_model_size_comparison.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
print(
    "Confusion Matrix for Performance of Increasing Model Parameter Count."
    "Each model's predictions are compared to the human ground truth labels "
    "which are treated as the gold standard. Based on this comparison, each model's "
    "decision can be classified as either 'Correct' or 'Incorrect'. "
    "Counts and percentages in each cell represent the "
    "number and percentage of propositions, respectively."
    "Any deviation from the diagonal indicates a difference in performance between the models."
    ""
)
# %%


def get_error_analysis_df(
    mask: pd.Series,
) -> pd.DataFrame:
    filtered_propositions = mask[mask].index

    # Select Propositions based on Mask
    proposition_ids = pd.Series(filtered_propositions).tolist()  # noqa: F841
    error_df = (
        gt_labels.query("index == @proposition_ids")
        .rename(columns={"verdict": "human_gt_verdict"})
        .assign(
            llama_8B_verdict=llama_8B.query("index == @proposition_ids").verdict,
            r1_8B_verdict=r1_8B.query("index == @proposition_ids").verdict,
            llama_70B_verdict=llama_70B.query("index == @proposition_ids").verdict,
            r1_70B_verdict=r1_70B.query("index == @proposition_ids").verdict,
            llama_8B_reason=llama_8B.query("index == @proposition_ids").reason,
            r1_8B_reason=r1_8B.query("index == @proposition_ids").reason,
            llama_70B_reason=llama_70B.query("index == @proposition_ids").reason,
            r1_70B_reason=r1_70B.query("index == @proposition_ids").reason,
            r1_8B_reasoning_chain=r1_8B.query("index == @proposition_ids").reasoning_chain,
            r1_8B_reasoning_final_answer=r1_8B.query(
                "index == @proposition_ids"
            ).reasoning_final_answer,
            r1_70B_reasoning_chain=r1_70B.query("index == @proposition_ids").reasoning_chain,
            r1_70B_reasoning_final_answer=r1_70B.query(
                "index == @proposition_ids"
            ).reasoning_final_answer,
        )
    )
    return error_df


def print_error_analysis(
    error_df: pd.DataFrame,
    include_models: list[str] = ["Llama-8B", "R1-8B", "Llama-70B", "R1-70B"],
    proposition_id: str | None = None,
    sample_index: int | None = None,
) -> None:
    """Print Error Analysis Comparison for Different Models.

    Select a specific proposition either by `proposition_id` or `sample_index`.
    If none is specified, then the first proposition from `error_df` is visualized.
    """
    # Select one of the Propositions for Detailed Error Analysis
    if proposition_id is not None:
        item = error_df.query("index == @proposition_id").iloc[0]
    elif sample_index is not None:
        item = error_df.iloc[sample_index]
    else:
        item = error_df.iloc[0]

    proposition_id = item.name
    proposition_text = item.text

    human_gt_verdict = item.human_gt_verdict
    llama_8B_verdict = item.llama_8B_verdict
    llama_70B_verdict = item.llama_70B_verdict
    r1_8B_verdict = item.r1_8B_verdict
    r1_70B_verdict = item.r1_70B_verdict

    llama_8B_reason = item.llama_8B_reason
    llama_70B_reason = item.llama_70B_reason
    r1_8B_reason = item.r1_8B_reason
    r1_70B_reason = item.r1_70B_reason

    r1_8B_reasoning_chain = item.r1_8B_reasoning_chain
    r1_8B_reasoning_final_answer = item.r1_8B_reasoning_final_answer
    r1_70B_reasoning_chain = item.r1_70B_reasoning_chain
    r1_70B_reasoning_final_answer = item.r1_70B_reasoning_final_answer

    reference_annotation = llama_8B.loc[[proposition_id]].reset_index().iloc[[0]]
    llama_8B_reference_context = attach_reference_payloads(reference_annotation, release_dir).iloc[
        0
    ]["reference"]

    proposition_str = f"""\
============================================
===              Proposition             ===
============================================
Proposition ID: {proposition_id}
Proposition Text: {proposition_text}
Human Ground Truth Verdict: {human_gt_verdict}
\n
"""
    llama_8B_verdict_str = f"""\
============================================
===         Llama-8B Verdicts            ===
============================================
Llama-8B Verdict: {llama_8B_verdict}
Llama-8B Reason: {llama_8B_reason}
\n
"""
    r1_8B_verdict_str = f"""\
============================================
===            R1-8B Verdicts            ===
============================================
R1-8B Verdict: {r1_8B_verdict}
R1-8B Reason: {r1_8B_reason}
R1-8B Reasoning Chain:\n{r1_8B_reasoning_chain}
R1-8B Reasoning Final Answer:\n{r1_8B_reasoning_final_answer}
\n
"""
    llama_70B_verdict_str = f"""\
============================================
===         Llama-70B Verdicts           ===
============================================
Llama-70B Verdict: {llama_70B_verdict}
Llama-70B Reason: {llama_70B_reason}
\n
"""
    r1_70B_verdict_str = f"""\
============================================
===            R1-70B Verdicts           ===
============================================
R1-70B Verdict: {r1_70B_verdict}
R1-70B Reason: {r1_70B_reason}
R1-70B Reasoning Chain:\n{r1_70B_reasoning_chain}
R1-70B Reasoning Final Answer:\n{r1_70B_reasoning_final_answer}
\n
"""
    reference_context_str = f"""\
============================================
===      EHR Facts Reference Context     ===
============================================
{llama_8B_reference_context}
"""

    # Compose Error Analysis String
    error_analysis_str = proposition_str
    for model in include_models:
        match model:
            case "Llama-8B":
                error_analysis_str += llama_8B_verdict_str
            case "R1-8B":
                error_analysis_str += r1_8B_verdict_str
            case "Llama-70B":
                error_analysis_str += llama_70B_verdict_str
            case "R1-70B":
                error_analysis_str += r1_70B_verdict_str

    error_analysis_str += reference_context_str
    return inspect.cleandoc(error_analysis_str)


# %% [markdown]
# ## Reasoning Error Analysis: Llama-70B vs. R1-70B
# %%
print("Reasoning Analysis: Propositions where Llama-70B Fails but R1-70B Succeeds")
mask = llama_70B_incorrect & r1_70B_correct
error_df = get_error_analysis_df(mask)
mask.value_counts()
# %%
err_str = print_error_analysis(
    error_df,
    include_models=["Llama-70B", "R1-70B"],
    # sample_index=1,
    proposition_id="014f38a6-3bbf-9b00-77d0-da9805163f75",
)
print(err_str)
# %%
# %%
print("Reasoning Analysis: Propositions where Llama-70B Succeeds but R1-70B Fails")
mask = llama_70B_correct & r1_70B_incorrect
mask.value_counts()
# %% [markdown]
# ## Model Size Error Analysis: R1-8B vs. R1-70B
# %%
print("Model Size Analysis: Propositions where R1-8B Fails but R1-70B Succeeds")
mask = r1_8B_incorrect & r1_70B_correct
error_df = get_error_analysis_df(mask)
mask.value_counts()
# %%
err_str = print_error_analysis(
    error_df,
    include_models=["R1-8B", "R1-70B"],
    proposition_id="04a3e2ee-1183-fbfd-b24d-413a90da7661",
)
print(err_str)

# %%
# %% [markdown]
# ## Reasoning Error Analysis: Llama-8B Fails, R1-8B Succeeds, Llama-70B Succeeds
# %%
print(
    "Reasoning Analysis: Propositions where Llama-8B Fails but R1-8B Succeeds & Llama-70B Succeeds"
)
mask = llama_8B_incorrect & r1_8B_correct & llama_70B_correct
error_df = get_error_analysis_df(mask)
mask.value_counts()
# %%
err_str = print_error_analysis(
    error_df,
    include_models=["Llama-8B", "Llama-70B", "R1-8B", "R1-70B"],
    sample_index=2,
    proposition_id="0208baed-eb56-8a17-8143-0b3f183aac25",
)
print(err_str)
# %%
