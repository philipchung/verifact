# %% [markdown]
# ## Confusion Matricies for Best VeriFact AI System vs. Human Clinician Ground Truth Labels
#
# This script generates confusion matricies comparing classification performance for
# all LLM-as-a-Judge for each scenario:
# 1. Llama-8B vs. Llama-70B
# 2. R1-8B vs. R1-70B
# 3. Llama-8B vs. R1-8B
# 4. Llama-70B vs. R1-70B

# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pandas.api.types import CategoricalDtype
from release_data import load_release_annotations, load_release_ground_truth
from sklearn.metrics import confusion_matrix
from utils import load_environment

load_environment()
analysis_dir = Path(os.environ["PROJECT_DIR"]) / "scripts" / "analysis_verifact"
release_dir = Path(os.environ["VERIFACTBHC_DATASET_DIR"])
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

models = ["Llama-8B", "Llama-70B", "R1-8B", "R1-70B"]
human_gt = load_release_ground_truth(release_dir)
ai_verdicts = load_release_annotations(release_dir, models=models)
# %%
# Set Scenario (Hyperparameters)
author_type = "llm"
proposition_type = "claim"
fact_type = "sentence"
top_n = 100
# fact_type = "claim"
# top_n = 150

# Get Human Ground Truth Labels for Scenario
gt = (
    human_gt.query(f"author_type == '{author_type}' & proposition_type == '{proposition_type}'")
    .set_index("proposition_id")
    .sort_index()
)

# Llama-8B Results for Scenario
model = "Llama-8B"
llama_8B = (
    ai_verdicts.query(
        f"model == '{model}' & author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & fact_type == '{fact_type}' &"
        f"top_n == {top_n} & retrieval_method == 'rerank' & reference_format == 'absolute time' & "
        f"reference_only_admission == True"
    )
    .set_index("proposition_id")
    .sort_index()
)

# Llama-70B Results for Scenario
model = "Llama-70B"
llama_70B = (
    ai_verdicts.query(
        f"model == 'Llama-70B' & author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & fact_type == '{fact_type}' &"
        f"top_n == {top_n} & retrieval_method == 'rerank' & reference_format == 'absolute time' & "
        f"reference_only_admission == True"
    )
    .set_index("proposition_id")
    .sort_index()
)

# R1-8B Results for Scenario
model = "R1-8B"
r1_8B = (
    ai_verdicts.query(
        f"model == '{model}' & author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & fact_type == '{fact_type}' &"
        f"top_n == {top_n} & retrieval_method == 'rerank' & reference_format == 'absolute time' & "
        f"reference_only_admission == True"
    )
    .set_index("proposition_id")
    .sort_index()
)

# R1-70B Results for Scenario
model = "R1-70B"
r1_70B = (
    ai_verdicts.query(
        f"model == '{model}' & author_type == '{author_type}' & "
        f"proposition_type == '{proposition_type}' & fact_type == '{fact_type}' &"
        f"top_n == {top_n} & retrieval_method == 'rerank' & reference_format == 'absolute time' & "
        f"reference_only_admission == True"
    )
    .set_index("proposition_id")
    .sort_index()
)

# Get Mask for Correct and Incorrect Verdicts when Compared to Ground Truth
llama_8B_correct = llama_8B.verdict == gt.verdict
llama_8B_incorrect = llama_8B.verdict != gt.verdict
llama_70B_correct = llama_70B.verdict == gt.verdict
llama_70B_incorrect = llama_70B.verdict != gt.verdict
r1_8B_correct = r1_8B.verdict == gt.verdict
r1_8B_incorrect = r1_8B.verdict != gt.verdict
r1_70B_correct = r1_70B.verdict == gt.verdict
r1_70B_incorrect = r1_70B.verdict != gt.verdict

# 2x2 Confusion Matrix Plot
CORRECT_DTYPE = CategoricalDtype(["Correct", "Incorrect"], ordered=True)
labels = CORRECT_DTYPE.categories.tolist()


def make_conf_mat_plot(
    df: pd.DataFrame,
    x_var: str,
    y_var: str,
    labels: list = labels,
    vmax: int = 200,
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
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
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), layout="tight")
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        annot_kws={"fontsize": 9},
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
    ax.set_title(f"{x_var} vs. {y_var}", size=11)
    ax.set_ylabel(y_var, size=10)
    ax.set_yticklabels(labels=labels, rotation=0, size=9)
    ax.set_xlabel(x_var, size=10)
    ax.set_xticklabels(labels=labels, rotation=0, size=9)
    ax.tick_params(axis="both", which="both", left=False, bottom=False)
    return fig, ax


# Figure for 2x2 Confusion Matrix Comparing Non-Reasoning vs. Reasoning Models
# (Llama-3.1-8B-Instruct vs. R1-Distill-Llama-8B) and
# (Llama-3.1-70B-Instruct vs. R1-Distill-Llama-70B)
data = (
    pd.concat(
        {
            "Llama-3.1-8B-Instruct": llama_8B_correct,
            "R1-Distill-Llama-8B": r1_8B_correct,
            "Llama-3.1-70B-Instruct": llama_70B_correct,
            "R1-Distill-Llama-70B": r1_70B_correct,
        },
        axis="columns",
    )
    .map(lambda x: "Correct" if x else "Incorrect")
    .astype(CORRECT_DTYPE)
)

### Create Figure with 2 Subfigures (One for EHR Fact Type as Atomic Claim, One for Sentence)
fig = plt.figure(layout="constrained", figsize=(5.5, 5))
subfig1, subfig2 = fig.subfigures(nrows=2, ncols=1, hspace=0.1)
## SubFigure 1: Performance Advantage From Increasing Model Size
(ax1, ax2) = subfig1.subplots(nrows=1, ncols=2)
make_conf_mat_plot(
    df=data,
    x_var="Llama-3.1-8B-Instruct",
    y_var="Llama-3.1-70B-Instruct",
    labels=labels,
    vmax=750,
    fig=fig,
    ax=ax1,
)
make_conf_mat_plot(
    df=data,
    x_var="R1-Distill-Llama-8B",
    y_var="R1-Distill-Llama-70B",
    labels=labels,
    vmax=750,
    fig=fig,
    ax=ax2,
)
ax1.set_title("Non-reasoning Models", size=11)
ax2.set_title("Reasoning Models", size=11)
subfig1.suptitle("Performance Advantage From Increasing Model Parameters", size=12)
## SubFigure 2: Performance Advantage From Reasoning Capability
(ax3, ax4) = subfig2.subplots(nrows=1, ncols=2)
make_conf_mat_plot(
    df=data,
    x_var="Llama-3.1-8B-Instruct",
    y_var="R1-Distill-Llama-8B",
    labels=labels,
    vmax=750,
    fig=fig,
    ax=ax3,
)
make_conf_mat_plot(
    df=data,
    x_var="Llama-3.1-70B-Instruct",
    y_var="R1-Distill-Llama-70B",
    labels=labels,
    vmax=750,
    fig=fig,
    ax=ax4,
)
ax3.set_title("8B Parameter Models", size=11)
ax4.set_title("70B Parameter Models", size=11)
subfig2.suptitle("Performance Advantage From Reasoning Capability", size=12)

# Save Figure
save_dir = analysis_dir / "9a_confusion_matrices_performance_advantage"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / "confmat_performance_advantage.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
# %%
