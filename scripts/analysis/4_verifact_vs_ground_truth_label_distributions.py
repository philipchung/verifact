# %% [markdown]
# ## Label Distributions for VeriFact vs. Human Clinican Ground Truth
# %%
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from matplotlib.gridspec import GridSpec
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_environment, load_pandas, save_pandas

load_environment()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

annotated_dataset_dir = Path(os.environ["ANNOTATED_DATASET_DIR"])
output_dir = Path.cwd() / "4_verifact_vs_ground_truth_label_distributions"
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


# Split Human Ground Truth into 4 Categories
human_gt_grps = {
    "llm_claim": human_gt.query("author_type == 'llm' & proposition_type == 'claim'"),
    "llm_sentence": human_gt.query("author_type == 'llm' & proposition_type == 'sentence'"),
    "human_claim": human_gt.query("author_type == 'human' & proposition_type == 'claim'"),
    "human_sentence": human_gt.query("author_type == 'human' & proposition_type == 'sentence'"),
}
# Binarize Human Ground Truth Verdicts (Supported, Not Supported)
human_gt_grps_binarized = {k: binarize_verdicts_in_df(v) for k, v in human_gt_grps.items()}

# Split VeriFact Verdicts into 4 Categories
ai_grps = {
    "llm_claim": ai_verdicts.query("author_type == 'llm' & proposition_type == 'claim'"),
    "llm_sentence": ai_verdicts.query("author_type == 'llm' & proposition_type == 'sentence'"),
    "human_claim": ai_verdicts.query("author_type == 'human' & proposition_type == 'claim'"),
    "human_sentence": ai_verdicts.query("author_type == 'human' & proposition_type == 'sentence'"),
}
# Binarize VeriFact Verdicts (Supported, Not Supported)
ai_grps_binarized = {k: binarize_verdicts_in_df(v) for k, v in ai_grps.items()}
# %%
# VeriFact - Select "Best" VeriFact, then get Label Distribution

# Select "Best VeriFact AI System":
# * Retrieval Method = Rerank
# * Top N = 50
# * Reference Format = Absolute Time
# * Only Current Admission = False
best_ai_grps = {
    k: v.query(
        "retrieval_method == 'rerank' & "
        "top_n == 50 & "
        "reference_format == 'absolute_time' & "
        "reference_only_admission == False"
    )
    for k, v in ai_grps.items()
}
ai_dist = pd.DataFrame(
    {k: v.verdict.value_counts(normalize=True) for k, v in best_ai_grps.items()}
).reindex(["Supported", "Not Supported", "Not Addressed"])
ai_dist_binarized = pd.DataFrame(
    {k: v.verdict.value_counts(normalize=True) for k, v in ai_grps_binarized.items()}
).reindex(["Supported", "Not Supported"])


# %%
# Human Ground Truth - Label Distribution

# Label Distribution for Human Ground Truth
human_gt_dist = pd.DataFrame(
    {k: v.verdict.value_counts(normalize=True) for k, v in human_gt_grps.items()}
).reindex(["Supported", "Not Supported", "Not Addressed"])
# Label Distribution for Human Ground Truth (Binarized)
human_gt_dist_binarized = pd.DataFrame(
    {k: v.verdict.value_counts(normalize=True) for k, v in human_gt_grps_binarized.items()}
).reindex(["Supported", "Not Supported"])


# %%
# Combine 3-Label Distributions into a single DataFrame
verdict_dist = pd.concat(
    {"Human Ground Truth": human_gt_dist, "Best VeriFact AI System": ai_dist},
    axis="index",
    names=["Rater", "Verdict"],
)
# Combine 2-Label Distributions into a single DataFrame
verdict_dist_binarized = pd.concat(
    {
        "Human Ground Truth": human_gt_dist_binarized,
        "Best VeriFact AI System": ai_dist_binarized,
    },
    axis="index",
    names=["Rater", "Verdict"],
)

# Label Distributions Table for Human Ground Truth & Best VeriFact AI System
df = pd.concat(
    {
        "Original": verdict_dist,
        "Binarized": verdict_dist_binarized,
    },
    axis="index",
    names=["Label Space", "Rater", "Verdict"],
)
# Create Multi-Index
author_type = ["LLM-written", "Human-written"]
proposition_type = ["Atomic Claim", "Sentence"]
df.columns = pd.MultiIndex(
    levels=[author_type, proposition_type],
    codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
    names=["Author Type", "Proposition Type"],
)
display(df)
# %%
# Save Label Distribution Table
save_pandas(df=df, filepath=output_dir / "label_distribution.csv", index=True)
save_pandas(df=df, filepath=output_dir / "label_distribution.xlsx")

# %%
# Compute Sens/Spec/PPV/NPV of Best VeriFact AI System for each Subgroup in Original Label Space
# (LLM-written, Human-written) x (Atomic Claim, Sentence)
# where we use the Human Ground Truth Labels as the reference standard.


def compute_sens_spec_ppv_npv(
    y_true: Sequence, y_pred: Sequence, labels: Sequence
) -> dict[Any, Any]:
    "Compute Sensitivity, Specificity, PPV, NPV, TP, FP, FN, TN, Support for each class."
    output: dict[str, float | int] = {}
    # Confusion matrix whose i-th row and j-th column entry indicates the number of samples
    # with true label being i-th class and predicted label being j-th class.
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    # Sens/Spec/PPV/NPV for each class
    for i, label in enumerate(labels):
        # Extract TP, FP, FN, TN from Confusion Matrix
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        # Support
        support = tp + fn
        # Compute Sensitivity (TPR, Recall), Specificity (TNR), PPV (Precision), NPV
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        output[label] = {
            "TPR": tpr,
            "TNR": tnr,
            "PPV": ppv,
            "NPV": npv,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Support": int(support),
        }
    return output


sklearn_cm = {}
sklearn_cr = {}
confmat_metrics = {}
binarized_sklearn_cm = {}
binarized_sklearn_cr = {}
binarized_confmat_metrics = {}

for input_text_type in input_text_types:
    # Select Data - Original Label Space
    best_ai = (
        best_ai_grps[input_text_type]
        .loc[:, ["proposition_id", "verdict"]]
        .set_index("proposition_id")
        .sort_index()
    )
    human_gt = (
        human_gt_grps[input_text_type]
        .loc[:, ["proposition_id", "verdict"]]
        .set_index("proposition_id")
        .sort_index()
    )
    labels = ["Supported", "Not Supported", "Not Addressed"]
    # Compute sklearn metrics
    y_true = human_gt
    y_pred = best_ai
    sklearn_cm[input_text_type] = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    sklearn_cr[input_text_type] = classification_report(y_true=y_true, y_pred=y_pred, labels=labels)
    # Compute self-defined metrics
    confmat_metrics[input_text_type] = compute_sens_spec_ppv_npv(
        y_true=y_true, y_pred=y_pred, labels=labels
    )

    # Select Data - Binarized Label Space
    best_ai_binarized = binarize_verdicts_in_df(best_ai)
    human_gt_binarized = binarize_verdicts_in_df(human_gt)
    labels = ["Supported", "Not Supported"]
    # Compute sklearn metrics
    y_true = human_gt_binarized
    y_pred = best_ai_binarized
    binarized_sklearn_cm[input_text_type] = confusion_matrix(
        y_true=y_true, y_pred=y_pred, labels=labels
    )
    binarized_sklearn_cr[input_text_type] = classification_report(
        y_true=y_true, y_pred=y_pred, labels=labels
    )
    # Compute self-defined metrics
    binarized_confmat_metrics[input_text_type] = compute_sens_spec_ppv_npv(
        y_true=y_true, y_pred=y_pred, labels=labels
    )

# %%
# Metrics for Each Input Text Type & Each Label - Original Label Space
metrics_table = (
    pd.concat(
        {k: pd.DataFrame(v) for k, v in confmat_metrics.items()},
        names=["Input Text Type", "Metric"],
    )
    .T.stack()
    .rename_axis(index=["Label", "Metric"])
)
metrics_table = metrics_table.drop(index=["TP", "FP", "FN", "TN", "Support"], level="Metric")
save_pandas(df=metrics_table, filepath=output_dir / "metrics_table.csv", index=True)
save_pandas(df=metrics_table, filepath=output_dir / "metrics_table.xlsx")
display(metrics_table)
# %%
# Metrics for Each Input Text Type & Each Label - Binarized Label Space
binarized_metrics_table = (
    pd.concat(
        {k: pd.DataFrame(v) for k, v in binarized_confmat_metrics.items()},
        names=["Input Text Type", "Metric"],
    )
    .T.stack()
    .rename_axis(index=["Label", "Metric"])
)
binarized_metrics_table = binarized_metrics_table.drop(
    index=["TP", "FP", "FN", "TN", "Support"], level="Metric"
)
save_pandas(
    df=binarized_metrics_table, filepath=output_dir / "binarized_metrics_table.csv", index=True
)
save_pandas(df=binarized_metrics_table, filepath=output_dir / "binarized_metrics_table.xlsx")
display(binarized_metrics_table)
# %%
# # Visualize Confusion Matrix
# for input_text_type in input_text_types:
#     print(f"Input Text Type: {input_text_type}")
#     cm = sklearn_cm[input_text_type]
#     labels = ["Supported", "Not Supported", "Not Addressed"]
#     # Confusion Matrix
#     display(ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot())
#     # Classification Report
#     cr = sklearn_cr[input_text_type]
#     print(cr)

# %%
# ## Create Stacked Bar Plot for Label Distribution

# Map of Group Name to Input Text Type Label (for X-axis)
grp_map = {
    "llm_claim": "LLM-written Text,\nAtomic Claim Proposition",
    "llm_sentence": "LLM-written Text,\nSentence Proposition",
    "human_claim": "Human-written Text,\nAtomic Claim Proposition",
    "human_sentence": "Human-written Text,\nSentence Proposition",
}
df.columns = list(grp_map.keys())
original_label_space = df.xs("Original", level="Label Space", axis="index")
human_gt = original_label_space.xs("Human Ground Truth", level="Rater", axis="index").fillna(0.0)
ai = original_label_space.xs("Best VeriFact AI System", level="Rater", axis="index").fillna(0.0)

# Create Figure & Axes
fig = plt.figure(layout="constrained", figsize=(8.5, 6))
gs = GridSpec(nrows=1, ncols=1, figure=fig, hspace=0.2)
p1 = fig.add_subplot(gs[0])
# p2 = fig.add_subplot(gs[2:4])
# Plot colors
colors = sns.color_palette("tab10")
supported_color = "green"
supported_color2 = "mediumseagreen"
supported_text_color = "green"
not_supported_color = "red"
not_supported_color2 = "salmon"
not_supported_text_color = "red"
not_addressed_color = "orange"
not_addressed_color2 = "sandybrown"
not_addressed_text_color = "darkorange"

## Define Groups
grps = human_gt.columns
## Define X-axis
x_label_locations = np.arange(len(grps))  # the label locations
bias = -0.5  # the x-position bias for the first bar
width = 0.3  # the width of the bars

## Create Bars for groups ("LLM Claim", "LLM Sentence", "Human Claim", "Human Sentence")
bar_containers = {}
for i, grp in enumerate(grps):
    offset = width + bias
    offset2 = 2 * width + bias

    grp_human_gt_supported = human_gt[grp]["Supported"]
    grp_human_gt_not_supported = human_gt[grp]["Not Supported"]
    grp_human_gt_not_addressed = human_gt[grp]["Not Addressed"]
    grp_ai_supported = ai[grp]["Supported"]
    grp_ai_not_supported = ai[grp]["Not Supported"]
    grp_ai_not_addressed = ai[grp]["Not Addressed"]

    # Plot Stacked Bar for Human Ground truth
    human_gt_supported_rects = p1.bar(
        x=i + offset,
        height=grp_human_gt_supported,
        width=width,
        label=grp,
        color=supported_color,
        edgecolor="white",
    )
    human_gt_not_supported_rects = p1.bar(
        x=i + offset,
        height=grp_human_gt_not_supported,
        bottom=grp_human_gt_supported,
        width=width,
        label=grp,
        color=not_supported_color,
        edgecolor="white",
    )
    human_gt_not_addressed_rects = p1.bar(
        x=i + offset,
        height=grp_human_gt_not_addressed,
        bottom=grp_human_gt_supported + grp_human_gt_not_supported,
        width=width,
        label=grp,
        color=not_addressed_color,
        edgecolor="white",
    )
    p1.bar_label(
        human_gt_supported_rects,
        padding=0,
        label_type="center",
        fmt="{:.1%}",
        color="white",
        fontweight="heavy",
    )
    # p1.bar_label(
    #     human_gt_not_supported_rects,
    #     padding=0,
    #     label_type="center",
    #     fmt="{:.1%}",
    #     color="white",
    #     fontweight="heavy",
    # )
    # p1.bar_label(
    #     human_gt_not_addressed_rects,
    #     label="Hi",
    #     padding=0,
    #     label_type="center",
    #     fmt="{:.1%}",
    #     color="white",
    #     fontweight="heavy",
    # )
    bar_containers[grp] = {
        "human_gt": {
            "Supported": human_gt_supported_rects,
            "Not Supported": human_gt_not_supported_rects,
            "Not Addressed": human_gt_not_addressed_rects,
        }
    }

    # Plot Stacked Bar for Mean VeriFact
    ai_supported_rects = p1.bar(
        x=i + offset2,
        height=grp_ai_supported,
        width=width,
        label=grp,
        color=supported_color,
        hatch="//////",
        alpha=0.5,
        edgecolor="white",
    )
    ai_not_supported_rects = p1.bar(
        x=i + offset2,
        height=grp_ai_not_supported,
        bottom=grp_ai_supported,
        width=width,
        label=grp,
        color=not_supported_color,
        hatch="//////",
        alpha=0.5,
        edgecolor="white",
    )
    ai_not_addressed_rects = p1.bar(
        x=i + offset2,
        height=grp_ai_not_addressed,
        bottom=grp_ai_supported + grp_ai_not_supported,
        width=width,
        label=grp,
        color=not_addressed_color,
        hatch="//////",
        alpha=0.5,
        edgecolor="white",
    )
    p1.bar_label(
        ai_supported_rects,
        padding=0,
        label_type="center",
        fmt="{:.1%}",
        color="black",
        fontweight="heavy",
    )
    # p1.bar_label(
    #     ai_not_supported_rects,
    #     padding=0,
    #     label_type="center",
    #     fmt="{:.1%}",
    #     color="black",
    #     fontweight="heavy",
    # )
    # p1.bar_label(
    #     ai_not_addressed_rects,
    #     padding=0,
    #     label_type="center",
    #     fmt="{:.1%}",
    #     color="black",
    #     fontweight="heavy",
    # )
    bar_containers[grp] |= {
        "ai": {
            "Supported": ai_supported_rects,
            "Not Supported": ai_not_supported_rects,
            "Not Addressed": ai_not_addressed_rects,
        }
    }

## Manually Position Annotations Relative to the Bars
# for Not Supported & Not Addressed Labels
# (Supported Labels are Centered in Bar by Default)
# -- LLM-Claim -- Human Ground Truth --
# Not Addressed
rater = "human_gt"
group = "llm_claim"
label = "Not Addressed"
value = human_gt.loc[label, group]
bar = bar_containers[group][rater][label]
patch = bar.patches[0]
# Get Bounding Box Corners for Patch
(x0, y0), (x1, y1) = patch.get_bbox().get_points()
x_arrowhead = x0
y_arrowhead = y0 + (y1 - y0) / 2
text_xoffset = -0.3
text_yoffset = 0.025
x_text = x0 + text_xoffset
y_text = y_arrowhead + text_yoffset
plt.annotate(
    text=f"{value:.1%}",
    color=not_addressed_text_color,
    fontweight="heavy",
    xy=(x_arrowhead, y_arrowhead),
    xytext=(x_text, y_text),
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
    },
)
# Not Supported
rater = "human_gt"
group = "llm_claim"
label = "Not Supported"
value = human_gt.loc[label, group]
bar = bar_containers[group][rater][label]
patch = bar.patches[0]
# Get Bounding Box Corners for Patch
(x0, y0), (x1, y1) = patch.get_bbox().get_points()
x_arrowhead = x0
y_arrowhead = y0 + (y1 - y0) / 2
text_xoffset = -0.3
text_yoffset = -0.05
x_text = x0 + text_xoffset
y_text = y_arrowhead + text_yoffset
plt.annotate(
    text=f"{value:.1%}",
    color=not_supported_text_color,
    fontweight="heavy",
    xy=(x_arrowhead, y_arrowhead),
    xytext=(x_text, y_text),
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
    },
)

# -- LLM-Claim -- VeriFact --
# Not Addressed
rater = "ai"
group = "llm_claim"
label = "Not Addressed"
value = ai.loc[label, group]
bar = bar_containers[group][rater][label]
patch = bar.patches[0]
# Get Bounding Box Corners for Patch
(x0, y0), (x1, y1) = patch.get_bbox().get_points()
x_arrowhead = x1
y_arrowhead = y0 + (y1 - y0) / 2
text_xoffset = 0.4
text_yoffset = 0.025
x_text = x0 + text_xoffset
y_text = y_arrowhead + text_yoffset
plt.annotate(
    text=f"{value:.1%}",
    color=not_addressed_text_color,
    fontweight="heavy",
    xy=(x_arrowhead, y_arrowhead),
    xytext=(x_text, y_text),
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
    },
)
# Not Supported
rater = "ai"
group = "llm_claim"
label = "Not Supported"
bar = bar_containers[group][rater][label]
p1.bar_label(
    bar,
    padding=0,
    label_type="center",
    fmt="{:.1%}",
    color="black",
    fontweight="heavy",
)

# -- LLM-Sentence -- Human Ground Truth --
# Not Addressed
rater = "human_gt"
group = "llm_sentence"
label = "Not Addressed"
value = human_gt.loc[label, group]
bar = bar_containers[group][rater][label]
patch = bar.patches[0]
# Get Bounding Box Corners for Patch
(x0, y0), (x1, y1) = patch.get_bbox().get_points()
x_arrowhead = x0
y_arrowhead = y0 + (y1 - y0) / 2
text_xoffset = -0.3
text_yoffset = -0.04
x_text = x0 + text_xoffset
y_text = y_arrowhead + text_yoffset
plt.annotate(
    text=f"{value:.1%}",
    color=not_addressed_text_color,
    fontweight="heavy",
    xy=(x_arrowhead, y_arrowhead),
    xytext=(x_text, y_text),
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
    },
)
# Not Supported
rater = "human_gt"
group = "llm_sentence"
label = "Not Supported"
value = human_gt.loc[label, group]
bar = bar_containers[group][rater][label]
patch = bar.patches[0]
# Get Bounding Box Corners for Patch
(x0, y0), (x1, y1) = patch.get_bbox().get_points()
x_arrowhead = x0
y_arrowhead = y0 + (y1 - y0) / 2
text_xoffset = -0.3
text_yoffset = -0.06
x_text = x0 + text_xoffset
y_text = y_arrowhead + text_yoffset
plt.annotate(
    text=f"{value:.1%}",
    color=not_supported_text_color,
    fontweight="heavy",
    xy=(x_arrowhead, y_arrowhead),
    xytext=(x_text, y_text),
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
    },
)

# -- LLM-Sentence -- VeriFact --
# Not Addressed
rater = "ai"
group = "llm_sentence"
label = "Not Addressed"
value = ai.loc[label, group]
bar = bar_containers[group][rater][label]
patch = bar.patches[0]
# Get Bounding Box Corners for Patch
(x0, y0), (x1, y1) = patch.get_bbox().get_points()
x_arrowhead = x1
y_arrowhead = y0 + (y1 - y0) / 2
text_xoffset = 0.4
text_yoffset = 0.025
x_text = x0 + text_xoffset
y_text = y_arrowhead + text_yoffset
plt.annotate(
    text=f"{value:.1%}",
    color=not_addressed_text_color,
    fontweight="heavy",
    xy=(x_arrowhead, y_arrowhead),
    xytext=(x_text, y_text),
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
    },
)
# Not Supported
rater = "ai"
group = "llm_sentence"
label = "Not Supported"
value = ai.loc[label, group]
bar = bar_containers[group][rater][label]
patch = bar.patches[0]
# Get Bounding Box Corners for Patch
(x0, y0), (x1, y1) = patch.get_bbox().get_points()
x_arrowhead = x1
y_arrowhead = y0 + (y1 - y0) / 2
text_xoffset = 0.4
text_yoffset = -0.05
x_text = x0 + text_xoffset
y_text = y_arrowhead + text_yoffset
plt.annotate(
    text=f"{value:.1%}",
    color=not_supported_text_color,
    fontweight="heavy",
    xy=(x_arrowhead, y_arrowhead),
    xytext=(x_text, y_text),
    arrowprops={
        "arrowstyle": "->",
        "color": "black",
    },
)

# -- Human-Claim -- Human Ground Truth --
# Not Addressed
rater = "human_gt"
group = "human_claim"
label = "Not Addressed"
bar = bar_containers[group][rater][label]
p1.bar_label(
    bar,
    padding=0,
    label_type="center",
    fmt="{:.1%}",
    color="white",
    fontweight="heavy",
)

# Not Supported
rater = "human_gt"
group = "human_claim"
label = "Not Supported"
bar = bar_containers[group][rater][label]
p1.bar_label(
    bar,
    padding=0,
    label_type="center",
    fmt="{:.1%}",
    color="white",
    fontweight="heavy",
)

# -- Human-Claim -- VeriFact --
# Not Addressed
rater = "ai"
group = "human_claim"
label = "Not Addressed"
bar = bar_containers[group][rater][label]
p1.bar_label(
    bar,
    padding=0,
    label_type="center",
    fmt="{:.1%}",
    color="black",
    fontweight="heavy",
)

# Not Supported
rater = "ai"
group = "human_claim"
label = "Not Supported"
bar = bar_containers[group][rater][label]
p1.bar_label(
    bar,
    padding=0,
    label_type="center",
    fmt="{:.1%}",
    color="black",
    fontweight="heavy",
)

# -- Human-Sentence -- Human Ground Truth --
# Not Addressed
rater = "human_gt"
group = "human_sentence"
label = "Not Addressed"
bar = bar_containers[group][rater][label]
p1.bar_label(
    bar,
    padding=0,
    label_type="center",
    fmt="{:.1%}",
    color="white",
    fontweight="heavy",
)

# Not Supported
rater = "human_gt"
group = "human_sentence"
label = "Not Supported"
bar = bar_containers[group][rater][label]
p1.bar_label(
    bar,
    padding=0,
    label_type="center",
    fmt="{:.1%}",
    color="white",
    fontweight="heavy",
)

# -- Human-Sentence -- VeriFact --
# Not Addressed
rater = "ai"
group = "human_sentence"
label = "Not Addressed"
bar = bar_containers[group][rater][label]
p1.bar_label(
    bar,
    padding=0,
    label_type="center",
    fmt="{:.1%}",
    color="black",
    fontweight="heavy",
)

# Not Supported
rater = "ai"
group = "human_sentence"
label = "Not Supported"
bar = bar_containers[group][rater][label]
p1.bar_label(
    bar,
    padding=0,
    label_type="center",
    fmt="{:.1%}",
    color="black",
    fontweight="heavy",
)


# Format X-axis
p1.set_xticks(ticks=x_label_locations, labels=list(grp_map.values()))
p1.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=2,
    frameon=False,
    # title="Input Text Type",
    handles=[
        human_gt_supported_rects,
        human_gt_not_supported_rects,
        human_gt_not_addressed_rects,
        ai_supported_rects,
        ai_not_supported_rects,
        ai_not_addressed_rects,
    ],
    labels=[
        "Supported (Human Ground Truth)",
        "Not Supported (Human Ground Truth)",
        "Not Addressed (Human Ground Truth)",
        "Supported (Best VeriFact AI System)",
        "Not Supported (Best VeriFact AI System)",
        "Not Addressed (Best VeriFact AI System)",
    ],
)
# Remove X-tick, but Keep X-Labels
plt.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=True,  # keep X-Labels
)
# Hide Y-axis
p1.get_yaxis().set_visible(False)
# Hide Spines
p1.spines["top"].set_visible(False)
p1.spines["right"].set_visible(False)
p1.spines["left"].set_visible(False)
# Set Title
p1.set_title("Label Distribution Assigned by Human Clinicians vs. VeriFact", pad=30)
# Save figure
fig.savefig(output_dir / "label_distribution.png", dpi=300, bbox_inches="tight")
# %%
