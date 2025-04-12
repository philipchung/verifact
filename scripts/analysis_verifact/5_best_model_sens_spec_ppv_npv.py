# %% [markdown]
# ## Classification Performance of Best VeriFact AI System vs. Human Clinician Ground Truth Labels
#
# This script identifies the best VeriFact AI system and compares its classification performance
# against the human clinician ground truth labels.
#
# The following classification metrics are computed for each label
# ("Supported", "Not Supported", "Not Addressed"):
# * Sensitivity
# * Specificity
# * Positive Predictive Value (PPV)
# * Negative Predictive Value (NPV)
#
# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.display import display
from irr_metrics import ClassificationMetrics, MetricBunch, binarize_verdicts_in_df, coerce_types
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_environment, load_pandas, save_text
from utils.file_utils import save_pandas

load_environment()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
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
    .loc[
        :,
        ["proposition_id", "text", "author_type", "proposition_type", "rater_name", "verdict"],
    ]
)

# Map of VeriFact Result Directories for Each Model
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
## Load Agreement Metrics for VeriFact AI Systems Computed in Original Label Space
metric = "percent_agreement"
name = f"ai_rater_{metric}_ci"
analysis_save_dir = Path.cwd() / "2_compute_verifact_agreement"
mb = MetricBunch.load(save_dir=analysis_save_dir, name=name)

# %%
# Compute Sens/Spec/PPV/NPV of Best VeriFact AI System for each Subgroup in Original Label Space
# (LLM-written, Human-written) x (Atomic Claim, Sentence)
# where we use the Human Ground Truth Labels as the reference standard.

# ## Set Author Type & Proposition Type
author_type = "llm"
proposition_type = "claim"

# Get Top Performing VeriFact AI System
best_ai = (
    mb.metrics.query(f"author_type == '{author_type}' & proposition_type == '{proposition_type}'")
    .sort_values(by="value", ascending=False)
    .head(1)
    .squeeze()
)
model = best_ai.model
fact_type = best_ai.fact_type
top_n = best_ai.top_n
retrieval_method = best_ai.retrieval_method
reference_format = best_ai.reference_format
reference_only_admission = best_ai.reference_only_admission

# Get Verdicts for Best AI System
best_ai_verdicts = ai_verdicts.query(
    f"model == '{model}' & author_type == '{author_type}' & "
    f"proposition_type == '{proposition_type}' & fact_type == '{fact_type}' & "
    f"top_n == {top_n} & retrieval_method == '{retrieval_method}' & "
    f"reference_format == '{reference_format}' & "
    f"reference_only_admission == {reference_only_admission}"
)

# Get Labels for Computing Metrics in Original Label Space
y_pred = (
    best_ai_verdicts.loc[:, ["proposition_id", "verdict"]].set_index("proposition_id").sort_index()
).squeeze()
human_gt_subset = human_gt.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}'"
)
y_true = (
    human_gt_subset.loc[:, ["proposition_id", "verdict"]].set_index("proposition_id").sort_index()
).squeeze()
labels = ["Supported", "Not Supported", "Not Addressed"]

# Compute Classification Metrics
cm = ClassificationMetrics.from_defaults(
    rater_verdicts=y_pred,
    ground_truth=y_true,
    verdict_labels=labels,
    bootstrap_iterations=1000,
    workers=10,
    show_progress=True,
)
# Convert Classification Metrics (Sens/Spec/PPV/NPV) DataFrame from Wide to Long Format
metrics_table = cm.metrics_table(fmt=".2f")
display(metrics_table)

# Compute Sklearn Classification Report to Sanity Check
cr = classification_report(y_true=y_true, y_pred=y_pred, labels=labels)
print(cr)

# Compute & Display Heatmap of Confusion Matrix
confmat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(4, 4), layout="tight")
sns.heatmap(
    confmat,
    annot=True,
    annot_kws={"size": 14},
    fmt="d",
    vmin=0,  # Minimum colormap value
    vmax=500,  # Maximum colormap value (values above this are clipped to this value)
    cmap="Blues",
    cbar=False,
    linewidths=0.5,
    linecolor="white",
    xticklabels=labels,
    yticklabels=labels,
    square=True,
    ax=ax,
)
ax.set_title(
    f"Best VeriFact AI System for\n"
    f"Author Type: {author_type}, Proposition Type: {proposition_type}",
    size=14,
)
ax.set_ylabel("Human Clinican\nGround Truth", size=12)
ax.set_yticklabels(labels=["Supported", "Not\nSupported", "Not\nAddressed"], rotation=0, size=10)
ax.set_xlabel("VeriFact AI System", size=12)
ax.set_xticklabels(labels=["Supported", "Not\nSupported", "Not\nAddressed"], rotation=0, size=10)
ax.tick_params(axis="both", which="both", bottom=False, left=False)
fig.show()
print(
    "Confusion Matrix comparing classification of propositions between "
    "Human Clinician Ground Truth (rows) and VeriFact AI System (columns)."
    "The propositions being evaluated are derived from LLM-written text "
    "with atomic claim propositions as the proposition type."
)

# Save Results to Disk
name = f"best_{author_type}_{proposition_type}"
save_dir = Path.cwd() / "5_best_model_sens_spec_ppv_npv" / name
save_dir.mkdir(exist_ok=True, parents=True)
# Save Rater Name
save_text(text=best_ai_verdicts.rater_name.iloc[0], filepath=save_dir / "rater_name.txt")
# Save Best AI Agreement Metrics
save_pandas(df=best_ai, filepath=save_dir / "agreement_metrics.csv")
# Save Best AI Verdicts
save_pandas(df=best_ai_verdicts, filepath=save_dir / "verdicts.csv")
# Save Classification Report
save_text(text=cr, filepath=save_dir / "classification_report.txt")
# Save Classification Metrics (Sens/Spec/PPV/NPV) with Confidence Intervals
save_pandas(
    df=metrics_table,
    filepath=save_dir / "sens_spec_ppv_npv.csv",
)
# Save Confusion Matrix Figure
fig.savefig(
    save_dir / "confusionmatrix.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)

# %%
## Load Agreement Metrics for VeriFact AI Systems Computed in Binarized Label Space
metric = "percent_agreement"
name = f"ai_rater_{metric}_ci_binarized"
analysis_save_dir = Path.cwd() / "2_compute_verifact_agreement"
mb_binarized = MetricBunch.load(save_dir=analysis_save_dir, name=name)
# %%
# Compute Sens/Spec/PPV/NPV of Best VeriFact AI System for each Subgroup in Binarized Label Space
# (LLM-written, Human-written) x (Atomic Claim, Sentence)
# where we use the Human Ground Truth Labels as the reference standard.

# ## Set Author Type & Proposition Type
author_type = "llm"
proposition_type = "claim"

# Get Top Performing VeriFact AI System
best_ai = (
    mb_binarized.metrics.query(
        f"author_type == '{author_type}' & proposition_type == '{proposition_type}'"
    )
    .sort_values(by="value", ascending=False)
    .head(1)
    .squeeze()
)
model = best_ai.model
fact_type = best_ai.fact_type
top_n = best_ai.top_n
retrieval_method = best_ai.retrieval_method
reference_format = best_ai.reference_format
reference_only_admission = best_ai.reference_only_admission

# Get Verdicts for Best AI System
best_ai_verdicts = ai_verdicts.query(
    f"model == '{model}' & author_type == '{author_type}' & "
    f"proposition_type == '{proposition_type}' & fact_type == '{fact_type}' & "
    f"top_n == {top_n} & retrieval_method == '{retrieval_method}' & "
    f"reference_format == '{reference_format}' & "
    f"reference_only_admission == {reference_only_admission}"
)

# Get Labels for Computing Metrics in Binarized Label Space
y_pred = (
    best_ai_verdicts.loc[:, ["proposition_id", "verdict"]].set_index("proposition_id").sort_index()
)
human_gt_subset = human_gt.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}'"
)
y_true = (
    human_gt_subset.loc[:, ["proposition_id", "verdict"]].set_index("proposition_id").sort_index()
)
y_pred_binarized = binarize_verdicts_in_df(y_pred).squeeze()
y_true_binarized = binarize_verdicts_in_df(y_true).squeeze()
labels_binarized = ["Supported", "Not Supported or Addressed"]

# Compute Classification Metrics
cm = ClassificationMetrics.from_defaults(
    rater_verdicts=y_pred_binarized,
    ground_truth=y_true_binarized,
    verdict_labels=labels_binarized,
    bootstrap_iterations=1000,
    workers=10,
    show_progress=True,
)
# Convert Classification Metrics (Sens/Spec/PPV/NPV) DataFrame from Wide to Long Format
metrics_table = cm.metrics_table(fmt=".2f")
display(metrics_table)

# Compute Sklearn Classification Report to Sanity Check
cr = classification_report(
    y_true=y_true_binarized, y_pred=y_pred_binarized, labels=labels_binarized
)
print(cr)

# Compute & Display Heatmap of Confusion Matrix
confmat = confusion_matrix(
    y_true=y_true_binarized, y_pred=y_pred_binarized, labels=labels_binarized
)

fig, ax = plt.subplots(figsize=(4, 4), layout="tight")
sns.heatmap(
    confmat,
    annot=True,
    annot_kws={"size": 14},
    fmt="d",
    vmin=0,  # Minimum colormap value
    vmax=500,  # Maximum colormap value (values above this are clipped to this value)
    cmap="Blues",
    cbar=False,
    linewidths=0.5,
    linecolor="white",
    xticklabels=labels_binarized,
    yticklabels=labels_binarized,
    square=True,
    ax=ax,
)
ax.set_title(
    f"Best VeriFact AI System for\n"
    f"Author Type: {author_type}, Proposition Type: {proposition_type}",
    size=14,
)
ax.set_ylabel("Human Clinican\nGround Truth", size=12)
ax.set_yticklabels(labels=["Supported", "Not Supported\n or Addressed"], rotation=0, size=10)
ax.set_xlabel("VeriFact AI System", size=12)
ax.set_xticklabels(labels=["Supported", "Not Supported\n or Addressed"], rotation=0, size=10)
ax.tick_params(axis="both", which="both", bottom=False, left=False)
fig.show()
print(
    "Confusion Matrix comparing classification of propositions between "
    "Human Clinician Ground Truth (rows) and VeriFact AI System (columns)."
    "The propositions being evaluated are derived from LLM-written text "
    "with atomic claim propositions as the proposition type."
)

# Save Results to Disk
name = f"best_{author_type}_{proposition_type}_binarized"
save_dir = Path.cwd() / "5_best_model_sens_spec_ppv_npv" / name
save_dir.mkdir(exist_ok=True, parents=True)
# Save Rater Name
save_text(text=best_ai_verdicts.rater_name.iloc[0], filepath=save_dir / "rater_name.txt")
# Save Best AI Agreement Metrics
save_pandas(
    df=best_ai,
    filepath=save_dir / "agreement_metrics.csv",
)
# Save Best AI Verdicts
save_pandas(df=best_ai_verdicts, filepath=save_dir / "verdicts.csv")
# Save Classification Report
save_text(text=cr, filepath=save_dir / "classification_report.txt")
# Save Classification Metrics (Sens/Spec/PPV/NPV) with Confidence Intervals
save_pandas(
    df=metrics_table,
    filepath=save_dir / "sens_spec_ppv_npv.csv",
)
# Save Confusion Matrix Figure
fig.savefig(
    save_dir / "confusionmatrix.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
# %%
