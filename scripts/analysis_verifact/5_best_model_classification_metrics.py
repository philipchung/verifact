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
from irr_metrics import ClassificationMetrics, MetricBunch
from release_data import load_release_annotations, load_release_ground_truth
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_environment, save_text
from utils.file_utils import save_pandas

load_environment()
analysis_dir = Path(os.environ["PROJECT_DIR"]) / "scripts" / "analysis_verifact"
release_dir = Path(os.environ["VERIFACTBHC_DATASET_DIR"])
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

models = [
    "Llama-8B",
    "Llama-70B",
    "R1-8B",
    "R1-70B",
    "Gemma3-12B",
    "Gemma3-27B",
    "Qwen3-32B",
    "Qwen3-30B-A3B-Instruct",
    "Qwen3-30B-A3B-Thinking",
]
human_gt = load_release_ground_truth(release_dir)
ai_verdicts = load_release_annotations(release_dir, models=models)

# Load PERCENT AGREEMENT MetricBunch - Original Label Space
metric = "percent_agreement"
name = f"ai_rater_{metric}_ci"
mb_save_dir = analysis_dir / "2_compute_verifact_metrics"
pa_mb = MetricBunch.load(save_dir=mb_save_dir, name=name)

# Load MCC MetricBunch - Original Label Space
metric = "mcc"
name = f"ai_rater_{metric}_ci"
mb_save_dir = analysis_dir / "2_compute_verifact_metrics"
mcc_mb = MetricBunch.load(save_dir=mb_save_dir, name=name)

# Load PERCENT AGREEMENT MetricBunch - Binarized Label Space
metric = "percent_agreement"
name = f"ai_rater_{metric}_ci_binarized"
mb_save_dir = analysis_dir / "2_compute_verifact_metrics"
pa_mb_binarized = MetricBunch.load(save_dir=mb_save_dir, name=name)

# Load MCC MetricBunch - Binarized Label Space
metric = "mcc"
name = f"ai_rater_{metric}_ci_binarized"
mb_save_dir = analysis_dir / "2_compute_verifact_metrics"
mcc_mb_binarized = MetricBunch.load(save_dir=mb_save_dir, name=name)

# %% Best AI System by Percent Agreement - Original Label Space

## Set Author Type & Proposition Type
for mb_name, mb in zip(
    ["percent_agreement", "mcc", "percent_agreement_binarized", "mcc_binarized"],
    [pa_mb, mcc_mb, pa_mb_binarized, mcc_mb_binarized],
):
    metric_name_map = {
        "percent_agreement": "Percent Agreement",
        "mcc": "Matthew's Correlation Coefficient (MCC)",
        "percent_agreement_binarized": "Percent Agreement (Binarized)",
        "mcc_binarized": "Matthew's Correlation Coefficient (MCC) (Binarized)",
    }
    metric_name = metric_name_map[mb_name]

    for author_type, proposition_type in [
        ("llm", "claim"),
        ("llm", "sentence"),
        ("human", "claim"),
    ]:
        print(f"MetricBunch: {mb_name}")
        print(f"Author Type: {author_type}, Proposition Type: {proposition_type}")
        # Get Top Performing VeriFact AI System according to Percent Agreement
        best_ai = (
            mb.metrics.query(
                f"author_type == '{author_type}' & proposition_type == '{proposition_type}'"
            )
            .sort_values(by="value", ascending=False)
            .head(1)
            .squeeze()
        )
        # Get Verdicts
        best_ai_verdicts = ai_verdicts.query(
            f"model == '{best_ai.model}' & author_type == '{best_ai.author_type}' & "
            f"proposition_type == '{best_ai.proposition_type}' & "
            f"fact_type == '{best_ai.fact_type}' & "
            f"top_n == {best_ai.top_n} & retrieval_method == '{best_ai.retrieval_method}' & "
            f"reference_format == '{best_ai.reference_format}' & "
            f"reference_only_admission == {best_ai.reference_only_admission}"
        )
        y_pred = (
            best_ai_verdicts.loc[:, ["proposition_id", "verdict"]]
            .set_index("proposition_id")
            .sort_index()
        ).squeeze()
        human_gt_subset = human_gt.query(
            f"author_type == '{author_type}' & proposition_type == '{proposition_type}'"
        )
        y_true = (
            human_gt_subset.loc[:, ["proposition_id", "verdict"]]
            .set_index("proposition_id")
            .sort_index()
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
        metrics_table = cm.metrics_table(fmt=".2f", condense_label_and_metric=True)
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
            f"Author Type: {author_type}, Proposition Type: {proposition_type}\n"
            f"by {metric_name}",
            size=14,
        )
        ax.set_ylabel("Human Clinician\nGround Truth", size=12)
        ax.set_yticklabels(
            labels=["Supported", "Not\nSupported", "Not\nAddressed"], rotation=0, size=10
        )
        ax.set_xlabel("VeriFact AI System", size=12)
        ax.set_xticklabels(
            labels=["Supported", "Not\nSupported", "Not\nAddressed"], rotation=0, size=10
        )
        ax.tick_params(axis="both", which="both", bottom=False, left=False)
        fig.show()
        proposition_type_display = "atomic claim" if proposition_type == "claim" else "sentence"
        print(
            "Confusion Matrix comparing classification of propositions between "
            "Human Clinician Ground Truth (rows) and VeriFact AI System (columns)."
            f"The propositions being evaluated are derived from {author_type}-written text "
            f"with {proposition_type_display} propositions as the proposition type."
        )

        # Save Results to Disk
        name = f"best_{author_type}_{proposition_type}_{mb_name}"
        save_dir = analysis_dir / "5_best_model_classification_metrics" / name
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
