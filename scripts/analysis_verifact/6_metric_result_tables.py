# %% [markdown]
# ## Final Metric Result Tables
#
# Metric is either "Percent Agreement" or "Gwet's AC1" with 95% Confidence Intervals
# %%
import os
from pathlib import Path

import pandas as pd
from irr_metrics import MetricBunch
from tqdm.auto import tqdm
from utils import load_environment
from utils.file_utils import save_pandas

load_environment()
analysis_dir = Path(os.environ["PROJECT_DIR"]) / "scripts" / "analysis_verifact"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

models_to_include = [
    "Gemma3-12B",
    "Gemma3-27B",
    "Qwen3-30B-A3B-Instruct",
    "Qwen3-30B-A3B-Thinking",
    "Qwen3-32B",
    "R1-8B",
    "R1-70B",
    "Llama-8B",
    "Llama-70B",
]
# %%
# Iteratively Add Each Metric to Final DataFrame
final_df: pd.DataFrame | None = None
for metric in (
    pbar := tqdm(
        [
            "percent_agreement",
            "gwet",
            "mcc",
            "s-tpr",
            "s-tnr",
            "s-ppv",
            "s-npv",
            "ns-tpr",
            "ns-tnr",
            "ns-ppv",
            "ns-npv",
            "na-tpr",
            "na-tnr",
            "na-ppv",
            "na-npv",
        ]
    )
):
    # Metric Name Map
    metric_name_map = {
        "percent_agreement": "Percent Agreement",
        "gwet": "Gwet's AC1",
        "mcc": "MCC",
        "s-tpr": "Supported TPR",
        "s-tnr": "Supported TNR",
        "s-ppv": "Supported PPV",
        "s-npv": "Supported NPV",
        "ns-tpr": "Not Supported TPR",
        "ns-tnr": "Not Supported TNR",
        "ns-ppv": "Not Supported PPV",
        "ns-npv": "Not Supported NPV",
        "na-tpr": "Not Addressed TPR",
        "na-tnr": "Not Addressed TNR",
        "na-ppv": "Not Addressed PPV",
        "na-npv": "Not Addressed NPV",
    }
    metric_display_name = metric_name_map[metric]
    # Update Progress Bar
    pbar.set_description(desc=f"Processing metric: {metric}")
    # Load MetricBunch from Disk
    name = f"ai_rater_{metric}_ci"
    mb_save_dir = analysis_dir / "2_compute_verifact_metrics"
    mb = MetricBunch.load(save_dir=mb_save_dir, name=name)
    df = mb.metrics.query(f"model in {models_to_include}").reset_index(drop=True)

    # Keep Relevant Columns and Rename for Clarity
    df = df[
        [
            "author_type",
            "proposition_type",
            "fact_type",
            "model",
            "top_n",
            "retrieval_method",
            "reference_format",
            "reference_only_admission",
            "display_str",
        ]
    ].rename(
        columns={
            "author_type": "Author Type",
            "proposition_type": "Proposition Type",
            "fact_type": "Fact Type",
            "model": "LLM Judge",
            "top_n": "Top N",
            "retrieval_method": "Retrieval Method",
            "reference_format": "Reference Context Format",
            "reference_only_admission": "Reference Only Admission",
            "display_str": f"{metric_display_name} (95% CI)",
        }
    )
    # Merge with Final DataFrame
    if final_df is None:
        final_df = df
    else:
        final_df = pd.merge(
            final_df,
            df,
            how="inner",
            on=[
                "Author Type",
                "Proposition Type",
                "Fact Type",
                "LLM Judge",
                "Top N",
                "Retrieval Method",
                "Reference Context Format",
                "Reference Only Admission",
            ],
            suffixes=("", f" {metric}"),
        )

# Save Final DataFrame to Disk
save_dir = analysis_dir / "6_metric_result_tables"
save_dir.mkdir(exist_ok=True, parents=True)
save_pandas(df=final_df, filepath=save_dir / "all_metrics.csv")

# Create a version with 95% CI separated by new-line token instead of space
final_df_newline = final_df.copy()
for col in final_df_newline.columns:
    if "95% CI" in col:
        final_df_newline[col] = final_df_newline[col].apply(
            lambda x: x.replace(" ", "\n") if isinstance(x, str) else x
        )
save_pandas(df=final_df_newline, filepath=save_dir / "all_metrics-newline.csv")

# Create filtered version with:
# Retrieval Method = "rerank"
# Reference Context Format = "absolute time"
# Reference Only Admission = True
# these represent the "best" settings from sensitivity analysis
# and are settings used in the main figures.
final_df_newline_filtered = final_df_newline.query(
    "`Retrieval Method` == 'rerank' and "
    "`Reference Context Format` == 'absolute time' and "
    "`Reference Only Admission` == True"
).drop(columns=["Retrieval Method", "Reference Context Format", "Reference Only Admission"])
save_pandas(
    df=final_df_newline_filtered,
    filepath=save_dir / "all_metrics-newline-filtered.csv",
)
# %%
# Iteratively Add Each Metric to Final DataFrame - Binarized
final_df_binarized: pd.DataFrame | None = None
for metric in (
    pbar := tqdm(
        [
            "percent_agreement",
            "gwet",
            "mcc",
            "s-tpr",
            "s-tnr",
            "s-ppv",
            "s-npv",
            "ns-tpr",
            "ns-tnr",
            "ns-ppv",
            "ns-npv",
            "na-tpr",
            "na-tnr",
            "na-ppv",
            "na-npv",
        ]
    )
):
    # Metric Name Map
    metric_name_map = {
        "percent_agreement": "Percent Agreement",
        "gwet": "Gwet's AC1",
        "mcc": "MCC",
        "s-tpr": "Supported TPR",
        "s-tnr": "Supported TNR",
        "s-ppv": "Supported PPV",
        "s-npv": "Supported NPV",
        "ns-tpr": "Not Supported TPR",
        "ns-tnr": "Not Supported TNR",
        "ns-ppv": "Not Supported PPV",
        "ns-npv": "Not Supported NPV",
        "na-tpr": "Not Addressed TPR",
        "na-tnr": "Not Addressed TNR",
        "na-ppv": "Not Addressed PPV",
        "na-npv": "Not Addressed NPV",
    }
    metric_display_name = metric_name_map[metric]
    # Update Progress Bar
    pbar.set_description(desc=f"Processing metric: {metric}")
    # Load MetricBunch from Disk (Binarized)
    name = f"ai_rater_{metric}_ci_binarized"
    mb_save_dir = analysis_dir / "2_compute_verifact_metrics"
    mb_binarized = MetricBunch.load(save_dir=mb_save_dir, name=name)
    df = mb_binarized.metrics.query(f"model in {models_to_include}").reset_index(drop=True)

    # Keep Relevant Columns and Rename for Clarity
    df = df[
        [
            "author_type",
            "proposition_type",
            "fact_type",
            "model",
            "top_n",
            "retrieval_method",
            "reference_format",
            "reference_only_admission",
            "display_str",
        ]
    ].rename(
        columns={
            "author_type": "Author Type",
            "proposition_type": "Proposition Type",
            "fact_type": "Fact Type",
            "model": "LLM Judge",
            "top_n": "Top N",
            "retrieval_method": "Retrieval Method",
            "reference_format": "Reference Context Format",
            "reference_only_admission": "Reference Only Admission",
            "display_str": f"{metric_display_name} (95% CI)",
        }
    )
    # Merge with Final DataFrame
    if final_df_binarized is None:
        final_df_binarized = df
    else:
        final_df_binarized = pd.merge(
            final_df_binarized,
            df,
            how="inner",
            on=[
                "Author Type",
                "Proposition Type",
                "Fact Type",
                "LLM Judge",
                "Top N",
                "Retrieval Method",
                "Reference Context Format",
                "Reference Only Admission",
            ],
            suffixes=("", f" {metric}"),
        )

# Save Final DataFrame to Disk
save_dir = analysis_dir / "6_metric_result_tables"
save_dir.mkdir(exist_ok=True, parents=True)
save_pandas(df=final_df_binarized, filepath=save_dir / "all_metrics-binarized.csv")

# Create a version with 95% CI separated by new-line token instead of space
final_df_binarized_newline = final_df_binarized.copy()
for col in final_df_binarized_newline.columns:
    if "95% CI" in col:
        final_df_binarized_newline[col] = final_df_binarized_newline[col].apply(
            lambda x: x.replace(" ", "\n") if isinstance(x, str) else x
        )
save_pandas(df=final_df_binarized_newline, filepath=save_dir / "all_metrics-binarized-newline.csv")

# Create filtered version with:
# Retrieval Method = "rerank"
# Reference Context Format = "absolute time"
# Reference Only Admission = True
# these represent the "best" settings from sensitivity analysis
# and are settings used in the main figures.
final_df_binarized_newline_filtered = final_df_binarized_newline.query(
    "`Retrieval Method` == 'rerank' and "
    "`Reference Context Format` == 'absolute time' and "
    "`Reference Only Admission` == True"
).drop(columns=["Retrieval Method", "Reference Context Format", "Reference Only Admission"])
save_pandas(
    df=final_df_binarized_newline_filtered,
    filepath=save_dir / "all_metrics-binarized-newline-filtered.csv",
)


# %%
