# %% [markdown]
# ## Analyze VeriFact Metrics Stratified by Round 1 Annotator Agreement
# %%
import itertools
import os
import warnings
from pathlib import Path

import pandas as pd
from irr_metrics import MetricBunch
from utils import load_environment, save_pandas

load_environment()
analysis_dir = Path(os.environ["PROJECT_DIR"]) / "scripts" / "analysis_verifact"
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)


# Map of Model Short Name to Data Directory Name
model_dir_map = {
    # "Llama-8B": "verifact_llama3_1_8B",
    "Llama-70B": "verifact_llama3_1_70B",
    # "R1-8B": "verifact_deepseek_r1_distill_llama_8B",
    "R1-70B": "verifact_deepseek_r1_distill_llama_70B",
    # "Gemma3-12B": "verifact_gemma3_12B",
    "Gemma3-27B": "verifact_gemma3_27B",
    # "Qwen3-32B": "verifact_qwen3_32B",
    # "Qwen3-30B-A3B-Instruct": "verifact_qwen3_30B-A3B-Instruct",
    "Qwen3-30B-A3B-Thinking": "verifact_qwen3_30B-A3B-Thinking",
}
# Enumerate Unique Models
models = list(model_dir_map.keys())

# %%
# Load Data from Metric Bunches
agreement_strata = [
    "all_disagree",
    "majority_agree",
    "unanimous_agree",
    # "all",
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

metric_df: pd.DataFrame | None = None
for metric, agreement_stratum in (pbar := list(itertools.product(metrics, agreement_strata))):
    # Load Metric Bunch
    metric_name = metric.replace(" ", "_")
    name = f"strata_{agreement_stratum}-{metric_name}_ci"
    save_dir = (
        Path(os.environ["PROJECT_DIR"])
        / "scripts"
        / "analysis_verifact"
        / "11_compute_verifact_metrics_stratified"
    )
    mb = MetricBunch.load(save_dir=save_dir, name=name, load_data=True)
    if mb is None:
        warnings.warn(f"MetricBunch {name} not found in {save_dir}")
        continue
    else:
        _df = mb.metrics.copy().assign(agreement_stratum=agreement_stratum)
        if metric_df is None:
            metric_df = _df
        else:
            metric_df = pd.concat(
                [metric_df, _df],
                axis="index",
                ignore_index=False,
            )

# %%
# Create separate contingency tables for each metric
# Column = author_type x proposition_type x fact_type
# Rows = model x agreement_stratum
metric_names = metric_df["metric_name"].unique()

contingency_tables = {}
for metric_name in metric_names:
    # Filter data for this metric
    metric_subset = metric_df[metric_df["metric_name"] == metric_name]

    # Create pivot table with hierarchical columns and rows
    contingency_table = metric_subset.pivot_table(
        index=["model", "agreement_stratum"],
        columns=["author_type", "proposition_type", "fact_type"],
        values="display_str",
        aggfunc="first",  # Use first since display_str should be unique per combination
    )

    # Store the table
    contingency_tables[metric_name] = contingency_table

    # Save to CSV
    save_pandas(
        df=contingency_table,
        filepath=(
            analysis_dir
            / "12_analyze_verifact_agreement_stratified"
            / f"contingency_table_{metric_name}.csv"
        ),
        index=True,
    )
# %%
