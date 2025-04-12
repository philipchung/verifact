# %% [markdown]
# ## Final Metric Result Tables
#
# Metric is either "Percent Agreement" or "Gwet's AC1" with 95% Confidence Intervals
# %%
from pathlib import Path

import pandas as pd
from irr_metrics import MetricBunch
from utils import load_environment
from utils.file_utils import save_pandas

load_environment()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

models_to_include = ["Llama-8B", "Llama-70B", "R1-8B", "R1-70B"]
# %%
# VeriFact Percent Agreement
metric = "percent_agreement"
name = f"ai_rater_{metric}_ci"
analysis_save_dir = Path.cwd() / "2_compute_verifact_agreement"
mb = MetricBunch.load(save_dir=analysis_save_dir, name=name)

df = mb.metrics.query(f"model in {models_to_include}").reset_index(drop=True)
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
        "model": "LLM-as-a-Judge",
        "top_n": "Top N",
        "retrieval_method": "Retrieval Method",
        "reference_format": "Reference Context Format",
        "reference_only_admission": "Reference Only Admission",
        "display_str": "Percent Agreement (95% CI)",
    }
)
save_dir = Path.cwd() / "6_metric_result_tables"
save_pandas(df=df, filepath=save_dir / "percent_agreement.csv")

# %%
# VeriFact Gwet's AC1
metric = "gwet"
name = f"ai_rater_{metric}_ci"
analysis_save_dir = Path.cwd() / "2_compute_verifact_agreement"
mb = MetricBunch.load(save_dir=analysis_save_dir, name=name)

df = mb.metrics.query(f"model in {models_to_include}").reset_index(drop=True)
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
        "model": "LLM-as-a-Judge",
        "top_n": "Top N",
        "retrieval_method": "Retrieval Method",
        "reference_format": "Reference Context Format",
        "reference_only_admission": "Reference Only Admission",
        "display_str": "Gwet's AC1 (95% CI)",
    }
)
save_dir = Path.cwd() / "6_metric_result_tables"
save_pandas(df=df, filepath=save_dir / "gwet.csv")
# %%
# VeriFact Percent Agreement - Binarized
metric = "percent_agreement"
name = f"ai_rater_{metric}_ci_binarized"
analysis_save_dir = Path.cwd() / "2_compute_verifact_agreement"
mb = MetricBunch.load(save_dir=analysis_save_dir, name=name)

df = mb.metrics.query(f"model in {models_to_include}").reset_index(drop=True)
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
        "model": "LLM-as-a-Judge",
        "top_n": "Top N",
        "retrieval_method": "Retrieval Method",
        "reference_format": "Reference Context Format",
        "reference_only_admission": "Reference Only Admission",
        "display_str": "Percent Agreement (95% CI)",
    }
)
save_dir = Path.cwd() / "6_metric_result_tables"
save_pandas(df=df, filepath=save_dir / "percent_agreement_binarized.csv")
# %%
# VeriFact Gwet's AC1
metric = "gwet"
name = f"ai_rater_{metric}_ci_binarized"
analysis_save_dir = Path.cwd() / "2_compute_verifact_agreement"
mb = MetricBunch.load(save_dir=analysis_save_dir, name=name)

df = mb.metrics.query(f"model in {models_to_include}").reset_index(drop=True)
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
        "model": "LLM-as-a-Judge",
        "top_n": "Top N",
        "retrieval_method": "Retrieval Method",
        "reference_format": "Reference Context Format",
        "reference_only_admission": "Reference Only Admission",
        "display_str": "Gwet's AC1 (95% CI)",
    }
)
save_dir = Path.cwd() / "6_metric_result_tables"
save_pandas(df=df, filepath=save_dir / "gwet_binarized.csv")
# %%
