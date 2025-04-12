# %% [markdown]
# ## Examine Human Clinician Inter-rater Agreement Analysis
#
# Percent Agreement & Gwet's AC1
# %%
import os
from pathlib import Path

import pandas as pd
from irr_metrics import (
    NOT_ADDRESSED,
    NOT_SUPPORTED,
    NOT_SUPPORTED_OR_ADDRESSED,
    SUPPORTED,
    MetricBunch,
    binarize_verdicts_in_df,
    coerce_types,
)
from utils import load_environment, load_pandas

load_environment()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

# Computed Interrater Agreement Results
irr_results_dir = Path.cwd() / "1_compute_interrater_agreement"

# Load Human Clinician Verdict Labels (One Row Per Proposition)
propositions_dir = Path(os.environ["VERIFACTBHC_PROPOSITIONS_DIR"])
human_verdicts = load_pandas(propositions_dir / "human_verdicts.csv.gz")
human_verdicts = coerce_types(human_verdicts)
# Isolate Human Ground Truth Labels
human_gt = (
    human_verdicts.assign(rater_name="human_gt")
    .astype({"rater_name": "string"})
    .rename(columns={"human_gt": "verdict"})
    .loc[:, ["proposition_id", "text", "author_type", "proposition_type", "rater_name", "verdict"]]
)

# %%
print("=== Counts with Original Label Space ===")
## Count for All Propositions
all_propositions_ct = human_verdicts.shape[0]
print(f"All Propositions Count: {all_propositions_ct}")

## Count for Propositions with at least one "Supported" Verdict
alo_supported_verdicts = human_verdicts.query(
    f"verdict1 == '{SUPPORTED}' or verdict2 == '{SUPPORTED}' or verdict3 == '{SUPPORTED}'"
)
alo_supported_ct = alo_supported_verdicts.shape[0]
print(f"Supported Propositions Count: {alo_supported_ct}")

## Count for Propositions with at least one "Not Supported" Verdict
alo_not_supported_verdicts = human_verdicts.query(
    f"verdict1 == '{NOT_SUPPORTED}' or "
    f"verdict2 == '{NOT_SUPPORTED}' or "
    f"verdict3 == '{NOT_SUPPORTED}'"
)
alo_not_supported_ct = alo_not_supported_verdicts.shape[0]
print(f"Not Supported Propositions Count: {alo_not_supported_ct}")

## Count for Propositions with at least one "Not Addressed" Verdict
alo_not_addressed_verdicts = human_verdicts.query(
    f"verdict1 == '{NOT_ADDRESSED}' or "
    f"verdict2 == '{NOT_ADDRESSED}' or "
    f"verdict3 == '{NOT_ADDRESSED}'"
)
alo_not_addressed_ct = alo_not_addressed_verdicts.shape[0]
print(f"Not Addressed Propositions Count: {alo_not_addressed_ct}")
# %%
print("=== Counts with Binarized Label Space ===")
## Binarize the Verdict Columns
binarized_human_verdicts = human_verdicts.copy()
for col in ["verdict1", "verdict2", "verdict3"]:
    binarized_human_verdicts = binarize_verdicts_in_df(binarized_human_verdicts, verdict_name=col)

## Count for All Propositions - Binarized
all_propositions_binarized_ct = binarized_human_verdicts.shape[0]
print(f"All Propositions Count (Binarized): {all_propositions_binarized_ct}")

## Count for Propositions with at least one "Supported" Verdict - Binarized
alo_supported_binarized_verdicts = binarized_human_verdicts.query(
    f"verdict1 == '{SUPPORTED}' or verdict2 == '{SUPPORTED}' or verdict3 == '{SUPPORTED}'"
)
alo_supported_binarized_ct = alo_supported_binarized_verdicts.shape[0]
print(f"Supported Propositions Count (Binarized): {alo_supported_binarized_ct}")

## Count for Propositions with at least one "Not Supported" Verdict - Binarized
alo_not_supported_binarized_verdicts = binarized_human_verdicts.query(
    f"verdict1 == '{NOT_SUPPORTED_OR_ADDRESSED}' or "
    f"verdict2 == '{NOT_SUPPORTED_OR_ADDRESSED}' or "
    f"verdict3 == '{NOT_SUPPORTED_OR_ADDRESSED}'"
)
alo_not_supported_binarized_ct = alo_not_supported_binarized_verdicts.shape[0]
print(f"Not Supported Propositions Count (Binarized): {alo_not_supported_binarized_ct}")

# %%
# Percent Agreement
all_pa_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_percent_agreement_ci_all"
)
supported_pa_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_percent_agreement_ci_supported"
)
not_supported_pa_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_percent_agreement_ci_not_supported"
)
not_addressed_pa_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_percent_agreement_ci_not_addressed"
)
all_binarized_pa_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_percent_agreement_ci_all_binarized"
)
supported_binarized_pa_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_percent_agreement_ci_supported_binarized"
)
not_supported_or_addressed_pa_mb = MetricBunch.load(
    save_dir=irr_results_dir,
    name="human_interrater_percent_agreement_ci_not_supported_or_addressed",
)
pa = pd.concat(
    [
        all_pa_mb.metrics.display_str.rename("All Propositions"),
        supported_pa_mb.metrics.display_str.rename("Supported Propositions"),
        not_supported_pa_mb.metrics.display_str.rename("Not Supported Propositions"),
        not_addressed_pa_mb.metrics.display_str.rename("Not Addressed Propositions"),
        all_binarized_pa_mb.metrics.display_str.rename("All Propositions (Binarized)"),
        supported_binarized_pa_mb.metrics.display_str.rename("Supported Propositions (Binarized)"),
        not_supported_or_addressed_pa_mb.metrics.display_str.rename(
            "Not Supported or Addressed Propositions (Binarized)"
        ),
    ],
    axis=1,
)


def extract_author_type_and_proposition_type(rater_name: str) -> tuple:
    author_type, proposition_type = rater_name.split(",")
    author_type = author_type.split("=")[1].strip()
    proposition_type = proposition_type.split("=")[1].strip()
    return author_type, proposition_type


author_type_and_proposition_type = pd.Series(pa.index).apply(
    extract_author_type_and_proposition_type
)
pa.index = pd.MultiIndex.from_tuples(
    author_type_and_proposition_type, names=["Author Type", "Proposition Type"]
)
pa.T

# %%
# Gwet's AC1
all_gwet_mb = MetricBunch.load(save_dir=irr_results_dir, name="human_interrater_gwet_ci_all")
supported_gwet_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_gwet_ci_supported"
)
not_supported_gwet_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_gwet_ci_not_supported"
)
not_addressed_gwet_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_gwet_ci_not_addressed"
)
all_binarized_gwet_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_gwet_ci_all_binarized"
)
supported_binarized_gwet_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_gwet_ci_supported_binarized"
)
not_supported_or_addressed_gwet_mb = MetricBunch.load(
    save_dir=irr_results_dir, name="human_interrater_gwet_ci_not_supported_or_addressed"
)
gwet = pd.concat(
    [
        all_gwet_mb.metrics.display_str.rename("All Propositions"),
        supported_gwet_mb.metrics.display_str.rename("Supported Propositions"),
        not_supported_gwet_mb.metrics.display_str.rename("Not Supported Propositions"),
        not_addressed_gwet_mb.metrics.display_str.rename("Not Addressed Propositions"),
        all_binarized_gwet_mb.metrics.display_str.rename("All Propositions (Binarized)"),
        supported_binarized_gwet_mb.metrics.display_str.rename(
            "Supported Propositions (Binarized)"
        ),
        not_supported_or_addressed_gwet_mb.metrics.display_str.rename(
            "Not Supported or Addressed Propositions (Binarized)"
        ),
    ],
    axis=1,
)

author_type_and_proposition_type = pd.Series(gwet.index).apply(
    extract_author_type_and_proposition_type
)
gwet.index = pd.MultiIndex.from_tuples(
    author_type_and_proposition_type, names=["Author Type", "Proposition Type"]
)
gwet.T

# %%
