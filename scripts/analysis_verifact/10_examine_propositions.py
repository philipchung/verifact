# %% [markdown]
# ## Examine Propositions for Validity, Interrater Disagreement,
# ## and Verifact Disagreement with Human Ground Truth
# %%
import os
from pathlib import Path

import pandas as pd
from irr_metrics import coerce_types
from utils import load_environment, load_pandas, save_pandas

load_environment()
analysis_dir = Path(os.environ["PROJECT_DIR"]) / "scripts" / "analysis_verifact"
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
# Load Propositions
propositions = load_pandas(Path(os.environ["VERIFACTBHC_PROPOSITIONS_DIR"]) / "propositions.csv.gz")
# Load Proposition Validity Analysis Annotations
proposition_validity = load_pandas(
    Path(os.environ["VERIFACTBHC_PROPOSITIONS_DIR"]) / "proposition_validity.csv.gz"
)
# %%
# Create Dataframe for Examining Propositions
left_df = human_verdicts.loc[
    :,
    [
        "proposition_id",
        "text",
        "author_type",
        "proposition_type",
        "round1_num_raters_agree",
        "round1_majority_vote",
        "adjudicated_verdict",
        "adjudicated_comment",
        "human_gt",
    ],
]
right_df = proposition_validity.loc[
    :, ["proposition_id", "invalid", "imperative", "interrogative", "incomplete", "vague"]
]
df = pd.merge(
    left=left_df,
    right=right_df,
    on="proposition_id",
    how="left",
)


# %%
## Get Propositions Requiring Adjudication to Display as Illustrative Examples in a Table
adjudicated_comments = human_verdicts.loc[
    :, ["proposition_id", "text", "adjudicated_comment", "human_gt"]
].dropna(subset=["adjudicated_comment"])
save_pandas(
    df=adjudicated_comments,
    filepath=analysis_dir / "10_examine_propositions" / "adjudicated_comments.csv",
)

# %%
## Examine Proposition Validity
# Create contingency table with author_type x proposition_type on columns.
# Each validity feature as separate row with True/False sub-rows
validity_cols = ["invalid", "imperative", "interrogative", "incomplete", "vague"]
validity_df = (
    (
        pd.concat(
            [
                df.groupby(["author_type", "proposition_type"])[col]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
                for col in validity_cols
            ],
            axis=1,
            keys=validity_cols,
        )
        * 100
    )
    .round(1)
    .map("{:.1f}%".format)
    .T
)
# Save to CSV
save_pandas(
    df=validity_df,
    filepath=analysis_dir / "10_examine_propositions" / "proposition_validity_overview.csv",
    index=True,
)
# %%
## Examine Interrater Agreement for Invalid Propositions
# Create contingency table with round1_num_raters_agree as row stratification
# with author_type x proposition_type on columns.
# Each validity feature as separate row with True/False sub-rows

# All validity features, stratified by clinician agreement
# for each of author_type x proposition_type
validity_cols_all = ["invalid", "imperative", "interrogative", "incomplete", "vague"]
validity_agreement_all_df = (
    (
        pd.concat(
            [
                df.groupby(["author_type", "proposition_type", "round1_num_raters_agree"])[col]
                .value_counts(normalize=True)
                .unstack(fill_value=0)
                for col in validity_cols_all
            ],
            axis=1,
            keys=validity_cols_all,
        )
        * 100
    )
    .unstack()
    .round(1)
    .map("{:.1f}%".format)
    .T
)

# Save to CSV
save_pandas(
    df=validity_agreement_all_df,
    filepath=analysis_dir / "10_examine_propositions" / "proposition_validity_by_agreement.csv",
    index=True,
)
# %%
