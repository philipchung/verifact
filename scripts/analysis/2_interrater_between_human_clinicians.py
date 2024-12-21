# %% [markdown]
# ## Interrater Agreement Analysis Between Human Clinicians
# %%
import os
from pathlib import Path
from typing import Any

import pandas as pd
from irr_metrics import MetricBunch
from utils import load_environment, load_pandas

load_environment()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

annotated_dataset_dir = Path(os.environ["ANNOTATED_DATASET_DIR"])
output_dir = Path.cwd() / "2_interrater_between_human_clinicians"

# Groups: Cross Product of Author Type & Proposition Type
input_text_types = ("llm_claim", "llm_sentence", "human_claim", "human_sentence")

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

# Uncollate/Explode so that we have one row per rater per proposition


def uncollate_proposition(row: pd.Series) -> list[dict[str, Any]]:
    """Uncollate multiple verdicts from different raters assigned to a single proposition
    into separate rows so we have one row per rater rater than one row per proposition.

    Expects a row with the following columns:
    "proposition_id", "text", "author_type", "proposition_type",
    "rater1", "rater2", "rater3", "verdict1", "verdict2", "verdict3",
    "uncertain_flag1", "uncertain_flag2", "uncertain_flag3",
    "comment1", "comment2", "comment3",
    """
    annotations: list[dict[str, Any]] = []
    for i in (1, 2, 3):
        annotation = {
            "proposition_id": row.proposition_id,
            "text": row.text,
            "author_type": row.author_type,
            "proposition_type": row.proposition_type,
            "rater_name": row[f"rater{i}"],
            "verdict": row[f"verdict{i}"],
            "uncertain_flag": row[f"uncertain_flag{i}"],
            "comment": row[f"comment{i}"],
        }
        annotations.append(annotation)

    df = pd.DataFrame(annotations)
    return df


human_verdicts_uncollated = human_verdicts.apply(uncollate_proposition, axis="columns")
human_verdicts_uncollated = (
    pd.concat(human_verdicts_uncollated.tolist(), axis="index")
    .reset_index(drop=True)
    .astype(
        {
            "proposition_id": "string",
            "text": "string",
            "author_type": "string",
            "proposition_type": "string",
            "rater_name": "string",
            "verdict": "string",
            "uncertain_flag": "boolean",
            "comment": "string",
        }
    )
)
# %% [markdown]
# ### Inter-Clinican Agreement Analysis
#
# Analysis is Stratified by Proposition Type (claim, sentence) and Author Type (llm, human)
# and agreement metrics are computed for each stratum group.
#
# And we further stratify specific subsets of Propositions by identifying the set of propositions
# where at least one human clinician rater assigned "Supported", "Not Supported", or "Not Addressed"
# and compute agreement to quanitify variance associated with each of these labels.
# %%
bootstrap_iterations = 1000
workers = 72

## Create Metric Bunches for Each Input Text Type Group
# Percent Agreement - Across All Raters & All Propositions
interrater_pa_mb = MetricBunch.from_defaults(
    rater_verdicts=human_verdicts_uncollated,
    rater_type="human",
    rater_id_col="rater_name",
    strata=input_text_types,
    metric="percent_agreement",
    bootstrap_iterations=bootstrap_iterations,
    workers=workers,
)
interrater_pa_mb.save(output_dir / "pa")
# Gwet's AC1 - Across All Raters & All Propositions
interrater_gwet_mb = MetricBunch.from_defaults(
    rater_verdicts=human_verdicts_uncollated,
    rater_type="human",
    rater_id_col="rater_name",
    strata=input_text_types,
    metric="gwet",
    bootstrap_iterations=bootstrap_iterations,
    workers=workers,
)
interrater_gwet_mb.save(output_dir / "gwet")

## Identify Propositions where human clincians assigned at least one of each label
# "S", "NS", "NA". Only one of the 3 human labels needs to be assigned to the
# label value to be included, which means the resulting sets are not mutually exclusive.
## Propositions with at least one Supported Label
nodes_with_S = human_verdicts_uncollated.groupby("proposition_id")["verdict"].apply(
    lambda verdicts: "Supported" in verdicts.tolist()
)
nodes_with_S = nodes_with_S.loc[nodes_with_S].index.tolist()
S_verdicts = human_verdicts_uncollated.query("proposition_id in @nodes_with_S")
## Propositions with at least one Not Supported Label
nodes_with_NS = human_verdicts_uncollated.groupby("proposition_id")["verdict"].apply(
    lambda verdicts: "Not Supported" in verdicts.tolist()
)
nodes_with_NS = nodes_with_NS.loc[nodes_with_NS].index.tolist()
NS_verdicts = human_verdicts_uncollated.query("proposition_id in @nodes_with_NS")
## Propositions with at least one Not Addressed Label
nodes_with_NA = human_verdicts_uncollated.groupby("proposition_id")["verdict"].apply(
    lambda verdicts: "Not Addressed" in verdicts.tolist()
)
nodes_with_NA = nodes_with_NA.loc[nodes_with_NA].index.tolist()
NA_verdicts = human_verdicts_uncollated.query("proposition_id in @nodes_with_NA")

# Percent Agreement -  Across All Raters & Only Supported Propositions
interrater_pa_S_mb = MetricBunch.from_defaults(
    rater_verdicts=S_verdicts,
    rater_type="human",
    rater_id_col="rater_name",
    strata=input_text_types,
    metric="percent_agreement",
    bootstrap_iterations=bootstrap_iterations,
    workers=workers,
)
interrater_pa_S_mb.save(output_dir / "pa_S")
# Percent Agreement -  Across All Raters & Only Not Supported Propositions
interrater_pa_NS_mb = MetricBunch.from_defaults(
    rater_verdicts=NS_verdicts,
    rater_type="human",
    rater_id_col="rater_name",
    strata=input_text_types,
    metric="percent_agreement",
    bootstrap_iterations=bootstrap_iterations,
    workers=workers,
)
interrater_pa_NS_mb.save(output_dir / "pa_NS")
# Percent Agreement -  Across All Raters & Only Not Addressed Propositions
interrater_pa_NA_mb = MetricBunch.from_defaults(
    rater_verdicts=NA_verdicts,
    rater_type="human",
    rater_id_col="rater_name",
    strata=input_text_types,
    metric="percent_agreement",
    bootstrap_iterations=bootstrap_iterations,
    workers=workers,
)
interrater_pa_NA_mb.save(output_dir / "pa_NA")

# Gwet's AC1 - Across All Raters & Only Supported Propositions
interrater_gwet_S_mb = MetricBunch.from_defaults(
    rater_verdicts=S_verdicts,
    rater_type="human",
    rater_id_col="rater_name",
    strata=input_text_types,
    metric="gwet",
    bootstrap_iterations=bootstrap_iterations,
    workers=workers,
)
interrater_gwet_S_mb.save(output_dir / "gwet_S")
# Gwet's AC1 - Across All Raters & Only Not Supported Propositions
interrater_gwet_NS_mb = MetricBunch.from_defaults(
    rater_verdicts=NS_verdicts,
    rater_type="human",
    rater_id_col="rater_name",
    strata=input_text_types,
    metric="gwet",
    bootstrap_iterations=bootstrap_iterations,
    workers=workers,
)
interrater_gwet_NS_mb.save(output_dir / "gwet_NS")
# Gwet's AC1 - Across All Raters & Only Not Addressed Propositions
interrater_gwet_NA_mb = MetricBunch.from_defaults(
    rater_verdicts=NA_verdicts,
    rater_type="human",
    rater_id_col="rater_name",
    strata=input_text_types,
    metric="gwet",
    bootstrap_iterations=bootstrap_iterations,
    workers=workers,
)
interrater_gwet_NA_mb.save(output_dir / "gwet_NA")

# %%
