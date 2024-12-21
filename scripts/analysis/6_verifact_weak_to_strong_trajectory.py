# %% [markdown]
# ## Weak->Strong VeriFact System: Label Assignment Shifts
#
# The goal is to examine the changes in AI system decisions as we go from
# Weak to Strong VeriFact AI System, and to see how these decisions compare to the
# human ground truth labels.
#
# Since Top N (number of retrieved facts) is the strongest hyperparameter affecting
# performance, we will examine the change in AI system rating decisions as we sweep this
# hyperparameter but keep the others fixed.
# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
from utils import load_environment, load_pandas

load_environment()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)


annotated_dataset_dir = Path(os.environ["ANNOTATED_DATASET_DIR"])
mb_dir = Path.cwd() / "3_verifact_vs_ground_truth_tables"
output_dir = Path.cwd() / "6_verifact_weak_to_strong_trajectory"
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

# %%
label_alias_map = {"Supported": "S", "Not Supported": "NS", "Not Addressed": "NA"}
verdict_dtype = CategoricalDtype(categories=["S", "NS", "NA"], ordered=False)

ai_verdicts_subset = ai_verdicts.query(
    "retrieval_method == 'dense' & "
    "reference_format == 'score' & "
    "reference_only_admission == False"
).set_index("proposition_id")
verdicts = {}
for top_n in (5, 10, 25, 50):
    verdicts[f"Judge{top_n}"] = ai_verdicts_subset.query(f"top_n == {top_n}").verdict

# Store Verdicts of Weak->Strong VeriFact AI System in DataFrame; Remap Labels to Aliases & Categorical DType
gt_labels = human_gt.set_index("proposition_id").verdict.map(label_alias_map).astype(verdict_dtype)
verdicts = pd.DataFrame(verdicts).map(lambda x: label_alias_map[x]).astype(verdict_dtype)


# Determine whether Verdicts are Same or Different from the one before as we go from
# Weak->Strong VeriFact AI System.
# First Unique Label is always A, Second Unique Label is B, Third Unique Label is C
def map_verdict_same_or_different_from_previous(row) -> str:
    i = 0
    new_labels = ["A", "B", "C"]
    mapping = {}
    output = {}
    for judge_name, verdict in row.items():
        # If verdict has not yet been seen, assign a new mapping identity
        if verdict not in mapping:
            mapping[verdict] = new_labels[i]
            i += 1
        # Use Mapping to Assign New Label
        output[judge_name] = mapping[verdict]
    return pd.Series(output, index=row.index)


verdicts_diff = verdicts.apply(map_verdict_same_or_different_from_previous, axis="columns")

# Whether Judge50 Agrees or Disagrees with Ground Truth Labels
final_verdicts_agree = verdicts.loc[:, "Judge50"].eq(gt_labels)

# Right Arrow Unicode
right_arrow = "\u2192"  # Right Common Arrow


def format_trajectory(row) -> str:
    return right_arrow.join(row)


def format_trajectory_accent_last(row) -> str:
    return right_arrow.join(row.iloc[:-1]) + f"{right_arrow}[{row.iloc[-1]}]"


# All Unique Trajectories (Original Label Space)
trajectories = verdicts_diff.sort_values(by=["Judge5", "Judge10", "Judge25", "Judge50"]).apply(
    lambda row: format_trajectory_accent_last(row), axis="columns"
)
agree_trajectories = trajectories.loc[final_verdicts_agree]
disagree_trajectories = trajectories.loc[~final_verdicts_agree]

# Counts
trajectory_cts = (
    pd.DataFrame(
        {
            "All": trajectories.value_counts(dropna=False),
            "TopN=50 Agree with GT": agree_trajectories.value_counts(dropna=False),
            "TopN=50 Disagree with GT": disagree_trajectories.value_counts(dropna=False),
        }
    )
    .fillna(0)
    .astype(int)
    .rename_axis(index=f"TopN=5{right_arrow}10{right_arrow}25{right_arrow}50")
)
# Percents
trajectory_pct = (
    pd.DataFrame(
        {
            "All": trajectories.value_counts(normalize=True, dropna=False),
            "Top N=50 Agree with GT": agree_trajectories.value_counts(normalize=True, dropna=False),
            "Top N=50 Disagree with GT": disagree_trajectories.value_counts(
                normalize=True, dropna=False
            ),
        }
    )
    .fillna(0.0)
    .rename_axis(index="TopN=5{right_arrow}10{right_arrow}25{right_arrow}50")
)
# %%
# Drop Uncommon Trajectories
drop_threshold = 0

# Set Data For Plotting
data = (
    trajectory_cts.sort_index(ascending=False).query(f"All >= {drop_threshold}").drop(columns="All")
)
data.columns = pd.CategoricalIndex(
    data.columns, categories=["TopN=50 Disagree with GT", "TopN=50 Agree with GT"], ordered=True
)
data = data.sort_index(axis="columns")
# Create Figure
colors = ["darkorange", "steelblue"]
fig, ax = plt.subplots(figsize=(6, 7), layout="constrained")
ax = data.plot(kind="barh", width=0.8, color=colors, edgecolor="white", ax=ax)

# Add Bar Labels
for i, containers in enumerate(ax.containers):
    ax.bar_label(
        containers,
        label_type="edge",
        padding=2,
        color=colors[i],
        fontsize=9,
        fontweight="bold",
    )


# Format Title, Axes
ax.set_title(f"Weak{right_arrow}Strong VeriFact System: Label Assignment Shifts")
ax.set_xlabel("Proposition Counts")
ax.set_ylabel(
    f"Number of Facts Retrieved" f"\nTop N: 5{right_arrow}10{right_arrow}25{right_arrow}[50]"
)
# Set X-Limits
ax.set_xlim(left=0, right=7500)
# Turn on Grid with Major only on Y-axis & Major+Minor on X-axis
ax.grid(axis="x", which="minor", alpha=0.2)
ax.grid(axis="both", which="major", alpha=0.5)
ax.minorticks_on()
ax.yaxis.set_tick_params(which="minor", bottom=False)
ax.set_axisbelow(True)
# set Legend
handles, labels = ax.get_legend_handles_labels()
order = [1, 0]
ax.legend(
    handles=[handles[idx] for idx in order],
    labels=[
        "VeriFact System with Top N=50 \nAgrees with Ground Truth",
        "VeriFact System with Top N=50 \nDisagrees with Ground Truth",
    ],
    loc="best",
)
fig.savefig(
    output_dir / "proposition_trajectories_topn.png",
    dpi=300,
    bbox_inches="tight",
)
# %%
