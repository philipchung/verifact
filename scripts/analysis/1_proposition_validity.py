# %% [markdown]
# ## Proposition Validity Analysis
# %%
import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from utils import load_environment, load_pandas

load_environment()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

annotated_dataset_dir = Path(os.environ["ANNOTATED_DATASET_DIR"])
output_dir = Path.cwd() / "1_proposition_validity"
output_dir.mkdir(exist_ok=True, parents=True)

# Load Verdicts
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
# Load Proposition Validity
proposition_validity = load_pandas(annotated_dataset_dir / "proposition_validity.csv.gz").astype(
    {
        "proposition_id": "string",
        "text": "string",
        "author_type": "string",
        "proposition_type": "string",
        "invalid": "boolean",
        "imperative": "boolean",
        "interrogative": "boolean",
        "incomplete": "boolean",
        "vague": "boolean",
        "imperative_draft": "boolean",
        "interrogative_draft": "boolean",
        "incomplete_draft": "boolean",
        "vague_draft": "boolean",
    }
)
# %%
proposition_validity = proposition_validity.assign(
    input_text_type=proposition_validity.apply(
        lambda row: f"{row.author_type}_{row.proposition_type}", axis="columns"
    )
)


def create_count_percent_tables(
    df: pd.DataFrame, label_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a contingency table with count and percent of valid claims indicated by
    the column `label_name`."""
    count_tables = {}
    percent_tables = {}
    author_types = ["llm", "human"]
    proposition_types = ["claim", "sentence"]
    for author_type, proposition_type in itertools.product(author_types, proposition_types):
        name = f"{author_type}_{proposition_type.split()[0]}"
        # Get count contingency table for each subset
        subset_df = df.query(
            "author_type == @author_type and proposition_type == @proposition_type"
        )
        label_col = subset_df.loc[:, label_name]
        label_counts = label_col.value_counts(normalize=False).rename(name)
        label_percent = label_col.value_counts(normalize=True).rename(name)
        count_tables[name] = label_counts
        percent_tables[name] = label_percent

    # Accumulate Counts and Percentages
    valid_ct = pd.concat(list(count_tables.values()), axis=1).fillna(0.0)
    valid_percent = pd.concat(list(percent_tables.values()), axis=1).fillna(0.0)
    return valid_ct, valid_percent


# Generate Tables and Accumulate in Dictionary
count_dict, percent_dict = {}, {}
for label_name in ["invalid", "imperative", "interrogative", "incomplete", "vague"]:
    count_df, percent_df = create_count_percent_tables(proposition_validity, label_name)
    count_dict[label_name] = count_df
    percent_dict[label_name] = percent_df

# Combine Count Tables into single DataFrame
counts = (
    pd.concat(count_dict, axis="index", names=["Classification", "Label"])
    .rename(
        columns={
            "llm_claim": "LLM-written Text,\nAtomic Claim Proposition",
            "llm_sentence": "LLM-written Text,\nSentence Proposition",
            "human_claim": "Human-written Text,\nAtomic Claim Proposition",
            "human_sentence": "Human-written Text,\nSentence Proposition",
        }
    )
    .astype(int)
)
# Combine Percentage Tables into single DataFrame
percentages = (
    pd.concat(percent_dict, axis="index", names=["Classification", "Label"]).rename(
        columns={
            "llm_claim": "LLM-written Text,\nAtomic Claim Proposition",
            "llm_sentence": "LLM-written Text,\nSentence Proposition",
            "human_claim": "Human-written Text,\nAtomic Claim Proposition",
            "human_sentence": "Human-written Text,\nSentence Proposition",
        }
    )
    # .map(lambda x: f"{x:.1%}")
)

# Get Totals for Each Group
group_totals = counts.xs(key="invalid", level="Classification")

# Keep only True Counts
true_counts = counts.xs(True, level="Label", axis="index")
true_percents = percentages.xs(True, level="Label", axis="index")
# %%
# Create Grid and Subplot Axes
fig = plt.figure(layout="constrained", figsize=(9, 7))
gs = GridSpec(nrows=5, ncols=1, figure=fig, hspace=0.2)
p1 = fig.add_subplot(gs[0:2])
p2 = fig.add_subplot(gs[2:4])

# Plot colors
colors = sns.color_palette("tab10")
valid_color = colors[8]
invalid_color = colors[6]
bottom_colors = (colors[0], colors[1], colors[5], colors[4])

# Top Bar Plot
x = np.arange(group_totals.shape[1])  # the label locations
bar_width = 0.9
valid_counts = group_totals.loc[False].tolist()
valid_percents = true_percents.loc["invalid", :].apply(lambda x: 1 - x).tolist()
invalid_counts = group_totals.loc[True].tolist()
invalid_percents = true_percents.loc["invalid", :].tolist()
valid_labels = [f"Valid: {x} ({y:.1%})" for x, y in zip(valid_counts, valid_percents, strict=False)]
invalid_labels = [
    f"Invalid: {x} ({y:.1%})" for x, y in zip(invalid_counts, invalid_percents, strict=False)
]
# Invalid Bars
rect1 = p1.bar(
    x=x,
    height=invalid_counts,
    width=bar_width,
    label=invalid_labels,
    color=invalid_color,
    edgecolor="white",
)
p1.bar_label(
    rect1,
    labels=invalid_labels,
    padding=1,
    label_type="edge",
    color="mediumvioletred",
    fontweight="heavy",
)
rect2 = p1.bar(
    x=x,
    height=valid_counts,
    width=bar_width,
    bottom=invalid_counts,
    label=valid_labels,
    color=valid_color,
    edgecolor="white",
)
p1.bar_label(
    rect2,
    labels=valid_labels,
    padding=1,
    label_type="edge",
    color="forestgreen",
    fontweight="heavy",
)
# Remove X-tick, but Keep X-Labels
p1.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=True,  # keep X-Labels
)
p1.set_xticks(ticks=x, labels=group_totals.columns)
p1.set_ylabel("Proposition Counts")
p1.set_title("Proposition Validity by Author Type & Proposition Type", pad=20)
p1.spines["top"].set_visible(False)
p1.spines["right"].set_visible(False)

# Bottom Bar Plot
invalid_breakdown_data = true_counts.T.drop(columns="invalid")
invalid_breakdown_groups = invalid_breakdown_data.index
x = np.arange(len(invalid_breakdown_data.index))  # the label locations
bias = -0.4  # the x-position bias for the first bar
width = 0.2  # the width of the bars
multiplier = 1
rect_handles = []
for i, (attribute, measurement) in enumerate(invalid_breakdown_data.items()):
    offset = width * multiplier + bias
    rects = p2.bar(
        x + offset,
        measurement,
        width,
        label=attribute,
        color=bottom_colors[i],
        edgecolor="white",
    )
    p2.bar_label(
        rects,
        padding=1,
        label_type="edge",
        color=bottom_colors[i],
        fontweight="heavy",
    )
    rect_handles.append(rects)
    multiplier += 1
p2.set_xticks(ticks=x, labels=invalid_breakdown_groups)
p2.set_ylabel("Proposition Counts")
p2.legend(
    loc="upper left",
    handles=rect_handles,
    labels=[
        "Imperative Statement",
        "Interrogative Statement",
        "Incomplete Statement",
        "Vague Statement",
    ],
)
p2.spines["top"].set_visible(False)
p2.spines["right"].set_visible(False)
p2.set_title("Reasons for Invalid Propositions")

# Remove X-tick, but Keep X-Labels
p2.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=True,  # keep X-Labels
)

# Save Figure
fig.savefig(output_dir / "1_proposition_validity.png", dpi=300, bbox_inches="tight")
# %%
