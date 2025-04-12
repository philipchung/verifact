# %% [markdown]
# Plot trend for agreement for increasing fact retrieval (Top N & Reference Context Length)
#
# For each experiment, we compute agreement for each
# VeriFact AI System Variation vs. Ground Truth Human Clinician Label
# %%
import os
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from irr_metrics import MetricBunch, coerce_categorical_types, coerce_types
from utils import load_environment, load_pandas

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

## Metrics in Original Label Space ("Supported", "Not Supported", "Not Addressed")
# Load MetricBunch with computed metrics from save_dir cache
metric = "percent_agreement"
name = f"ai_rater_{metric}_ci"
analysis_save_dir = Path.cwd() / "2_compute_verifact_agreement"
mb = MetricBunch.load(save_dir=analysis_save_dir, name=name)

## Add Reference Context Lengths to Metric Bunch
# Compute Mean Reference Context Lengths from Verdict Data's Reference Context
# for each Metric Bunch evaluation category
reference_contexts = dict(
    iter(
        ai_verdicts.groupby(
            [
                "author_type",
                "proposition_type",
                "fact_type",
                "model",
                "top_n",
                "retrieval_method",
                "reference_format",
                "reference_only_admission",
            ]
        )["reference"]
    )
)
word_lengths = {k: v.apply(lambda x: len(x.split())) for k, v in reference_contexts.items()}
mean_word_lengths = {k: v.mean() for k, v in word_lengths.items()}
char_lengths = {k: v.apply(len) for k, v in reference_contexts.items()}
mean_char_lengths = {k: v.mean() for k, v in char_lengths.items()}
# Format into DataFrame of Reference Context of Word & Character Lengths
mean_reference_context_lengths = pd.DataFrame(
    {"mean_word_length": mean_word_lengths, "mean_char_length": mean_char_lengths}
).reset_index(
    names=[
        "author_type",
        "proposition_type",
        "fact_type",
        "model",
        "top_n",
        "retrieval_method",
        "reference_format",
        "reference_only_admission",
    ]
)
# Join Mean Reference Context Lengths with Metric Bunch Data
mb.metrics = pd.merge(
    mb.metrics,
    mean_reference_context_lengths,
    on=[
        "author_type",
        "proposition_type",
        "fact_type",
        "model",
        "top_n",
        "retrieval_method",
        "reference_format",
        "reference_only_admission",
    ],
    how="left",
)

## Metrics in Binarized Label Space ("Supported", "Not Supported or Addressed")
# Load MetricBunch with computed metrics from save_dir cache
metric = "percent_agreement"
name = f"ai_rater_{metric}_ci_binarized"
analysis_save_dir = Path.cwd() / "2_compute_verifact_agreement"
mb_binarized = MetricBunch.load(save_dir=analysis_save_dir, name=name)

## Add Reference Context Lengths to Metric Bunch
# Compute Mean Reference Context Lengths from Verdict Data's Reference Context
# for each Metric Bunch evaluation category
mb_binarized.metrics = pd.merge(
    mb_binarized.metrics,
    mean_reference_context_lengths,
    on=[
        "author_type",
        "proposition_type",
        "fact_type",
        "model",
        "top_n",
        "retrieval_method",
        "reference_format",
        "reference_only_admission",
    ],
    how="left",
)

# %%
## LLM-written BHC w/ Claim Proposition - Original Labels
author_type = "llm"
proposition_type = "claim"
data = mb.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
data = coerce_categorical_types(data)

data_8B = data.query("model == 'Llama-8B' or model == 'R1-8B'")
data_8B = data_8B.assign(
    category=data_8B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

data_70B = data.query("model == 'Llama-70B' or model == 'R1-70B'")
data_70B = data_70B.assign(
    category=data_70B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

# Define a mapping of Top N values to specific markers
marker_map = {
    5: "^",
    10: "X",
    25: "s",
    50: "o",
    75: "v",
    100: "d",
    125: r"$\star$",
    150: r"$\clubsuit$",
}

## Make Subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), layout="constrained", sharey=True)

## First Plot: 8-Billion Parameter Models
category_order1 = ["Llama-8B: claim", "Llama-8B: sentence", "R1-8B: claim", "R1-8B: sentence"]
color_dict1 = dict(zip(category_order1, sns.color_palette("muted")))
# Create line plot
sns.lineplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order1,
    palette=color_dict1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_8B.query("fact_type == 'claim' & top_n == 150"),
        data_8B.query("fact_type == 'sentence' & top_n == 100"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)

## Second Plot: 70-Billion Parameter Models
category_order2 = ["Llama-70B: claim", "Llama-70B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict2 = dict(zip(category_order2, sns.color_palette("muted")))
# Create Line Plot
sns.lineplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order2,
    palette=color_dict2,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_70B.query("fact_type == 'claim' & top_n == 150"),
        data_70B.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict2[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)

# Set Gridlines
for p in [ax1, ax2]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2]:
    p.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    p.set_xlim(0, 4000)
    p.set_yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.set_ylim(0.6, 1.0)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))


# Set Titles & Labels
fig.suptitle(
    f"Llama 3 & Deepseek-R1-Distilled Models\n"
    f"Author Type: {author_type}, Proposition Type: {proposition_type}"
)
ax1.set_title("8-Billion Parameter Models")
ax2.set_title("70-Billion Parameter Models")
fig.supylabel("Percent Agreement", fontsize=12)
ax1.set_ylabel("")
ax2.set_ylabel("")
fig.supxlabel(
    "Mean Word Length of EHR Facts Reference Context Provided to LLM-as-a-Judge", fontsize=12
)
ax1.set_xlabel("")
ax2.set_xlabel("")

# Set Legends
ax1.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)
ax2.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)

# Create handles and labels for Top N markers
top_n_legend_handles = [
    mlines.Line2D(
        [],
        [],
        color="black",
        marker=marker,
        linestyle="None",
        markersize=6,
        label=f"N = {top_n}",
    )
    for top_n, marker in marker_map.items()
]
top_n_legend_labels = [f"N = {top_n}" for top_n in marker_map]
legend = fig.legend(
    title="Number of Facts Retrieved from EHR (N)",
    handles=[*top_n_legend_handles],
    labels=[*top_n_legend_labels],
    loc="center",
    bbox_to_anchor=(0.5, -0.07),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=9,
)

# Adjust Text Labels
adjust_text(
    texts1,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax1,
)
adjust_text(
    texts2,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax2,
)

# Save and Display Plot
save_dir = Path.cwd() / "4_top_n_vs_reference_context_length"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
# %%
## LLM-written BHC w/ Claim Proposition - Binarized Labels
author_type = "llm"
proposition_type = "claim"
data = mb_binarized.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
data = coerce_categorical_types(data)

data_8B = data.query("model == 'Llama-8B' or model == 'R1-8B'")
data_8B = data_8B.assign(
    category=data_8B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

data_70B = data.query("model == 'Llama-70B' or model == 'R1-70B'")
data_70B = data_70B.assign(
    category=data_70B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

# Define a mapping of Top N values to specific markers
marker_map = {
    5: "^",
    10: "X",
    25: "s",
    50: "o",
    75: "v",
    100: "d",
    125: r"$\star$",
    150: r"$\clubsuit$",
}

## Make Subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), layout="constrained", sharey=True)

## First Plot: 8-Billion Parameter Models
category_order1 = ["Llama-8B: claim", "Llama-8B: sentence", "R1-8B: claim", "R1-8B: sentence"]
color_dict1 = dict(zip(category_order1, sns.color_palette("muted")))
# Create line plot
sns.lineplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order1,
    palette=color_dict1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_8B.query("fact_type == 'claim' & top_n == 150"),
        data_8B.query("fact_type == 'sentence' & top_n == 100"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)

## Second Plot: 70-Billion Parameter Models
category_order2 = ["Llama-70B: claim", "Llama-70B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict2 = dict(zip(category_order2, sns.color_palette("muted")))
# Create Line Plot
sns.lineplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order2,
    palette=color_dict2,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_70B.query("fact_type == 'claim' & top_n == 150"),
        data_70B.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict2[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)

# Set Gridlines
for p in [ax1, ax2]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2]:
    p.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    p.set_xlim(0, 4000)
    p.set_yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.set_ylim(0.6, 1.0)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))


# Set Titles & Labels
fig.suptitle(
    f"Llama 3 & Deepseek-R1-Distilled Models\n"
    f"Author Type: {author_type}, Proposition Type: {proposition_type}"
)
ax1.set_title("8-Billion Parameter Models")
ax2.set_title("70-Billion Parameter Models")
fig.supylabel("Percent Agreement", fontsize=12)
ax1.set_ylabel("")
ax2.set_ylabel("")
fig.supxlabel(
    "Mean Word Length of EHR Facts Reference Context Provided to LLM-as-a-Judge", fontsize=12
)
ax1.set_xlabel("")
ax2.set_xlabel("")

# Set Legends
ax1.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)
ax2.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)

# Create handles and labels for Top N markers
top_n_legend_handles = [
    mlines.Line2D(
        [],
        [],
        color="black",
        marker=marker,
        linestyle="None",
        markersize=6,
        label=f"N = {top_n}",
    )
    for top_n, marker in marker_map.items()
]
top_n_legend_labels = [f"N = {top_n}" for top_n in marker_map]
legend = fig.legend(
    title="Number of Facts Retrieved from EHR (N)",
    handles=[*top_n_legend_handles],
    labels=[*top_n_legend_labels],
    loc="center",
    bbox_to_anchor=(0.5, -0.07),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=9,
)

# Adjust Text Labels
adjust_text(
    texts1,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax1,
)
adjust_text(
    texts2,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax2,
)

# Save and Display Plot
save_dir = Path.cwd() / "4_top_n_vs_reference_context_length"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-binarized.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
# %%
## LLM-written BHC w/ Sentence Proposition - Original Labels
author_type = "llm"
proposition_type = "sentence"
data = mb.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
data = coerce_categorical_types(data)

data_8B = data.query("model == 'Llama-8B' or model == 'R1-8B'")
data_8B = data_8B.assign(
    category=data_8B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

data_70B = data.query("model == 'Llama-70B' or model == 'R1-70B'")
data_70B = data_70B.assign(
    category=data_70B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

# Define a mapping of Top N values to specific markers
marker_map = {
    5: "^",
    10: "X",
    25: "s",
    50: "o",
    75: "v",
    100: "d",
    125: r"$\star$",
    150: r"$\clubsuit$",
}

## Make Subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), layout="constrained", sharey=True)

## First Plot: 8-Billion Parameter Models
category_order1 = ["Llama-8B: claim", "Llama-8B: sentence", "R1-8B: claim", "R1-8B: sentence"]
color_dict1 = dict(zip(category_order1, sns.color_palette("muted")))
# Create line plot
sns.lineplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order1,
    palette=color_dict1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_8B.query("fact_type == 'claim' & top_n == 150"),
        data_8B.query("fact_type == 'sentence' & top_n == 100"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)

## Second Plot: 70-Billion Parameter Models
category_order2 = ["Llama-70B: claim", "Llama-70B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict2 = dict(zip(category_order2, sns.color_palette("muted")))
# Create Line Plot
sns.lineplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order2,
    palette=color_dict2,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_70B.query("fact_type == 'claim' & top_n == 150"),
        data_70B.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict2[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)

# Set Gridlines
for p in [ax1, ax2]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2]:
    p.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    p.set_xlim(0, 4000)
    p.set_yticks([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.set_ylim(0.4, 1.0)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))


# Set Titles & Labels
fig.suptitle(
    f"Llama 3 & Deepseek-R1-Distilled Models\n"
    f"Author Type: {author_type}, Proposition Type: {proposition_type}"
)
ax1.set_title("8-Billion Parameter Models")
ax2.set_title("70-Billion Parameter Models")
fig.supylabel("Percent Agreement", fontsize=12)
ax1.set_ylabel("")
ax2.set_ylabel("")
fig.supxlabel(
    "Mean Word Length of EHR Facts Reference Context Provided to LLM-as-a-Judge", fontsize=12
)
ax1.set_xlabel("")
ax2.set_xlabel("")

# Set Legends
ax1.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)
ax2.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)

# Create handles and labels for Top N markers
top_n_legend_handles = [
    mlines.Line2D(
        [],
        [],
        color="black",
        marker=marker,
        linestyle="None",
        markersize=6,
        label=f"N = {top_n}",
    )
    for top_n, marker in marker_map.items()
]
top_n_legend_labels = [f"N = {top_n}" for top_n in marker_map]
legend = fig.legend(
    title="Number of Facts Retrieved from EHR (N)",
    handles=[*top_n_legend_handles],
    labels=[*top_n_legend_labels],
    loc="center",
    bbox_to_anchor=(0.5, -0.07),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=9,
)

# Adjust Text Labels
adjust_text(
    texts1,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax1,
)
adjust_text(
    texts2,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax2,
)

# Save and Display Plot
save_dir = Path.cwd() / "4_top_n_vs_reference_context_length"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
# %%
## LLM-written BHC w/ Sentence Proposition - Binarized Labels
author_type = "llm"
proposition_type = "sentence"
data = mb_binarized.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
data = coerce_categorical_types(data)

data_8B = data.query("model == 'Llama-8B' or model == 'R1-8B'")
data_8B = data_8B.assign(
    category=data_8B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

data_70B = data.query("model == 'Llama-70B' or model == 'R1-70B'")
data_70B = data_70B.assign(
    category=data_70B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

# Define a mapping of Top N values to specific markers
marker_map = {
    5: "^",
    10: "X",
    25: "s",
    50: "o",
    75: "v",
    100: "d",
    125: r"$\star$",
    150: r"$\clubsuit$",
}

## Make Subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), layout="constrained", sharey=True)

## First Plot: 8-Billion Parameter Models
category_order1 = ["Llama-8B: claim", "Llama-8B: sentence", "R1-8B: claim", "R1-8B: sentence"]
color_dict1 = dict(zip(category_order1, sns.color_palette("muted")))
# Create line plot
sns.lineplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order1,
    palette=color_dict1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_8B.query("fact_type == 'claim' & top_n == 150"),
        data_8B.query("fact_type == 'sentence' & top_n == 100"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)

## Second Plot: 70-Billion Parameter Models
category_order2 = ["Llama-70B: claim", "Llama-70B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict2 = dict(zip(category_order2, sns.color_palette("muted")))
# Create Line Plot
sns.lineplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order2,
    palette=color_dict2,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_70B.query("fact_type == 'claim' & top_n == 150"),
        data_70B.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict2[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)

# Set Gridlines
for p in [ax1, ax2]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2]:
    p.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    p.set_xlim(0, 4000)
    p.set_yticks([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.set_ylim(0.4, 1.0)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))


# Set Titles & Labels
fig.suptitle(
    f"Llama 3 & Deepseek-R1-Distilled Models\n"
    f"Author Type: {author_type}, Proposition Type: {proposition_type}"
)
ax1.set_title("8-Billion Parameter Models")
ax2.set_title("70-Billion Parameter Models")
fig.supylabel("Percent Agreement", fontsize=12)
ax1.set_ylabel("")
ax2.set_ylabel("")
fig.supxlabel(
    "Mean Word Length of EHR Facts Reference Context Provided to LLM-as-a-Judge", fontsize=12
)
ax1.set_xlabel("")
ax2.set_xlabel("")

# Set Legends
ax1.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)
ax2.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)

# Create handles and labels for Top N markers
top_n_legend_handles = [
    mlines.Line2D(
        [],
        [],
        color="black",
        marker=marker,
        linestyle="None",
        markersize=6,
        label=f"N = {top_n}",
    )
    for top_n, marker in marker_map.items()
]
top_n_legend_labels = [f"N = {top_n}" for top_n in marker_map]
legend = fig.legend(
    title="Number of Facts Retrieved from EHR (N)",
    handles=[*top_n_legend_handles],
    labels=[*top_n_legend_labels],
    loc="center",
    bbox_to_anchor=(0.5, -0.07),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=9,
)

# Adjust Text Labels
adjust_text(
    texts1,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax1,
)
adjust_text(
    texts2,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax2,
)

# Save and Display Plot
save_dir = Path.cwd() / "4_top_n_vs_reference_context_length"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-binarized.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()

# %%
## Human-written BHC w/ Claim Proposition - Original Labels
author_type = "human"
proposition_type = "claim"
data = mb.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
data = coerce_categorical_types(data)

data_8B = data.query("model == 'Llama-8B' or model == 'R1-8B'")
data_8B = data_8B.assign(
    category=data_8B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

data_70B = data.query("model == 'Llama-70B' or model == 'R1-70B'")
data_70B = data_70B.assign(
    category=data_70B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

# Define a mapping of Top N values to specific markers
marker_map = {
    5: "^",
    10: "X",
    25: "s",
    50: "o",
    75: "v",
    100: "d",
    125: r"$\star$",
    150: r"$\clubsuit$",
}

## Make Subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), layout="constrained", sharey=True)

## First Plot: 8-Billion Parameter Models
category_order1 = ["Llama-8B: claim", "Llama-8B: sentence", "R1-8B: claim", "R1-8B: sentence"]
color_dict1 = dict(zip(category_order1, sns.color_palette("muted")))
# Create line plot
sns.lineplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order1,
    palette=color_dict1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_8B.query("fact_type == 'claim' & top_n == 150"),
        data_8B.query("fact_type == 'sentence' & top_n == 100"),
        data_8B.query("fact_type == 'sentence' & top_n == 75 & model == 'R1-8B'"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)

## Second Plot: 70-Billion Parameter Models
category_order2 = ["Llama-70B: claim", "Llama-70B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict2 = dict(zip(category_order2, sns.color_palette("muted")))
# Create Line Plot
sns.lineplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order2,
    palette=color_dict2,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_70B.query("fact_type == 'claim' & top_n == 150"),
        data_70B.query("fact_type == 'sentence' & top_n == 100"),
        data_70B.query("fact_type == 'sentence' & top_n == 75 & model == 'R1-70B'"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict2[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)

# Set Gridlines
for p in [ax1, ax2]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2]:
    p.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    p.set_xlim(0, 4000)
    p.set_yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    p.set_ylim(0.45, 0.8)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))


# Set Titles & Labels
fig.suptitle(
    f"Llama 3 & Deepseek-R1-Distilled Models\n"
    f"Author Type: {author_type}, Proposition Type: {proposition_type}"
)
ax1.set_title("8-Billion Parameter Models")
ax2.set_title("70-Billion Parameter Models")
fig.supylabel("Percent Agreement", fontsize=12)
ax1.set_ylabel("")
ax2.set_ylabel("")
fig.supxlabel(
    "Mean Word Length of EHR Facts Reference Context Provided to LLM-as-a-Judge", fontsize=12
)
ax1.set_xlabel("")
ax2.set_xlabel("")

# Set Legends
ax1.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)
ax2.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)

# Create handles and labels for Top N markers
top_n_legend_handles = [
    mlines.Line2D(
        [],
        [],
        color="black",
        marker=marker,
        linestyle="None",
        markersize=6,
        label=f"N = {top_n}",
    )
    for top_n, marker in marker_map.items()
]
top_n_legend_labels = [f"N = {top_n}" for top_n in marker_map]
legend = fig.legend(
    title="Number of Facts Retrieved from EHR (N)",
    handles=[*top_n_legend_handles],
    labels=[*top_n_legend_labels],
    loc="center",
    bbox_to_anchor=(0.5, -0.07),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=9,
)

# Adjust Text Labels
adjust_text(
    texts1,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax1,
)
adjust_text(
    texts2,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax2,
)

# Save and Display Plot
save_dir = Path.cwd() / "4_top_n_vs_reference_context_length"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()

# %%
## Human-written BHC w/ Claim Proposition - Binarized Labels
author_type = "human"
proposition_type = "claim"
data = mb_binarized.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
data = coerce_categorical_types(data)

data_8B = data.query("model == 'Llama-8B' or model == 'R1-8B'")
data_8B = data_8B.assign(
    category=data_8B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

data_70B = data.query("model == 'Llama-70B' or model == 'R1-70B'")
data_70B = data_70B.assign(
    category=data_70B.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)

# Define a mapping of Top N values to specific markers
marker_map = {
    5: "^",
    10: "X",
    25: "s",
    50: "o",
    75: "v",
    100: "d",
    125: r"$\star$",
    150: r"$\clubsuit$",
}

## Make Subplots
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 5), layout="constrained", sharey=True)

## First Plot: 8-Billion Parameter Models
category_order1 = ["Llama-8B: claim", "Llama-8B: sentence", "R1-8B: claim", "R1-8B: sentence"]
color_dict1 = dict(zip(category_order1, sns.color_palette("muted")))
# Create line plot
sns.lineplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order1,
    palette=color_dict1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_8B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_8B.query("fact_type == 'claim' & top_n == 150"),
        data_8B.query("fact_type == 'sentence' & top_n == 100"),
        data_8B.query("fact_type == 'sentence' & top_n == 75 & model == 'R1-8B'"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)

## Second Plot: 70-Billion Parameter Models
category_order2 = ["Llama-70B: claim", "Llama-70B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict2 = dict(zip(category_order2, sns.color_palette("muted")))
# Create Line Plot
sns.lineplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order2,
    palette=color_dict2,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_70B,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_70B.query("fact_type == 'claim' & top_n == 150"),
        data_70B.query("fact_type == 'sentence' & top_n == 100"),
        data_70B.query("fact_type == 'sentence' & top_n == 75 & model == 'R1-70B'"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict2[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)

# Set Gridlines
for p in [ax1, ax2]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2]:
    p.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000])
    p.set_xlim(0, 4000)
    p.set_yticks([0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    p.set_ylim(0.55, 0.9)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))


# Set Titles & Labels
fig.suptitle(
    f"Llama 3 & Deepseek-R1-Distilled Models\n"
    f"Author Type: {author_type}, Proposition Type: {proposition_type}"
)
ax1.set_title("8-Billion Parameter Models")
ax2.set_title("70-Billion Parameter Models")
fig.supylabel("Percent Agreement", fontsize=12)
ax1.set_ylabel("")
ax2.set_ylabel("")
fig.supxlabel(
    "Mean Word Length of EHR Facts Reference Context Provided to LLM-as-a-Judge", fontsize=12
)
ax1.set_xlabel("")
ax2.set_xlabel("")

# Set Legends
ax1.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)
ax2.legend(title="Model & EHR Fact Type", loc="lower right", fontsize=9)

# Create handles and labels for Top N markers
top_n_legend_handles = [
    mlines.Line2D(
        [],
        [],
        color="black",
        marker=marker,
        linestyle="None",
        markersize=6,
        label=f"N = {top_n}",
    )
    for top_n, marker in marker_map.items()
]
top_n_legend_labels = [f"N = {top_n}" for top_n in marker_map]
legend = fig.legend(
    title="Number of Facts Retrieved from EHR (N)",
    handles=[*top_n_legend_handles],
    labels=[*top_n_legend_labels],
    loc="center",
    bbox_to_anchor=(0.5, -0.07),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=9,
)

# Adjust Text Labels
adjust_text(
    texts1,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax1,
)
adjust_text(
    texts2,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2, 2),
    force_text=(2, 2),
    force_static=(3, 3),
    force_explode=(3, 3),
    ax=ax2,
)

# Save and Display Plot
save_dir = Path.cwd() / "4_top_n_vs_reference_context_length"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-binarized.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
# %%
