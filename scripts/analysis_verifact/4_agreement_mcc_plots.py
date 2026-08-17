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
from irr_metrics import MetricBunch, coerce_categorical_types
from matplotlib.patches import Rectangle
from release_data import load_reference_lengths, load_release_annotations
from utils import load_environment

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
ai_verdicts = load_release_annotations(release_dir, models=models)

## Get Mean Reference Context Lengths per VeriFact hyperparameter setting
# Compute Mean Reference Context Lengths without loading reference text
# for each Metric Bunch evaluation category
reference_lengths = load_reference_lengths(
    release_dir, reference_ids=ai_verdicts["reference_id"].unique()
)
ai_verdicts = ai_verdicts.merge(
    reference_lengths,
    on="reference_id",
    how="left",
    validate="many_to_one",
)
mean_reference_context_lengths = (
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
        ],
        observed=True,
    )
    .agg(
        mean_word_length=("reference_word_count", "mean"),
        mean_char_length=("reference_char_count", "mean"),
    )
    .reset_index()
)

### Load Data - Original Labels
## PERCENT AGREEMENT
# Load MetricBunch with computed metrics from save_dir cache
metric = "percent_agreement"
name = f"ai_rater_{metric}_ci"
mb_save_dir = analysis_dir / "2_compute_verifact_metrics"
pa_mb = MetricBunch.load(save_dir=mb_save_dir, name=name)

# Join Mean Reference Context Lengths with Metric Bunch Data
pa_mb.metrics = pd.merge(
    pa_mb.metrics,
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

## MATTHEWS CORRELATION COEFFICIENT
# Load MetricBunch with computed metrics from save_dir cache
metric = "mcc"
name = f"ai_rater_{metric}_ci"
mb_save_dir = analysis_dir / "2_compute_verifact_metrics"
mcc_mb = MetricBunch.load(save_dir=mb_save_dir, name=name)


# Join Mean Reference Context Lengths with Metric Bunch Data
mcc_mb.metrics = pd.merge(
    mcc_mb.metrics,
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
### Load Data - Binarized Labels
## PERCENT AGREEMENT
# Load MetricBunch with computed metrics from save_dir cache
metric = "percent_agreement"
name = f"ai_rater_{metric}_ci_binarized"
mb_save_dir = analysis_dir / "2_compute_verifact_metrics"
pa_mb_binarized = MetricBunch.load(save_dir=mb_save_dir, name=name)

# Join Mean Reference Context Lengths with Metric Bunch Data
pa_mb_binarized.metrics = pd.merge(
    pa_mb_binarized.metrics,
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

## MATTHEWS CORRELATION COEFFICIENT
# Load MetricBunch with computed metrics from save_dir cache
metric = "mcc"
name = f"ai_rater_{metric}_ci_binarized"
mb_save_dir = analysis_dir / "2_compute_verifact_metrics"
mcc_mb_binarized = MetricBunch.load(save_dir=mb_save_dir, name=name)


# Join Mean Reference Context Lengths with Metric Bunch Data
mcc_mb_binarized.metrics = pd.merge(
    mcc_mb_binarized.metrics,
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


# %% [markdown]
# ## Original Labels (Supported, Not Supported, Not Addressed)
# %% [markdown]
# ## Percent Agreement & MCC vs. Reference Context Length, Original Labels
# %% Author Type = LLM, Proposition Type = Claim, Original Labels
author_type = "llm"
proposition_type = "claim"

# Subset Percent Agreement Metric Bunch for Author Type & Proposition Type
pa_data = pa_mb.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
pa_data = coerce_categorical_types(pa_data)
# Subset MCC Metric Bunch for Author Type & Proposition Type
mcc_data = mcc_mb.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
mcc_data = coerce_categorical_types(mcc_data)

## Make Subplots
fig, [(ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)] = plt.subplots(
    nrows=2, ncols=4, figsize=(11, 8), layout="constrained"
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

### Gemma3
category_order_gemma = [
    "Gemma3-12B: claim",
    "Gemma3-12B: sentence",
    "Gemma3-27B: claim",
    "Gemma3-27B: sentence",
]
color_dict_gemma = dict(zip(category_order_gemma, sns.color_palette("Dark2")))

## Plot 1. Percent Agreement: Gemma
data_gemma = pa_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)


## Plot 5. MCC: Gemma3
data_gemma = mcc_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax5,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax5,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts5 = []
for i in range(len(x)):
    text = ax5.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts5.append(text)


### Qwen3
category_order_qwen = [
    "Qwen3-32B: claim",
    "Qwen3-32B: sentence",
    "Qwen3-30B-A3B-Thinking: claim",
    "Qwen3-30B-A3B-Thinking: sentence",
    "Qwen3-30B-A3B-Instruct: claim",
    "Qwen3-30B-A3B-Instruct: sentence",
]
color_dict_qwen = dict(zip(category_order_qwen, sns.color_palette("Dark2")))
## 2. Percent Agreement: Qwen3
data_qwen = pa_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)


## 6. MCC: Qwen3
data_qwen = mcc_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax6,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax6,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts6 = []
for i in range(len(x)):
    text = ax6.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts6.append(text)


### DeepSeek-R1-Distill-Llama
category_order_r1 = ["R1-8B: claim", "R1-8B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict_r1 = dict(zip(category_order_r1, sns.color_palette("Dark2")))
## 3. Percent Agreement: DeepSeek-R1-Distill-Llama
data_r1 = pa_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax3,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax3,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts3 = []
for i in range(len(x)):
    text = ax3.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts3.append(text)


## 7. MCC: R1-Distilled
data_r1 = mcc_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax7,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax7,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts7 = []
for i in range(len(x)):
    text = ax7.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts7.append(text)


### Llama3
category_order_llama = [
    "Llama-8B: claim",
    "Llama-8B: sentence",
    "Llama-70B: claim",
    "Llama-70B: sentence",
]
color_dict_llama = dict(zip(category_order_llama, sns.color_palette("Dark2")))
## 4. Percent Agreement: Llama3
data_llama = pa_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax4,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax4,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts4 = []
for i in range(len(x)):
    text = ax4.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts4.append(text)


## 8. MCC: Llama3
data_llama = mcc_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax8,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax8,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts8 = []
for i in range(len(x)):
    text = ax8.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts8.append(text)

# Set Gridlines
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    p.set_xlim(0, 4500)

for p in [ax1, ax2, ax3, ax4]:
    p.set_yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.set_ylim(0.60, 1.0)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

for p in [ax5, ax6, ax7, ax8]:
    p.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    p.set_ylim(0.0, 0.4)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f} "))

# Make more compact by removing ticks & labels for plots other than first in each row
for p in [ax2, ax3, ax4, ax6, ax7, ax8]:
    p.set_yticklabels([])
    p.tick_params(axis="y", length=0)

# Set Titles & Labels
author_type_str = (
    "LLM-generated Brief Hospital Course"
    if author_type == "llm"
    else "Human-written Brief Hospital Course"
)
proposition_type_str = "Atomic Claim" if proposition_type == "claim" else "Sentence"
fig.suptitle(
    f"{author_type_str} Evaluated Using {proposition_type_str} Propositions",
    fontsize=14,
)
ax1.set_title("Gemma-3", fontsize=10)
ax2.set_title("Qwen-3", fontsize=10)
ax3.set_title("Deepseek-R1-Distill-Llama", fontsize=10)
ax4.set_title("Llama-3.1", fontsize=10)
ax1.set_ylabel("Percent Agreement", fontsize=12)
ax2.set_ylabel("")
ax3.set_ylabel("")
ax4.set_ylabel("")
ax5.set_ylabel("Matthews Correlation Coefficient (MCC)", fontsize=12)
ax6.set_ylabel("")
ax7.set_ylabel("")
ax8.set_ylabel("")

fig.supxlabel("Mean Word Length of EHR Facts Reference Context Provided to LLM Judge", fontsize=12)
ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("")
ax4.set_xlabel("")
ax5.set_xlabel("")
ax6.set_xlabel("")
ax7.set_xlabel("")
ax8.set_xlabel("")

### Set Legends
# Custom legend for ax1 with rectangle patches and custom labels
custom_labels_gemma = {
    "Gemma3-12B: claim": "Gemma-3-12B (Claim Fact)",
    "Gemma3-12B: sentence": "Gemma-3-12B (Sentence Fact)",
    "Gemma3-27B: claim": "Gemma-3-27B (Claim Fact)",
    "Gemma3-27B: sentence": "Gemma-3-27B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax1 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_gemma[cat], linewidth=0) for cat in category_order_gemma
]
custom_labels_ax1 = [custom_labels_gemma[cat] for cat in category_order_gemma]
# Set custom legend for ax1
ax1.legend(
    handles=custom_handles_ax1,
    labels=custom_labels_ax1,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax2 with rectangle patches and custom labels
custom_labels_qwen = {
    "Qwen3-32B: claim": "Qwen-3-32B (Claim Fact)",
    "Qwen3-32B: sentence": "Qwen-3-32B (Sentence Fact)",
    "Qwen3-30B-A3B-Thinking: claim": "Qwen-3-30B-A3B-Thinking (Claim Fact)",
    "Qwen3-30B-A3B-Thinking: sentence": "Qwen-3-30B-A3B-Thinking (Sentence Fact)",
    "Qwen3-30B-A3B-Instruct: claim": "Qwen-3-30B-A3B-Instruct (Claim Fact)",
    "Qwen3-30B-A3B-Instruct: sentence": "Qwen-3-30B-A3B-Instruct (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax2 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_qwen[cat], linewidth=0) for cat in category_order_qwen
]
custom_labels_ax2 = [custom_labels_qwen[cat] for cat in category_order_qwen]
# Set custom legend for ax2
ax2.legend(
    handles=custom_handles_ax2,
    labels=custom_labels_ax2,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax3 with rectangle patches and custom labels
custom_labels_r1 = {
    "R1-8B: claim": "R1-Distill-Llama-8B (Claim Fact)",
    "R1-8B: sentence": "R1-Distill-Llama-8B (Sentence Fact)",
    "R1-70B: claim": "R1-Distill-Llama-70B (Claim Fact)",
    "R1-70B: sentence": "R1-Distill-Llama-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax3 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_r1[cat], linewidth=0) for cat in category_order_r1
]
custom_labels_ax3 = [custom_labels_r1[cat] for cat in category_order_r1]
# Set custom legend for ax3
ax3.legend(
    handles=custom_handles_ax3,
    labels=custom_labels_ax3,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax4 with rectangle patches and custom labels
custom_labels_llama = {
    "Llama-8B: claim": "Llama-3.1-8B (Claim Fact)",
    "Llama-8B: sentence": "Llama-3.1-8B (Sentence Fact)",
    "Llama-70B: claim": "Llama-3.1-70B (Claim Fact)",
    "Llama-70B: sentence": "Llama-3.1-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax4 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_llama[cat], linewidth=0) for cat in category_order_llama
]
custom_labels_ax4 = [custom_labels_llama[cat] for cat in category_order_llama]
# Set custom legend for ax4
ax4.legend(
    handles=custom_handles_ax4,
    labels=custom_labels_ax4,
    loc="lower right",
    fontsize=7,
)
# Remove legends in ax5-ax8 because they are duplicates of ax1-ax4
ax5.legend().remove()
ax6.legend().remove()
ax7.legend().remove()
ax8.legend().remove()

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
    bbox_to_anchor=(0.5, -0.05),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=10,
)

# Adjust Text (per adjustText package, should be called last in figure generation)
for text, ax in zip(
    [texts1, texts2, texts3, texts4, texts5, texts6, texts7, texts8],
    [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8],
):
    adjust_text(
        text,
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.3),
        expand=(2, 2),
        force_text=(2, 2),
        force_static=(3, 3),
        force_explode=(3, 3),
        ax=ax,
    )


# Save and Display Plot
save_dir = analysis_dir / "4_agreement_mcc_plots"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}.svg",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()

# %% Author Type = LLM, Proposition Type = Sentence, Original Labels
author_type = "llm"
proposition_type = "sentence"

# Subset Percent Agreement Metric Bunch for Author Type & Proposition Type
pa_data = pa_mb.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
pa_data = coerce_categorical_types(pa_data)
# Subset MCC Metric Bunch for Author Type & Proposition Type
mcc_data = mcc_mb.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
mcc_data = coerce_categorical_types(mcc_data)

## Make Subplots
fig, [(ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)] = plt.subplots(
    nrows=2, ncols=4, figsize=(11, 8), layout="constrained"
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

### Gemma3
category_order_gemma = [
    "Gemma3-12B: claim",
    "Gemma3-12B: sentence",
    "Gemma3-27B: claim",
    "Gemma3-27B: sentence",
]
color_dict_gemma = dict(zip(category_order_gemma, sns.color_palette("Dark2")))

## Plot 1. Percent Agreement: Gemma
data_gemma = pa_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)


## Plot 5. MCC: Gemma3
data_gemma = mcc_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax5,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax5,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts5 = []
for i in range(len(x)):
    text = ax5.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts5.append(text)


### Qwen3
category_order_qwen = [
    "Qwen3-32B: claim",
    "Qwen3-32B: sentence",
    "Qwen3-30B-A3B-Thinking: claim",
    "Qwen3-30B-A3B-Thinking: sentence",
    "Qwen3-30B-A3B-Instruct: claim",
    "Qwen3-30B-A3B-Instruct: sentence",
]
color_dict_qwen = dict(zip(category_order_qwen, sns.color_palette("Dark2")))
## 2. Percent Agreement: Qwen3
data_qwen = pa_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)


## 6. MCC: Qwen3
data_qwen = mcc_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax6,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax6,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts6 = []
for i in range(len(x)):
    text = ax6.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts6.append(text)


### Deepseek-R1-Distill-Llama
category_order_r1 = ["R1-8B: claim", "R1-8B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict_r1 = dict(zip(category_order_r1, sns.color_palette("Dark2")))
## 3. Percent Agreement: Deepseek-R1-Distill-Llama
data_r1 = pa_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax3,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax3,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts3 = []
for i in range(len(x)):
    text = ax3.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts3.append(text)


## 7. MCC: R1-Distilled
data_r1 = mcc_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax7,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax7,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts7 = []
for i in range(len(x)):
    text = ax7.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts7.append(text)


### Llama3
category_order_llama = [
    "Llama-8B: claim",
    "Llama-8B: sentence",
    "Llama-70B: claim",
    "Llama-70B: sentence",
]
color_dict_llama = dict(zip(category_order_llama, sns.color_palette("Dark2")))
## 4. Percent Agreement: Llama3
data_llama = pa_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax4,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax4,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts4 = []
for i in range(len(x)):
    text = ax4.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts4.append(text)


## 8. MCC: Llama3
data_llama = mcc_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax8,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax8,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts8 = []
for i in range(len(x)):
    text = ax8.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts8.append(text)

# Set Gridlines
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    p.set_xlim(0, 4500)

for p in [ax1, ax2, ax3, ax4]:
    p.set_yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.set_ylim(0.60, 1.0)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

for p in [ax5, ax6, ax7, ax8]:
    p.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    p.set_ylim(-0.05, 0.3)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f} "))

# Make more compact by removing ticks & labels for plots other than first in each row
for p in [ax2, ax3, ax4, ax6, ax7, ax8]:
    p.set_yticklabels([])
    p.tick_params(axis="y", length=0)

# Set Titles & Labels
author_type_str = (
    "LLM-generated Brief Hospital Course"
    if author_type == "llm"
    else "Human-written Brief Hospital Course"
)
proposition_type_str = "Atomic Claim" if proposition_type == "claim" else "Sentence"
fig.suptitle(
    f"{author_type_str} Evaluated Using {proposition_type_str} Propositions",
    fontsize=14,
)
ax1.set_title("Gemma-3", fontsize=10)
ax2.set_title("Qwen-3", fontsize=10)
ax3.set_title("Deepseek-R1-Distill-Llama", fontsize=10)
ax4.set_title("Llama-3.1", fontsize=10)
ax1.set_ylabel("Percent Agreement", fontsize=12)
ax2.set_ylabel("")
ax3.set_ylabel("")
ax4.set_ylabel("")
ax5.set_ylabel("Matthews Correlation Coefficient (MCC)", fontsize=12)
ax6.set_ylabel("")
ax7.set_ylabel("")
ax8.set_ylabel("")

fig.supxlabel("Mean Word Length of EHR Facts Reference Context Provided to LLM Judge", fontsize=12)
ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("")
ax4.set_xlabel("")
ax5.set_xlabel("")
ax6.set_xlabel("")
ax7.set_xlabel("")
ax8.set_xlabel("")

### Set Legends
# Custom legend for ax1 with rectangle patches and custom labels
custom_labels_gemma = {
    "Gemma3-12B: claim": "Gemma-3-12B (Claim Fact)",
    "Gemma3-12B: sentence": "Gemma-3-12B (Sentence Fact)",
    "Gemma3-27B: claim": "Gemma-3-27B (Claim Fact)",
    "Gemma3-27B: sentence": "Gemma-3-27B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax1 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_gemma[cat], linewidth=0) for cat in category_order_gemma
]
custom_labels_ax1 = [custom_labels_gemma[cat] for cat in category_order_gemma]
# Set custom legend for ax1
ax1.legend(
    handles=custom_handles_ax1,
    labels=custom_labels_ax1,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax2 with rectangle patches and custom labels
custom_labels_qwen = {
    "Qwen3-32B: claim": "Qwen-3-32B (Claim Fact)",
    "Qwen3-32B: sentence": "Qwen-3-32B (Sentence Fact)",
    "Qwen3-30B-A3B-Thinking: claim": "Qwen-3-30B-A3B-Thinking (Claim Fact)",
    "Qwen3-30B-A3B-Thinking: sentence": "Qwen-3-30B-A3B-Thinking (Sentence Fact)",
    "Qwen3-30B-A3B-Instruct: claim": "Qwen-3-30B-A3B-Instruct (Claim Fact)",
    "Qwen3-30B-A3B-Instruct: sentence": "Qwen-3-30B-A3B-Instruct (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax2 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_qwen[cat], linewidth=0) for cat in category_order_qwen
]
custom_labels_ax2 = [custom_labels_qwen[cat] for cat in category_order_qwen]
# Set custom legend for ax2
ax2.legend(
    handles=custom_handles_ax2,
    labels=custom_labels_ax2,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax3 with rectangle patches and custom labels
custom_labels_r1 = {
    "R1-8B: claim": "R1-Distill-Llama-8B (Claim Fact)",
    "R1-8B: sentence": "R1-Distill-Llama-8B (Sentence Fact)",
    "R1-70B: claim": "R1-Distill-Llama-70B (Claim Fact)",
    "R1-70B: sentence": "R1-Distill-Llama-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax3 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_r1[cat], linewidth=0) for cat in category_order_r1
]
custom_labels_ax3 = [custom_labels_r1[cat] for cat in category_order_r1]
# Set custom legend for ax3
ax3.legend(
    handles=custom_handles_ax3,
    labels=custom_labels_ax3,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax4 with rectangle patches and custom labels
custom_labels_llama = {
    "Llama-8B: claim": "Llama-3.1-8B (Claim Fact)",
    "Llama-8B: sentence": "Llama-3.1-8B (Sentence Fact)",
    "Llama-70B: claim": "Llama-3.1-70B (Claim Fact)",
    "Llama-70B: sentence": "Llama-3.1-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax4 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_llama[cat], linewidth=0) for cat in category_order_llama
]
custom_labels_ax4 = [custom_labels_llama[cat] for cat in category_order_llama]
# Set custom legend for ax4
ax4.legend(
    handles=custom_handles_ax4,
    labels=custom_labels_ax4,
    loc="lower right",
    fontsize=7,
)
# Remove legends in ax5-ax8 because they are duplicates of ax1-ax4
ax5.legend().remove()
ax6.legend().remove()
ax7.legend().remove()
ax8.legend().remove()

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
    bbox_to_anchor=(0.5, -0.05),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=10,
)

# Adjust Text (per adjustText package, should be called last in figure generation)
for text, ax in zip(
    [texts1, texts2, texts3, texts4, texts5, texts6, texts7, texts8],
    [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8],
):
    adjust_text(
        text,
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.3),
        expand=(2, 2),
        force_text=(2, 2),
        force_static=(3, 3),
        force_explode=(4, 4) if ax == ax7 else (3, 3),
        ax=ax,
    )


# Save and Display Plot
save_dir = analysis_dir / "4_agreement_mcc_plots"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}.svg",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
# %% Author Type = Human, Proposition Type = Claim, Original Labels
author_type = "human"
proposition_type = "claim"

# Subset Percent Agreement Metric Bunch for Author Type & Proposition Type
pa_data = pa_mb.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
pa_data = coerce_categorical_types(pa_data)
# Subset MCC Metric Bunch for Author Type & Proposition Type
mcc_data = mcc_mb.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
mcc_data = coerce_categorical_types(mcc_data)

## Make Subplots
fig, [(ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)] = plt.subplots(
    nrows=2, ncols=4, figsize=(11, 8), layout="constrained"
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

### Gemma3
category_order_gemma = [
    "Gemma3-12B: claim",
    "Gemma3-12B: sentence",
    "Gemma3-27B: claim",
    "Gemma3-27B: sentence",
]
color_dict_gemma = dict(zip(category_order_gemma, sns.color_palette("Dark2")))

## Plot 1. Percent Agreement: Gemma
data_gemma = pa_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)


## Plot 5. MCC: Gemma3
data_gemma = mcc_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax5,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax5,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts5 = []
for i in range(len(x)):
    text = ax5.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts5.append(text)


### Qwen3
category_order_qwen = [
    "Qwen3-32B: claim",
    "Qwen3-32B: sentence",
    "Qwen3-30B-A3B-Thinking: claim",
    "Qwen3-30B-A3B-Thinking: sentence",
    "Qwen3-30B-A3B-Instruct: claim",
    "Qwen3-30B-A3B-Instruct: sentence",
]
color_dict_qwen = dict(zip(category_order_qwen, sns.color_palette("Dark2")))
## 2. Percent Agreement: Qwen3
data_qwen = pa_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)


## 6. MCC: Qwen3
data_qwen = mcc_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax6,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax6,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts6 = []
for i in range(len(x)):
    text = ax6.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts6.append(text)


### Deepseek-R1-Distill-Llama
category_order_r1 = ["R1-8B: claim", "R1-8B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict_r1 = dict(zip(category_order_r1, sns.color_palette("Dark2")))
## 3. Percent Agreement: R1-Distilled
data_r1 = pa_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax3,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax3,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts3 = []
for i in range(len(x)):
    text = ax3.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts3.append(text)


## 7. MCC: R1-Distilled
data_r1 = mcc_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax7,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax7,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts7 = []
for i in range(len(x)):
    text = ax7.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts7.append(text)


### Llama3
category_order_llama = [
    "Llama-8B: claim",
    "Llama-8B: sentence",
    "Llama-70B: claim",
    "Llama-70B: sentence",
]
color_dict_llama = dict(zip(category_order_llama, sns.color_palette("Dark2")))
## 4. Percent Agreement: Llama3
data_llama = pa_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax4,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax4,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts4 = []
for i in range(len(x)):
    text = ax4.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts4.append(text)


## 8. MCC: Llama3
data_llama = mcc_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax8,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax8,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts8 = []
for i in range(len(x)):
    text = ax8.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts8.append(text)

# Set Gridlines
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    p.set_xlim(0, 4500)

for p in [ax1, ax2, ax3, ax4]:
    p.set_yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.set_ylim(0.45, 0.8)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

for p in [ax5, ax6, ax7, ax8]:
    p.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    p.set_ylim(0.2, 0.6)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f} "))

# Make more compact by removing ticks & labels for plots other than first in each row
for p in [ax2, ax3, ax4, ax6, ax7, ax8]:
    p.set_yticklabels([])
    p.tick_params(axis="y", length=0)

# Set Titles & Labels
author_type_str = (
    "LLM-generated Brief Hospital Course"
    if author_type == "llm"
    else "Human-written Brief Hospital Course"
)
proposition_type_str = "Atomic Claim" if proposition_type == "claim" else "Sentence"
fig.suptitle(
    f"{author_type_str} Evaluated Using {proposition_type_str} Propositions",
    fontsize=14,
)
ax1.set_title("Gemma-3", fontsize=10)
ax2.set_title("Qwen-3", fontsize=10)
ax3.set_title("Deepseek-R1-Distill-Llama", fontsize=10)
ax4.set_title("Llama-3.1", fontsize=10)
ax1.set_ylabel("Percent Agreement", fontsize=12)
ax2.set_ylabel("")
ax3.set_ylabel("")
ax4.set_ylabel("")
ax5.set_ylabel("Matthews Correlation Coefficient (MCC)", fontsize=12)
ax6.set_ylabel("")
ax7.set_ylabel("")
ax8.set_ylabel("")

fig.supxlabel("Mean Word Length of EHR Facts Reference Context Provided to LLM Judge", fontsize=12)
ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("")
ax4.set_xlabel("")
ax5.set_xlabel("")
ax6.set_xlabel("")
ax7.set_xlabel("")
ax8.set_xlabel("")

### Set Legends
# Custom legend for ax1 with rectangle patches and custom labels
custom_labels_gemma = {
    "Gemma3-12B: claim": "Gemma-3-12B (Claim Fact)",
    "Gemma3-12B: sentence": "Gemma-3-12B (Sentence Fact)",
    "Gemma3-27B: claim": "Gemma-3-27B (Claim Fact)",
    "Gemma3-27B: sentence": "Gemma-3-27B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax1 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_gemma[cat], linewidth=0) for cat in category_order_gemma
]
custom_labels_ax1 = [custom_labels_gemma[cat] for cat in category_order_gemma]
# Set custom legend for ax1
ax1.legend(
    handles=custom_handles_ax1,
    labels=custom_labels_ax1,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax2 with rectangle patches and custom labels
custom_labels_qwen = {
    "Qwen3-32B: claim": "Qwen-3-32B (Claim Fact)",
    "Qwen3-32B: sentence": "Qwen-3-32B (Sentence Fact)",
    "Qwen3-30B-A3B-Thinking: claim": "Qwen-3-30B-A3B-Thinking (Claim Fact)",
    "Qwen3-30B-A3B-Thinking: sentence": "Qwen-3-30B-A3B-Thinking (Sentence Fact)",
    "Qwen3-30B-A3B-Instruct: claim": "Qwen-3-30B-A3B-Instruct (Claim Fact)",
    "Qwen3-30B-A3B-Instruct: sentence": "Qwen-3-30B-A3B-Instruct (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax2 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_qwen[cat], linewidth=0) for cat in category_order_qwen
]
custom_labels_ax2 = [custom_labels_qwen[cat] for cat in category_order_qwen]
# Set custom legend for ax2
ax2.legend(
    handles=custom_handles_ax2,
    labels=custom_labels_ax2,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax3 with rectangle patches and custom labels
custom_labels_r1 = {
    "R1-8B: claim": "R1-Distill-Llama-8B (Claim Fact)",
    "R1-8B: sentence": "R1-Distill-Llama-8B (Sentence Fact)",
    "R1-70B: claim": "R1-Distill-Llama-70B (Claim Fact)",
    "R1-70B: sentence": "R1-Distill-Llama-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax3 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_r1[cat], linewidth=0) for cat in category_order_r1
]
custom_labels_ax3 = [custom_labels_r1[cat] for cat in category_order_r1]
# Set custom legend for ax3
ax3.legend(
    handles=custom_handles_ax3,
    labels=custom_labels_ax3,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax4 with rectangle patches and custom labels
custom_labels_llama = {
    "Llama-8B: claim": "Llama-3.1-8B (Claim Fact)",
    "Llama-8B: sentence": "Llama-3.1-8B (Sentence Fact)",
    "Llama-70B: claim": "Llama-3.1-70B (Claim Fact)",
    "Llama-70B: sentence": "Llama-3.1-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax4 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_llama[cat], linewidth=0) for cat in category_order_llama
]
custom_labels_ax4 = [custom_labels_llama[cat] for cat in category_order_llama]
# Set custom legend for ax4
ax4.legend(
    handles=custom_handles_ax4,
    labels=custom_labels_ax4,
    loc="lower right",
    fontsize=7,
)
# Remove legends in ax5-ax8 because they are duplicates of ax1-ax4
ax5.legend().remove()
ax6.legend().remove()
ax7.legend().remove()
ax8.legend().remove()

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
    bbox_to_anchor=(0.5, -0.05),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=10,
)

# Adjust Text (per adjustText package, should be called last in figure generation)
for text, ax in zip(
    [texts1, texts2, texts3, texts4, texts5, texts6, texts7, texts8],
    [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8],
):
    adjust_text(
        text,
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.3),
        expand=(2, 2),
        force_text=(2, 2),
        force_static=(3, 3),
        force_explode=(4, 4) if ax in (ax4, ax6) else (3, 3),
        ax=ax,
    )


# Save and Display Plot
save_dir = analysis_dir / "4_agreement_mcc_plots"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}.svg",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
# %% [markdown]
# ## Binarized Labels (Supported, Not Supported or Addressed)
# %% [markdown]
# ## Percent Agreement & MCC vs. Reference Context Length, Binarized Labels
# %% Author Type = LLM, Proposition Type = Claim, Binarized Labels
author_type = "llm"
proposition_type = "claim"

# Subset Percent Agreement Metric Bunch for Author Type & Proposition Type
pa_data = pa_mb_binarized.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
pa_data = coerce_categorical_types(pa_data)
# Subset MCC Metric Bunch for Author Type & Proposition Type
mcc_data = mcc_mb_binarized.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
mcc_data = coerce_categorical_types(mcc_data)

## Make Subplots
fig, [(ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)] = plt.subplots(
    nrows=2, ncols=4, figsize=(11, 8), layout="constrained"
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

### Gemma3
category_order_gemma = [
    "Gemma3-12B: claim",
    "Gemma3-12B: sentence",
    "Gemma3-27B: claim",
    "Gemma3-27B: sentence",
]
color_dict_gemma = dict(zip(category_order_gemma, sns.color_palette("Dark2")))

## Plot 1. Percent Agreement: Gemma
data_gemma = pa_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)


## Plot 5. MCC: Gemma3
data_gemma = mcc_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax5,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax5,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts5 = []
for i in range(len(x)):
    text = ax5.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts5.append(text)


### Qwen3
category_order_qwen = [
    "Qwen3-32B: claim",
    "Qwen3-32B: sentence",
    "Qwen3-30B-A3B-Thinking: claim",
    "Qwen3-30B-A3B-Thinking: sentence",
    "Qwen3-30B-A3B-Instruct: claim",
    "Qwen3-30B-A3B-Instruct: sentence",
]
color_dict_qwen = dict(zip(category_order_qwen, sns.color_palette("Dark2")))
## 2. Percent Agreement: Qwen3
data_qwen = pa_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)


## 6. MCC: Qwen3
data_qwen = mcc_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax6,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax6,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts6 = []
for i in range(len(x)):
    text = ax6.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts6.append(text)


### Deepseek-R1-Distill-Llama
category_order_r1 = ["R1-8B: claim", "R1-8B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict_r1 = dict(zip(category_order_r1, sns.color_palette("Dark2")))
## 3. Percent Agreement: Deepseek-R1-Distill-Llama
data_r1 = pa_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax3,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax3,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts3 = []
for i in range(len(x)):
    text = ax3.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts3.append(text)


## 7. MCC: R1-Distilled
data_r1 = mcc_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax7,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax7,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts7 = []
for i in range(len(x)):
    text = ax7.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts7.append(text)


### Llama3
category_order_llama = [
    "Llama-8B: claim",
    "Llama-8B: sentence",
    "Llama-70B: claim",
    "Llama-70B: sentence",
]
color_dict_llama = dict(zip(category_order_llama, sns.color_palette("Dark2")))
## 4. Percent Agreement: Llama3
data_llama = pa_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax4,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax4,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts4 = []
for i in range(len(x)):
    text = ax4.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts4.append(text)


## 8. MCC: Llama3
data_llama = mcc_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax8,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax8,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts8 = []
for i in range(len(x)):
    text = ax8.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts8.append(text)

# Set Gridlines
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    p.set_xlim(0, 4500)

for p in [ax1, ax2, ax3, ax4]:
    p.set_yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.set_ylim(0.60, 1.0)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

for p in [ax5, ax6, ax7, ax8]:
    p.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
    p.set_ylim(0.0, 0.4)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f} "))

# Make more compact by removing ticks & labels for plots other than first in each row
for p in [ax2, ax3, ax4, ax6, ax7, ax8]:
    p.set_yticklabels([])
    p.tick_params(axis="y", length=0)

# Set Titles & Labels
author_type_str = (
    "LLM-generated Brief Hospital Course"
    if author_type == "llm"
    else "Human-written Brief Hospital Course"
)
proposition_type_str = "Atomic Claim" if proposition_type == "claim" else "Sentence"
fig.suptitle(
    f"{author_type_str} Evaluated Using {proposition_type_str} Propositions (Binarized Labels)",
    fontsize=14,
)
ax1.set_title("Gemma-3", fontsize=10)
ax2.set_title("Qwen-3", fontsize=10)
ax3.set_title("Deepseek-R1-Distill-Llama", fontsize=10)
ax4.set_title("Llama-3.1", fontsize=10)
ax1.set_ylabel("Percent Agreement", fontsize=12)
ax2.set_ylabel("")
ax3.set_ylabel("")
ax4.set_ylabel("")
ax5.set_ylabel("Matthews Correlation Coefficient (MCC)", fontsize=12)
ax6.set_ylabel("")
ax7.set_ylabel("")
ax8.set_ylabel("")

fig.supxlabel("Mean Word Length of EHR Facts Reference Context Provided to LLM Judge", fontsize=12)
ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("")
ax4.set_xlabel("")
ax5.set_xlabel("")
ax6.set_xlabel("")
ax7.set_xlabel("")
ax8.set_xlabel("")

### Set Legends
# Custom legend for ax1 with rectangle patches and custom labels
custom_labels_gemma = {
    "Gemma3-12B: claim": "Gemma-3-12B (Claim Fact)",
    "Gemma3-12B: sentence": "Gemma-3-12B (Sentence Fact)",
    "Gemma3-27B: claim": "Gemma-3-27B (Claim Fact)",
    "Gemma3-27B: sentence": "Gemma-3-27B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax1 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_gemma[cat], linewidth=0) for cat in category_order_gemma
]
custom_labels_ax1 = [custom_labels_gemma[cat] for cat in category_order_gemma]
# Set custom legend for ax1
ax1.legend(
    handles=custom_handles_ax1,
    labels=custom_labels_ax1,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax2 with rectangle patches and custom labels
custom_labels_qwen = {
    "Qwen3-32B: claim": "Qwen-3-32B (Claim Fact)",
    "Qwen3-32B: sentence": "Qwen-3-32B (Sentence Fact)",
    "Qwen3-30B-A3B-Thinking: claim": "Qwen-3-30B-A3B-Thinking (Claim Fact)",
    "Qwen3-30B-A3B-Thinking: sentence": "Qwen-3-30B-A3B-Thinking (Sentence Fact)",
    "Qwen3-30B-A3B-Instruct: claim": "Qwen-3-30B-A3B-Instruct (Claim Fact)",
    "Qwen3-30B-A3B-Instruct: sentence": "Qwen-3-30B-A3B-Instruct (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax2 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_qwen[cat], linewidth=0) for cat in category_order_qwen
]
custom_labels_ax2 = [custom_labels_qwen[cat] for cat in category_order_qwen]
# Set custom legend for ax2
ax2.legend(
    handles=custom_handles_ax2,
    labels=custom_labels_ax2,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax3 with rectangle patches and custom labels
custom_labels_r1 = {
    "R1-8B: claim": "R1-Distill-Llama-8B (Claim Fact)",
    "R1-8B: sentence": "R1-Distill-Llama-8B (Sentence Fact)",
    "R1-70B: claim": "R1-Distill-Llama-70B (Claim Fact)",
    "R1-70B: sentence": "R1-Distill-Llama-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax3 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_r1[cat], linewidth=0) for cat in category_order_r1
]
custom_labels_ax3 = [custom_labels_r1[cat] for cat in category_order_r1]
# Set custom legend for ax3
ax3.legend(
    handles=custom_handles_ax3,
    labels=custom_labels_ax3,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax4 with rectangle patches and custom labels
custom_labels_llama = {
    "Llama-8B: claim": "Llama-3.1-8B (Claim Fact)",
    "Llama-8B: sentence": "Llama-3.1-8B (Sentence Fact)",
    "Llama-70B: claim": "Llama-3.1-70B (Claim Fact)",
    "Llama-70B: sentence": "Llama-3.1-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax4 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_llama[cat], linewidth=0) for cat in category_order_llama
]
custom_labels_ax4 = [custom_labels_llama[cat] for cat in category_order_llama]
# Set custom legend for ax4
ax4.legend(
    handles=custom_handles_ax4,
    labels=custom_labels_ax4,
    loc="lower right",
    fontsize=7,
)
# Remove legends in ax5-ax8 because they are duplicates of ax1-ax4
ax5.legend().remove()
ax6.legend().remove()
ax7.legend().remove()
ax8.legend().remove()

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
    bbox_to_anchor=(0.5, -0.05),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=10,
)

# Adjust Text (per adjustText package, should be called last in figure generation)
for text, ax in zip(
    [texts1, texts2, texts3, texts4, texts5, texts6, texts7, texts8],
    [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8],
):
    adjust_text(
        text,
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.3),
        expand=(2, 2),
        force_text=(2, 2),
        force_static=(3, 3),
        force_explode=(3, 3),
        ax=ax,
    )


# Save and Display Plot
save_dir = analysis_dir / "4_agreement_mcc_plots"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-binarized.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-binarized.svg",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()

# %% Author Type = LLM, Proposition Type = Sentence, Binarized Labels
author_type = "llm"
proposition_type = "sentence"

# Subset Percent Agreement Metric Bunch for Author Type & Proposition Type
pa_data = pa_mb_binarized.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
pa_data = coerce_categorical_types(pa_data)
# Subset MCC Metric Bunch for Author Type & Proposition Type
mcc_data = mcc_mb_binarized.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
mcc_data = coerce_categorical_types(mcc_data)

## Make Subplots
fig, [(ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)] = plt.subplots(
    nrows=2, ncols=4, figsize=(11, 8), layout="constrained"
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

### Gemma3
category_order_gemma = [
    "Gemma3-12B: claim",
    "Gemma3-12B: sentence",
    "Gemma3-27B: claim",
    "Gemma3-27B: sentence",
]
color_dict_gemma = dict(zip(category_order_gemma, sns.color_palette("Dark2")))

## Plot 1. Percent Agreement: Gemma
data_gemma = pa_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)


## Plot 5. MCC: Gemma3
data_gemma = mcc_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax5,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax5,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts5 = []
for i in range(len(x)):
    text = ax5.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts5.append(text)


### Qwen3
category_order_qwen = [
    "Qwen3-32B: claim",
    "Qwen3-32B: sentence",
    "Qwen3-30B-A3B-Thinking: claim",
    "Qwen3-30B-A3B-Thinking: sentence",
    "Qwen3-30B-A3B-Instruct: claim",
    "Qwen3-30B-A3B-Instruct: sentence",
]
color_dict_qwen = dict(zip(category_order_qwen, sns.color_palette("Dark2")))
## 2. Percent Agreement: Qwen3
data_qwen = pa_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)


## 6. MCC: Qwen3
data_qwen = mcc_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax6,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax6,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts6 = []
for i in range(len(x)):
    text = ax6.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts6.append(text)


### Deepseek-R1-Distill-Llama
category_order_r1 = ["R1-8B: claim", "R1-8B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict_r1 = dict(zip(category_order_r1, sns.color_palette("Dark2")))
## 3. Percent Agreement: Deepseek-R1-Distill-Llama
data_r1 = pa_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax3,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax3,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts3 = []
for i in range(len(x)):
    text = ax3.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts3.append(text)


## 7. MCC: R1-Distilled
data_r1 = mcc_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax7,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax7,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts7 = []
for i in range(len(x)):
    text = ax7.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts7.append(text)


### Llama3
category_order_llama = [
    "Llama-8B: claim",
    "Llama-8B: sentence",
    "Llama-70B: claim",
    "Llama-70B: sentence",
]
color_dict_llama = dict(zip(category_order_llama, sns.color_palette("Dark2")))
## 4. Percent Agreement: Llama3
data_llama = pa_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax4,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax4,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts4 = []
for i in range(len(x)):
    text = ax4.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts4.append(text)


## 8. MCC: Llama3
data_llama = mcc_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax8,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax8,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts8 = []
for i in range(len(x)):
    text = ax8.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts8.append(text)

# Set Gridlines
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    p.set_xlim(0, 4500)

for p in [ax1, ax2, ax3, ax4]:
    p.set_yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.set_ylim(0.60, 1.0)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

for p in [ax5, ax6, ax7, ax8]:
    p.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4])
    p.set_ylim(-0.07, 0.3)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f} "))

# Make more compact by removing ticks & labels for plots other than first in each row
for p in [ax2, ax3, ax4, ax6, ax7, ax8]:
    p.set_yticklabels([])
    p.tick_params(axis="y", length=0)

# Set Titles & Labels
author_type_str = (
    "LLM-generated Brief Hospital Course"
    if author_type == "llm"
    else "Human-written Brief Hospital Course"
)
proposition_type_str = "Atomic Claim" if proposition_type == "claim" else "Sentence"
fig.suptitle(
    f"{author_type_str} Evaluated Using {proposition_type_str} Propositions (Binarized Labels)",
    fontsize=14,
)
ax1.set_title("Gemma-3", fontsize=10)
ax2.set_title("Qwen-3", fontsize=10)
ax3.set_title("Deepseek-R1-Distill-Llama", fontsize=10)
ax4.set_title("Llama-3.1", fontsize=10)
ax1.set_ylabel("Percent Agreement", fontsize=12)
ax2.set_ylabel("")
ax3.set_ylabel("")
ax4.set_ylabel("")
ax5.set_ylabel("Matthews Correlation Coefficient (MCC)", fontsize=12)
ax6.set_ylabel("")
ax7.set_ylabel("")
ax8.set_ylabel("")

fig.supxlabel("Mean Word Length of EHR Facts Reference Context Provided to LLM Judge", fontsize=12)
ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("")
ax4.set_xlabel("")
ax5.set_xlabel("")
ax6.set_xlabel("")
ax7.set_xlabel("")
ax8.set_xlabel("")

### Set Legends
# Custom legend for ax1 with rectangle patches and custom labels
custom_labels_gemma = {
    "Gemma3-12B: claim": "Gemma-3-12B (Claim Fact)",
    "Gemma3-12B: sentence": "Gemma-3-12B (Sentence Fact)",
    "Gemma3-27B: claim": "Gemma-3-27B (Claim Fact)",
    "Gemma3-27B: sentence": "Gemma-3-27B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax1 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_gemma[cat], linewidth=0) for cat in category_order_gemma
]
custom_labels_ax1 = [custom_labels_gemma[cat] for cat in category_order_gemma]
# Set custom legend for ax1
ax1.legend(
    handles=custom_handles_ax1,
    labels=custom_labels_ax1,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax2 with rectangle patches and custom labels
custom_labels_qwen = {
    "Qwen3-32B: claim": "Qwen-3-32B (Claim Fact)",
    "Qwen3-32B: sentence": "Qwen-3-32B (Sentence Fact)",
    "Qwen3-30B-A3B-Thinking: claim": "Qwen-3-30B-A3B-Thinking (Claim Fact)",
    "Qwen3-30B-A3B-Thinking: sentence": "Qwen-3-30B-A3B-Thinking (Sentence Fact)",
    "Qwen3-30B-A3B-Instruct: claim": "Qwen-3-30B-A3B-Instruct (Claim Fact)",
    "Qwen3-30B-A3B-Instruct: sentence": "Qwen-3-30B-A3B-Instruct (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax2 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_qwen[cat], linewidth=0) for cat in category_order_qwen
]
custom_labels_ax2 = [custom_labels_qwen[cat] for cat in category_order_qwen]
# Set custom legend for ax2
ax2.legend(
    handles=custom_handles_ax2,
    labels=custom_labels_ax2,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax3 with rectangle patches and custom labels
custom_labels_r1 = {
    "R1-8B: claim": "R1-Distill-Llama-8B (Claim Fact)",
    "R1-8B: sentence": "R1-Distill-Llama-8B (Sentence Fact)",
    "R1-70B: claim": "R1-Distill-Llama-70B (Claim Fact)",
    "R1-70B: sentence": "R1-Distill-Llama-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax3 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_r1[cat], linewidth=0) for cat in category_order_r1
]
custom_labels_ax3 = [custom_labels_r1[cat] for cat in category_order_r1]
# Set custom legend for ax3
ax3.legend(
    handles=custom_handles_ax3,
    labels=custom_labels_ax3,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax4 with rectangle patches and custom labels
custom_labels_llama = {
    "Llama-8B: claim": "Llama-3.1-8B (Claim Fact)",
    "Llama-8B: sentence": "Llama-3.1-8B (Sentence Fact)",
    "Llama-70B: claim": "Llama-3.1-70B (Claim Fact)",
    "Llama-70B: sentence": "Llama-3.1-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax4 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_llama[cat], linewidth=0) for cat in category_order_llama
]
custom_labels_ax4 = [custom_labels_llama[cat] for cat in category_order_llama]
# Set custom legend for ax4
ax4.legend(
    handles=custom_handles_ax4,
    labels=custom_labels_ax4,
    loc="lower right",
    fontsize=7,
)
# Remove legends in ax5-ax8 because they are duplicates of ax1-ax4
ax5.legend().remove()
ax6.legend().remove()
ax7.legend().remove()
ax8.legend().remove()

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
    bbox_to_anchor=(0.5, -0.05),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=10,
)

# Adjust Text (per adjustText package, should be called last in figure generation)
for text, ax in zip(
    [texts1, texts2, texts3, texts4, texts5, texts6, texts7, texts8],
    [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8],
):
    adjust_text(
        text,
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.3),
        expand=(2, 2),
        force_text=(2, 2),
        force_static=(3, 3),
        force_explode=(4, 4) if ax == ax7 else (3, 3),
        ax=ax,
    )


# Save and Display Plot
save_dir = analysis_dir / "4_agreement_mcc_plots"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-binarized.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-binarized.svg",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
# %% Author Type = Human, Proposition Type = Claim, Binarized Labels
author_type = "human"
proposition_type = "claim"

# Subset Percent Agreement Metric Bunch for Author Type & Proposition Type
pa_data = pa_mb_binarized.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
pa_data = coerce_categorical_types(pa_data)
# Subset MCC Metric Bunch for Author Type & Proposition Type
mcc_data = mcc_mb_binarized.metrics.query(
    f"author_type == '{author_type}' & proposition_type == '{proposition_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
mcc_data = coerce_categorical_types(mcc_data)

## Make Subplots
fig, [(ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)] = plt.subplots(
    nrows=2, ncols=4, figsize=(11, 8), layout="constrained"
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

### Gemma3
category_order_gemma = [
    "Gemma3-12B: claim",
    "Gemma3-12B: sentence",
    "Gemma3-27B: claim",
    "Gemma3-27B: sentence",
]
color_dict_gemma = dict(zip(category_order_gemma, sns.color_palette("Dark2")))

## Plot 1. Percent Agreement: Gemma
data_gemma = pa_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax1,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax1,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts1 = []
for i in range(len(x)):
    text = ax1.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts1.append(text)


## Plot 5. MCC: Gemma3
data_gemma = mcc_data.query("model == 'Gemma3-12B' or model == 'Gemma3-27B'")
data_gemma = data_gemma.assign(
    category=data_gemma.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax5,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_gemma,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_gemma,
    palette=color_dict_gemma,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax5,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_gemma.query("fact_type == 'claim' & top_n == 150"),
        data_gemma.query("fact_type == 'sentence' & top_n == 100"),
    ]
)

x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_gemma[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts5 = []
for i in range(len(x)):
    text = ax5.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts5.append(text)


### Qwen3
category_order_qwen = [
    "Qwen3-32B: claim",
    "Qwen3-32B: sentence",
    "Qwen3-30B-A3B-Thinking: claim",
    "Qwen3-30B-A3B-Thinking: sentence",
    "Qwen3-30B-A3B-Instruct: claim",
    "Qwen3-30B-A3B-Instruct: sentence",
]
color_dict_qwen = dict(zip(category_order_qwen, sns.color_palette("Dark2")))
## 2. Percent Agreement: Qwen3
data_qwen = pa_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax2,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax2,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts2 = []
for i in range(len(x)):
    text = ax2.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts2.append(text)


## 6. MCC: Qwen3
data_qwen = mcc_data.query(
    "model == 'Qwen3-32B' or model == 'Qwen3-30B-A3B-Thinking' or model == 'Qwen3-30B-A3B-Instruct'"
)
data_qwen = data_qwen.assign(
    category=data_qwen.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create line plot
sns.lineplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax6,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_qwen,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_qwen,
    palette=color_dict_qwen,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax6,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_qwen.query("fact_type == 'claim' & top_n == 150"),
        data_qwen.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_qwen[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts6 = []
for i in range(len(x)):
    text = ax6.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts6.append(text)


### Deepseek-R1-Distill-Llama
category_order_r1 = ["R1-8B: claim", "R1-8B: sentence", "R1-70B: claim", "R1-70B: sentence"]
color_dict_r1 = dict(zip(category_order_r1, sns.color_palette("Dark2")))
## 3. Percent Agreement: Deepseek-R1-Distill-Llama
data_r1 = pa_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax3,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax3,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts3 = []
for i in range(len(x)):
    text = ax3.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts3.append(text)


## 7. MCC: R1-Distilled
data_r1 = mcc_data.query("model == 'R1-8B' or model == 'R1-70B'")
data_r1 = data_r1.assign(
    category=data_r1.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax7,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_r1,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_r1,
    palette=color_dict_r1,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax7,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_r1.query("fact_type == 'claim' & top_n == 150"),
        data_r1.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_r1[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts7 = []
for i in range(len(x)):
    text = ax7.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts7.append(text)


### Llama3
category_order_llama = [
    "Llama-8B: claim",
    "Llama-8B: sentence",
    "Llama-70B: claim",
    "Llama-70B: sentence",
]
color_dict_llama = dict(zip(category_order_llama, sns.color_palette("Dark2")))
## 4. Percent Agreement: Llama3
data_llama = pa_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax4,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax4,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.1%}" for item in y]
texts4 = []
for i in range(len(x)):
    text = ax4.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts4.append(text)


## 8. MCC: Llama3
data_llama = mcc_data.query("model == 'Llama-8B' or model == 'Llama-70B'")
data_llama = data_llama.assign(
    category=data_llama.apply(lambda row: f"{row.model}: {row.fact_type}", axis="columns")
)
# Create Line Plot
sns.lineplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
    ax=ax8,
)
# Overlay Scatter Plot
sns.scatterplot(
    data=data_llama,
    x="mean_word_length",
    y="value",
    hue="category",
    style="top_n",
    hue_order=category_order_llama,
    palette=color_dict_llama,
    markers=marker_map,
    legend=False,  # Avoid duplicate legends
    s=25,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    ax=ax8,
)
# Add Text Annotations to Scatter Plot
annotation_subset = pd.concat(
    [
        data_llama.query("fact_type == 'claim' & top_n == 150"),
        data_llama.query("fact_type == 'sentence' & top_n == 100"),
    ]
)
x = annotation_subset.loc[:, "mean_word_length"].to_numpy()
y = annotation_subset.loc[:, "value"].to_numpy()
color = [color_dict_llama[x] for x in annotation_subset.loc[:, "category"].to_numpy()]
text_str = [f"{item:.2f}" for item in y]
texts8 = []
for i in range(len(x)):
    text = ax8.text(x[i], y[i], text_str[i], fontsize=8, color=color[i])
    texts8.append(text)

# Set Gridlines
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.grid(which="both", linestyle="--", alpha=0.5)
    p.set_axisbelow(True)

# Set Axis & Ticks
for p in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]:
    p.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
    p.set_xlim(0, 4500)

for p in [ax1, ax2, ax3, ax4]:
    p.set_yticks([0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.set_ylim(0.55, 0.9)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

for p in [ax5, ax6, ax7, ax8]:
    p.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    p.set_ylim(0.2, 0.7)
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f} "))

# Make more compact by removing ticks & labels for plots other than first in each row
for p in [ax2, ax3, ax4, ax6, ax7, ax8]:
    p.set_yticklabels([])
    p.tick_params(axis="y", length=0)

# Set Titles & Labels
author_type_str = (
    "LLM-generated Brief Hospital Course"
    if author_type == "llm"
    else "Human-written Brief Hospital Course"
)
proposition_type_str = "Atomic Claim" if proposition_type == "claim" else "Sentence"
fig.suptitle(
    f"{author_type_str} Evaluated Using {proposition_type_str} Propositions (Binarized Labels)",
    fontsize=14,
)
ax1.set_title("Gemma-3", fontsize=10)
ax2.set_title("Qwen-3", fontsize=10)
ax3.set_title("Deepseek-R1-Distill-Llama", fontsize=10)
ax4.set_title("Llama-3.1", fontsize=10)
ax1.set_ylabel("Percent Agreement", fontsize=12)
ax2.set_ylabel("")
ax3.set_ylabel("")
ax4.set_ylabel("")
ax5.set_ylabel("Matthews Correlation Coefficient (MCC)", fontsize=12)
ax6.set_ylabel("")
ax7.set_ylabel("")
ax8.set_ylabel("")

fig.supxlabel("Mean Word Length of EHR Facts Reference Context Provided to LLM Judge", fontsize=12)
ax1.set_xlabel("")
ax2.set_xlabel("")
ax3.set_xlabel("")
ax4.set_xlabel("")
ax5.set_xlabel("")
ax6.set_xlabel("")
ax7.set_xlabel("")
ax8.set_xlabel("")

### Set Legends
# Custom legend for ax1 with rectangle patches and custom labels
custom_labels_gemma = {
    "Gemma3-12B: claim": "Gemma-3-12B (Claim Fact)",
    "Gemma3-12B: sentence": "Gemma-3-12B (Sentence Fact)",
    "Gemma3-27B: claim": "Gemma-3-27B (Claim Fact)",
    "Gemma3-27B: sentence": "Gemma-3-27B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax1 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_gemma[cat], linewidth=0) for cat in category_order_gemma
]
custom_labels_ax1 = [custom_labels_gemma[cat] for cat in category_order_gemma]
# Set custom legend for ax1
ax1.legend(
    handles=custom_handles_ax1,
    labels=custom_labels_ax1,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax2 with rectangle patches and custom labels
custom_labels_qwen = {
    "Qwen3-32B: claim": "Qwen-3-32B (Claim Fact)",
    "Qwen3-32B: sentence": "Qwen-3-32B (Sentence Fact)",
    "Qwen3-30B-A3B-Thinking: claim": "Qwen-3-30B-A3B-Thinking (Claim Fact)",
    "Qwen3-30B-A3B-Thinking: sentence": "Qwen-3-30B-A3B-Thinking (Sentence Fact)",
    "Qwen3-30B-A3B-Instruct: claim": "Qwen-3-30B-A3B-Instruct (Claim Fact)",
    "Qwen3-30B-A3B-Instruct: sentence": "Qwen-3-30B-A3B-Instruct (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax2 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_qwen[cat], linewidth=0) for cat in category_order_qwen
]
custom_labels_ax2 = [custom_labels_qwen[cat] for cat in category_order_qwen]
# Set custom legend for ax2
ax2.legend(
    handles=custom_handles_ax2,
    labels=custom_labels_ax2,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax3 with rectangle patches and custom labels
custom_labels_r1 = {
    "R1-8B: claim": "R1-Distill-Llama-8B (Claim Fact)",
    "R1-8B: sentence": "R1-Distill-Llama-8B (Sentence Fact)",
    "R1-70B: claim": "R1-Distill-Llama-70B (Claim Fact)",
    "R1-70B: sentence": "R1-Distill-Llama-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax3 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_r1[cat], linewidth=0) for cat in category_order_r1
]
custom_labels_ax3 = [custom_labels_r1[cat] for cat in category_order_r1]
# Set custom legend for ax3
ax3.legend(
    handles=custom_handles_ax3,
    labels=custom_labels_ax3,
    loc="lower right",
    fontsize=7,
)
# Custom legend for ax4 with rectangle patches and custom labels
custom_labels_llama = {
    "Llama-8B: claim": "Llama-3.1-8B (Claim Fact)",
    "Llama-8B: sentence": "Llama-3.1-8B (Sentence Fact)",
    "Llama-70B: claim": "Llama-3.1-70B (Claim Fact)",
    "Llama-70B: sentence": "Llama-3.1-70B (Sentence Fact)",
}
# Create rectangle patch handles with existing colors
custom_handles_ax4 = [
    Rectangle((0, 0), 1, 1, fc=color_dict_llama[cat], linewidth=0) for cat in category_order_llama
]
custom_labels_ax4 = [custom_labels_llama[cat] for cat in category_order_llama]
# Set custom legend for ax4
ax4.legend(
    handles=custom_handles_ax4,
    labels=custom_labels_ax4,
    loc="lower right",
    fontsize=7,
)
# Remove legends in ax5-ax8 because they are duplicates of ax1-ax4
ax5.legend().remove()
ax6.legend().remove()
ax7.legend().remove()
ax8.legend().remove()

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
    bbox_to_anchor=(0.5, -0.05),  # Moves the legend outside to the bottom
    borderaxespad=0,  # Removes padding between axes and legend
    ncol=4,
    fontsize=10,
)

# Adjust Text (per adjustText package, should be called last in figure generation)
for text, ax in zip(
    [texts1, texts2, texts3, texts4, texts5, texts6, texts7, texts8],
    [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8],
):
    adjust_text(
        text,
        arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.3),
        expand=(2, 2),
        force_text=(2, 2),
        force_static=(3, 3),
        force_explode=(3, 3),
        ax=ax,
    )


# Save and Display Plot
save_dir = analysis_dir / "4_agreement_mcc_plots"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-binarized.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-binarized.svg",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
# %%
