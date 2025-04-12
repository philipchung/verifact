# %% [markdown]
# ## Sensitivity Analysis for Top N, Retrieval Method, Reference Context, Reference Only Admission
#
# For each experiment, we compute agreement for each
# VeriFact AI System Variation vs. Ground Truth Human Clinician Label
#
# Percent Agreement & Gwet's AC1 is computed for both original and binarized labels
# (treating Not Addressed labels as Not Supported). Then the results are displayed in table format.
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

# %% [markdown]
# ## Sensitivity Analysis: Top N EHR Facts in Reference Context
# %%
# Top N Sensitivity Analysis: 5, 10, 25, 50, 75, 100, 125, 150
author_type = "llm"
proposition_type = "claim"
fact_type = "claim"
data = mb.metrics.query(
    f"author_type == '{author_type}' "
    f"& proposition_type == '{proposition_type}' & fact_type == '{fact_type}' "
    "& retrieval_method == 'rerank' & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
data = coerce_categorical_types(data)

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

## Make Plot
model_order = ["Llama-8B", "Llama-70B", "R1-8B", "R1-70B"]
color_dict = dict(zip(model_order, sns.color_palette("muted")))
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 5), layout="constrained", sharey=True)
data1 = data.query("model in @model_order")
# Line Plot
ax = sns.lineplot(
    data=data1,
    x="top_n",
    y="value",
    hue="model",
    hue_order=model_order,
    ax=ax,
    palette=color_dict,
    marker=None,  # Ensure the line is drawn without interfering with markers
    dashes="",  # Keep solid lines
    alpha=0.5,  # Reduce line opacity for better visibility
)
# Scatter Plot
sns.scatterplot(
    data=data1,
    x="top_n",
    y="value",
    hue="model",
    style="top_n",
    markers=marker_map,
    palette=color_dict,
    legend=False,  # Avoid duplicate legends
    s=50,  # Adjust marker size for better visibility
    edgecolor="black",  # Add black edge color for better visibility
    alpha=0.75,  # Reduce marker opacity for better visibility
    ax=ax,
)

# Set Gridlines
ax.grid(which="both", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Set Axis & Ticks
ax.set_xticks([5, 10, 25, 50, 75, 100, 125, 150])
ax.set_yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

# Set Legend
ax.legend(title="Model", loc="lower right")

# Labels and formatting
ax.set_title(
    f"Author Type: {author_type}\nProposition Type: {proposition_type}, Fact Type: {fact_type}"
)
ax.set_xlabel("Number of EHR Facts in Reference Context")
ax.set_ylabel("Percent Agreement")

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
)

# Save and Display Plot
save_dir = Path.cwd() / "3_sensitivity_analysis_plots"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-{fact_type}_top_n.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()

# %% [markdown]
# ## Sensitivity Analysis: Retrieval Method
# %%
# Sensitivity Analysis: Retrieval Method
author_type = "llm"
proposition_type = "claim"
fact_type = "claim"
data = mb.metrics.query(
    f"author_type == '{author_type}' "
    f"& proposition_type == '{proposition_type}' & fact_type == '{fact_type}' "
    "& top_n == 50 & reference_format == 'absolute time' "
    "& reference_only_admission == True"
)
data = coerce_categorical_types(data)

# Make Plot
model_order = ["Llama-8B", "Llama-70B", "R1-8B", "R1-70B"]
retrieval_method_order = ["Dense", "Sparse", "Hybrid", "Rerank"]
color_dict = dict(zip(retrieval_method_order, sns.color_palette("muted")))
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
ax = sns.barplot(
    data=data,
    x="model",
    y="value",
    hue="retrieval_method",
    order=model_order,
    hue_order=retrieval_method_order,
    ax=ax,
    palette=color_dict,
    errorbar=None,  # Disable built-in error bars
)

# Loop Through Containers & Bars to Manually Add Error Bars & Bar Labels
if data["ci_lower"].notnull().all() and data["ci_upper"].notnull().all():
    # Loop Through Each Container (Group of Bars for each Model on X-Axis)
    for container, model in zip(ax.containers, model_order):
        # Loop Through Each Bar in the Container (Corresponds to Retrieval Method)
        for bar, retrieval_method in zip(container, retrieval_method_order):
            # Get Bar's X and Y Values
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            # Get Data for the Current Bar's Confidence Intervals
            bar_data = data.query(f"model == '{model}' & retrieval_method == '{retrieval_method}'")
            yerr_lower = (bar_data["value"] - bar_data["ci_lower"]).item()
            yerr_upper = (bar_data["ci_upper"] - bar_data["value"]).item()
            yerr = [[yerr_lower], [yerr_upper]]
            # Plot Error Bars
            ax.errorbar(
                x,
                y,
                yerr=[[yerr_lower], [yerr_upper]],
                fmt="none",  # No marker, just the error bar
                capsize=3,
                capthick=1,
                color="black",
                alpha=0.5,
                lw=0.5,
            )
        # Set Bar Labels
        ax.bar_label(container, fmt="{:.1%}", fontsize=8, padding=15)

# Set Axis
ax.set_ylim(bottom=0.60)
ax.set_yticks([0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.9, 0.95, 1.0])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

# Set Gridlines
ax.grid(which="both", axis="y", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Set Legend
ax.legend(title="Retrieval Method", loc="upper left")

# Labels and formatting
ax.set_title(
    f"Author Type: {author_type}, Proposition Type: {proposition_type}, Fact Type: {fact_type}"
)
ax.set_xlabel("Model")
ax.set_ylabel("Percent Agreement")

# Save and Display Plot
save_dir = Path.cwd() / "3_sensitivity_analysis_plots"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-{fact_type}_retrieval_method.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()

# %% [markdown]
# ## Sensitivity Analysis: Reference Context Format
# %%
# Sensitivity Analysis: Reference Context Format
author_type = "llm"
proposition_type = "claim"
fact_type = "claim"
data = mb.metrics.query(
    f"author_type == '{author_type}' "
    f"& proposition_type == '{proposition_type}' & fact_type == '{fact_type}' "
    "& top_n == 50 & retrieval_method == 'rerank' "
    "& reference_only_admission == True"
)
data = coerce_categorical_types(data)

# Make Plot
model_order = ["Llama-8B", "Llama-70B", "R1-8B", "R1-70B"]
reference_format_order = ["Score", "Absolute Time", "Relative Time"]
color_dict = dict(zip(reference_format_order, sns.color_palette("muted")))
fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
ax = sns.barplot(
    data=data,
    x="model",
    y="value",
    hue="reference_format",
    order=model_order,
    hue_order=reference_format_order,
    palette=color_dict,
    errorbar=None,  # Disable built-in error bars
    ax=ax,
)

# Loop Through Containers & Bars to Manually Add Error Bars & Bar Labels
if data["ci_lower"].notnull().all() and data["ci_upper"].notnull().all():
    # Loop Through Each Container (Group of Bars for each Model on X-Axis)
    for container, reference_format in zip(ax.containers, reference_format_order):
        # Loop Through Each Bar in the Container (Corresponds to Retrieval Method)
        for bar, model in zip(container, model_order):
            # Get Bar's X and Y Values
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            # Get Data for the Current Bar's Confidence Intervals
            bar_data = data.query(f"model == '{model}' & reference_format == '{reference_format}'")
            yerr_lower = (bar_data["value"] - bar_data["ci_lower"]).item()
            yerr_upper = (bar_data["ci_upper"] - bar_data["value"]).item()
            yerr = [[yerr_lower], [yerr_upper]]
            # Plot Error Bars
            ax.errorbar(
                x,
                y,
                yerr=[[yerr_lower], [yerr_upper]],
                fmt="none",  # No marker, just the error bar
                capsize=3,
                capthick=1,
                color="black",
                alpha=0.5,
                lw=0.5,
            )
        # Set Bar Labels
        ax.bar_label(container, fmt="{:.1%}", fontsize=8, padding=20)

# Set Axis
ax.set_ylim(bottom=0.6)
ax.set_yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

# Set Gridlines
ax.grid(which="both", axis="y", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Set Legend
ax.legend(title="Reference Context Format", loc="upper left")

# Labels and formatting
ax.set_title(
    f"Author Type: {author_type}, Proposition Type: {proposition_type}, Fact Type: {fact_type}"
)
ax.set_xlabel("Model")
ax.set_ylabel("Percent Agreement")

# Save and Display Plot
save_dir = Path.cwd() / "3_sensitivity_analysis_plots"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-{fact_type}_reference_context_format.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
# %% [markdown]
# ## Sensitivity Analysis: Reference Only Admission
# %%
# Sensitivity Analysis: Reference Only Admission
author_type = "llm"
proposition_type = "claim"
fact_type = "claim"
data = mb.metrics.query(
    f"author_type == '{author_type}' "
    f"& proposition_type == '{proposition_type}' & fact_type == '{fact_type}' "
    "& top_n == 50 & retrieval_method == 'rerank' "
    "& reference_format == 'absolute time' "
)
data = coerce_categorical_types(data)

# Make Plot
model_order = ["Llama-8B", "Llama-70B", "R1-8B", "R1-70B"]
reference_only_admission_order = ["Yes", "No"]
color_dict = dict(zip(reference_only_admission_order, sns.color_palette("muted")))
fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
ax = sns.barplot(
    data=data,
    x="model",
    y="value",
    hue="reference_only_admission",
    order=model_order,
    hue_order=reference_only_admission_order,
    palette=color_dict,
    errorbar=None,  # Disable built-in error bars
    ax=ax,
)

# Loop Through Containers & Bars to Manually Add Error Bars & Bar Labels
if data["ci_lower"].notnull().all() and data["ci_upper"].notnull().all():
    # Loop Through Each Container (Group of Bars for each Model on X-Axis)
    for container, reference_only_admission in zip(ax.containers, reference_only_admission_order):
        # Loop Through Each Bar in the Container (Corresponds to Retrieval Method)
        for bar, model in zip(container, model_order):
            # Get Bar's X and Y Values
            x = bar.get_x() + bar.get_width() / 2
            y = bar.get_height()
            # Get Data for the Current Bar's Confidence Intervals
            bar_data = data.query(
                f"model == '{model}' & reference_only_admission == '{reference_only_admission}'"
            )
            yerr_lower = (bar_data["value"] - bar_data["ci_lower"]).item()
            yerr_upper = (bar_data["ci_upper"] - bar_data["value"]).item()
            yerr = [[yerr_lower], [yerr_upper]]
            # Plot Error Bars
            ax.errorbar(
                x,
                y,
                yerr=[[yerr_lower], [yerr_upper]],
                fmt="none",  # No marker, just the error bar
                capsize=3,
                capthick=1,
                color="black",
                alpha=0.5,
                lw=0.5,
            )
        # Set Bar Labels
        ax.bar_label(container, fmt="{:.1%}", fontsize=8, padding=20)

# Set Axis
ax.set_ylim(bottom=0.6)
ax.set_yticks([0.6, 0.65, 0.7, 0.75, 0.80, 0.85, 0.9, 0.95, 1.0])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

# Set Gridlines
ax.grid(which="both", axis="y", linestyle="--", alpha=0.5)
ax.set_axisbelow(True)

# Set Legend
ax.legend(title="Only Retrieve EHR Facts \nFrom Current Admission", loc="upper left")

# Labels and formatting
ax.set_title(
    f"Author Type: {author_type}, Proposition Type: {proposition_type}, Fact Type: {fact_type}"
)
ax.set_xlabel("Model")
ax.set_ylabel("Percent Agreement")

# Save and Display Plot
save_dir = Path.cwd() / "3_sensitivity_analysis_plots"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}-{fact_type}_reference_only_admission.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()
# %% [markdown]
# ## Compare Reasoning & Non-Reasoning Models of Specific Sizes with Sentences vs. Claim Facts
# %%
# Percent Agreement vs. Reference Context Length for 8B and 70B Models
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
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(9, 9), layout="constrained", sharey=True)

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
x = data_8B.loc[:, "mean_word_length"].to_numpy()
y = data_8B.loc[:, "value"].to_numpy()
color = [color_dict1[x] for x in data_8B.loc[:, "category"].to_numpy()]
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
x = data_70B.loc[:, "mean_word_length"].to_numpy()
y = data_70B.loc[:, "value"].to_numpy()
color = [color_dict2[x] for x in data_70B.loc[:, "category"].to_numpy()]
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
    p.set_xticks([0, 500, 1000, 1500, 2000, 2500, 3000, 3500])
    p.set_yticks([0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0])
    p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))


# Set Titles & Labels
fig.suptitle(
    f"Llama 3 & Deepseek-R1-Distilled Models\n"
    f"Author Type: {author_type}, Proposition Type: {proposition_type}"
)
ax1.set_title("8-Billion Parameter Models")
ax2.set_title("70-Billion Parameter Models")
fig.supylabel("Percent Agreement")
ax1.set_ylabel("")
ax2.set_ylabel("")
fig.supxlabel("Mean Word Length of EHR Facts Reference Context Provided to LLM-as-a-Judge")
ax1.set_xlabel("")
ax2.set_xlabel("")

# Set Legends
ax1.legend(title="Model & EHR Fact Type", loc="lower right")
ax2.legend(title="Model & EHR Fact Type", loc="lower right")

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
)

# Adjust Text Labels
adjust_text(
    texts1,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2.5, 2.5),
    force_text=(2, 2),
    force_static=(1, 1),
    force_explode=(1.5, 1.5),
    ax=ax1,
)
adjust_text(
    texts2,
    arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.5),
    expand=(2.5, 2.5),
    force_text=(2, 2),
    force_static=(1, 1),
    force_explode=(1.5, 1.5),
    ax=ax2,
)

# Save and Display Plot
save_dir = Path.cwd() / "3_sensitivity_analysis_plots"
save_dir.mkdir(exist_ok=True)
fig.savefig(
    save_dir / f"{author_type}-{proposition_type}_reference_context_length_llama_models.png",
    bbox_inches="tight",
    dpi=300,
    transparent=True,
)
fig.show()

# %%
