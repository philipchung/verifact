# %% [markdown]
# ## Confusion Matricies for Best VeriFact AI System vs. Human Clinician Ground Truth Labels
#
# This script generates confusion matricies comparing classification performance for
# all LLM-as-a-Judge combinations of author type, proposition type, and fact type,
# in both the original and binarized label spaces.
# %%
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from release_data import load_release_annotations, load_release_ground_truth
from sklearn.metrics import confusion_matrix
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
human_gt = load_release_ground_truth(release_dir)
ai_verdicts = load_release_annotations(release_dir, models=models)
# %%
# Isolate Data for Each Model & Scenario
ai_verdicts_dict: dict[str, pd.DataFrame] = {}
y_pred_dict: dict[str, pd.Series] = {}
y_true_dict: dict[str, pd.Series] = {}
labels = ["Supported", "Not Supported", "Not Addressed"]
# Isolate Dataframes of AI Verdicts for Each Model & Scenario
for author_type, proposition_type, fact_type in [
    ("llm", "claim", "claim"),
    ("llm", "claim", "sentence"),
    ("llm", "sentence", "claim"),
    ("llm", "sentence", "sentence"),
    ("human", "claim", "claim"),
    ("human", "claim", "sentence"),
]:
    top_n = 150 if fact_type == "claim" else 100
    for model in models:
        id_tuple = (model, author_type, proposition_type, fact_type)
        print(
            f"Model: '{model}', Author Type: '{author_type}', "
            f"Proposition Type: '{proposition_type}', Fact Type: '{fact_type}'"
        )
        # Isolate AI Verdicts for Model & Scenario
        ai_verdicts_dict[model] = ai_verdicts.query(
            f"model == '{model}' & author_type == '{author_type}' & "
            f"proposition_type == '{proposition_type}' & fact_type == '{fact_type}' &"
            f"top_n == {top_n} & retrieval_method == 'rerank' & "
            f"reference_format == 'absolute time' & reference_only_admission == True"
        )
        # Get Predicted & Ground Truth Labels
        y_pred_dict[id_tuple] = (
            ai_verdicts_dict[model]
            .loc[:, ["proposition_id", "verdict"]]
            .set_index("proposition_id")
            .sort_index()
        ).squeeze()
        human_gt_subset = human_gt.query(
            f"author_type == '{author_type}' & proposition_type == '{proposition_type}'"
        )
        y_true_dict[id_tuple] = (
            human_gt_subset.loc[:, ["proposition_id", "verdict"]]
            .set_index("proposition_id")
            .sort_index()
        ).squeeze()

    # # Make Confusion Matrix Figure (Subplot per Model)
    # fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
    #     nrows=3, ncols=3, figsize=(7, 7), layout="tight", sharey=True, sharex=True
    # )
    # # Model Axes Map
    # model_ax_map = {
    #     "Gemma3-12B": ax1,
    #     "Gemma3-27B": ax2,
    #     "Qwen3-32B": ax3,
    #     "Qwen3-30B-A3B-Instruct": ax4,
    #     "Qwen3-30B-A3B-Thinking": ax5,
    #     "R1-8B": ax6,
    #     "R1-70B": ax7,
    #     "Llama-8B": ax8,
    #     "Llama-70B": ax9,
    # }
    # for model in models:
    #     # Get Axes & Data for Model
    #     ax = model_ax_map[model]
    #     y_pred = y_pred_dict[model]
    #     y_true = y_true_dict[model]
    #     # Compute & Display Heatmap of Confusion Matrix
    #     confmat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    #     sns.heatmap(
    #         confmat,
    #         annot=True,
    #         annot_kws={"size": 10},
    #         fmt="d",
    #         vmin=0,  # Minimum colormap value
    #         vmax=500,  # Maximum colormap value (values above this are clipped to this value)
    #         cmap="Blues",
    #         cbar=False,
    #         linewidths=0.5,
    #         linecolor="white",
    #         xticklabels=labels,
    #         yticklabels=labels,
    #         square=True,
    #         ax=ax,
    #     )
    #     model_name = model_display_map[model]
    #     ax.set_title(f"{model_name}", size=9)
    #     ax.set_yticklabels(
    #         labels=["Supported", "Not\nSupported", "Not\nAddressed"], rotation=0, size=9
    #     )
    #     ax.set_xticklabels(
    #         labels=["Supported", "Not\nSupported", "Not\nAddressed"],
    #         rotation=90,
    #         size=9,
    #         ma="right",
    #     )
    #     ax.tick_params(axis="both", which="both", bottom=False, left=False)
    #     fig.supxlabel("VeriFact AI System", size=12)
    #     fig.supylabel("Human Clinician\nGround Truth", size=12)
    #     # Save Figure
    #     save_dir = analysis_dir / "9_confusion_matrices"
    #     save_dir.mkdir(exist_ok=True)
    #     fig.savefig(
    #         save_dir / f"confmat_{author_type}-{proposition_type}-{fact_type}.png",
    #         bbox_inches="tight",
    #         dpi=300,
    #         transparent=True,
    #     )
    #     fig.show()
# %%
# Map for Display Names for Each Model
model_display_map = {
    "Gemma3-12B": "Gemma-3-12B",
    "Gemma3-27B": "Gemma-3-27B",
    "Qwen3-32B": "Qwen-3-32B",
    "Qwen3-30B-A3B-Instruct": "Qwen-3-30B-A3B-Instruct",
    "Qwen3-30B-A3B-Thinking": "Qwen-3-30B-A3B-Thinking",
    "R1-8B": "R1-Distill-Llama-8B",
    "R1-70B": "R1-Distill-Llama-70B",
    "Llama-8B": "Llama-3.1-8B",
    "Llama-70B": "Llama-3.1-70B",
}

for author_type, proposition_type in [("llm", "claim"), ("llm", "sentence"), ("human", "claim")]:
    ### Create Figure with 2 Subfigures (One for EHR Fact Type as Atomic Claim, One for Sentence)
    fig = plt.figure(layout="constrained", figsize=(5, 7))
    subfig1, subfig2 = fig.subfigures(nrows=2, ncols=1, hspace=0.05)
    ## SubFigure 1: EHR Fact Type as Atomic Claim
    ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = subfig1.subplots(
        nrows=3, ncols=3, sharey=True, sharex=True
    )
    # Model Axes Map
    model_ax_map = {
        "Gemma3-12B": ax1,
        "Gemma3-27B": ax2,
        "Qwen3-32B": ax3,
        "Qwen3-30B-A3B-Instruct": ax4,
        "Qwen3-30B-A3B-Thinking": ax5,
        "R1-8B": ax6,
        "R1-70B": ax7,
        "Llama-8B": ax8,
        "Llama-70B": ax9,
    }
    for model in models:
        # Get Axes & Data for Model
        fact_type = "claim"
        id_tuple = (model, author_type, proposition_type, fact_type)
        ax = model_ax_map[model]
        y_pred = y_pred_dict[id_tuple]
        y_true = y_true_dict[id_tuple]
        # Compute & Display Heatmap of Confusion Matrix
        confmat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
        sns.heatmap(
            confmat,
            annot=True,
            annot_kws={"size": 8},
            fmt="d",
            vmin=0,  # Minimum colormap value
            vmax=500,  # Maximum colormap value (values above this are clipped to this value)
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="silver",
            clip_on=False,
            xticklabels=labels,
            yticklabels=labels,
            # square=True,
            ax=ax,
        )
        model_name = model_display_map[model]
        ax.set_title(f"{model_name}", size=8)
        ax.set_yticklabels(labels=["S", "NS", "NA"], rotation=0, size=8)
        ax.set_xticklabels(labels=["S", "NS", "NA"], rotation=0, size=8)
        ax.tick_params(axis="both", which="both", bottom=False, left=False)
        author_type_str = "LLM-generated" if author_type == "llm" else "Human-written"
        proposition_type_str = "Atomic Claim" if proposition_type == "claim" else "Sentence"
        fact_type_str = "Atomic Claim" if fact_type == "claim" else "Sentence"
        subfig1.suptitle(
            f"{author_type_str} Brief Hospital Course evaluated with\n"
            f"{proposition_type_str} Propositions and {fact_type_str} EHR Facts",
            size=11,
        )
        subfig1.supxlabel("VeriFact AI System", size=10)
        subfig1.supylabel("Human Clinician Ground Truth", size=10)
    ## SubFigure 2: EHR Fact Type as Atomic Claim
    ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = subfig2.subplots(
        nrows=3, ncols=3, sharey=True, sharex=True
    )
    # Model Axes Map
    model_ax_map = {
        "Gemma3-12B": ax1,
        "Gemma3-27B": ax2,
        "Qwen3-32B": ax3,
        "Qwen3-30B-A3B-Instruct": ax4,
        "Qwen3-30B-A3B-Thinking": ax5,
        "R1-8B": ax6,
        "R1-70B": ax7,
        "Llama-8B": ax8,
        "Llama-70B": ax9,
    }
    for model in models:
        # Get Axes & Data for Model
        fact_type = "sentence"
        id_tuple = (model, author_type, proposition_type, fact_type)
        ax = model_ax_map[model]
        y_pred = y_pred_dict[id_tuple]
        y_true = y_true_dict[id_tuple]
        # Compute & Display Heatmap of Confusion Matrix
        confmat = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
        sns.heatmap(
            confmat,
            annot=True,
            annot_kws={"size": 8},
            fmt="d",
            vmin=0,  # Minimum colormap value
            vmax=500,  # Maximum colormap value (values above this are clipped to this value)
            cmap="Blues",
            cbar=False,
            linewidths=0.5,
            linecolor="silver",
            clip_on=False,
            xticklabels=labels,
            yticklabels=labels,
            # square=True,
            ax=ax,
        )
        model_name = model_display_map[model]
        ax.set_title(f"{model_name}", size=8)
        ax.set_yticklabels(labels=["S", "NS", "NA"], rotation=0, size=8)
        ax.set_xticklabels(labels=["S", "NS", "NA"], rotation=0, size=8)
        ax.tick_params(axis="both", which="both", bottom=False, left=False)
        author_type_str = "LLM-generated" if author_type == "llm" else "Human-written"
        proposition_type_str = "Atomic Claim" if proposition_type == "claim" else "Sentence"
        fact_type_str = "Atomic Claim" if fact_type == "claim" else "Sentence"
        subfig2.suptitle(
            f"{author_type_str} Brief Hospital Course evaluated with\n"
            f"{proposition_type_str} Propositions and {fact_type_str} EHR Facts",
            size=11,
        )
        subfig2.supxlabel("VeriFact AI System", size=10)
        subfig2.supylabel("Human Clinician Ground Truth", size=10)
    # Save Figure
    save_dir = analysis_dir / "9_confusion_matrices"
    save_dir.mkdir(exist_ok=True)
    fig.savefig(
        save_dir / f"confmat_{author_type}-{proposition_type}.png",
        bbox_inches="tight",
        dpi=300,
        transparent=True,
    )
    fig.show()
# %%
