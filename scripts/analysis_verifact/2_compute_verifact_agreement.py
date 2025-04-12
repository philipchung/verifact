# %% [markdown]
# ## Agreement for each VeriFact AI System Variation vs. Ground Truth Human Clinician Label
#
# Percent Agreement & Gwet's AC1 is computed for both original and binarized labels
# (treating Not Addressed labels as Not Supported). Then the results are displayed in table format.
# %%
import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from irr_metrics import MetricBunch, coerce_types
from irr_metrics.binarize import binarize_verdicts_in_df
from utils import (
    get_function_status_string,
    get_local_time,
    get_utc_time,
    load_environment,
    load_pandas,
    send_notification,
)


def main(
    metric: Annotated[
        str,
        typer.Option(
            help="Interrater agreement metric to compute ('percent_agreement' or 'gwet')."
        ),
    ] = "percent_agreement",
    models: Annotated[
        list[str],
        typer.Option(
            help="List of VeriFact AI models to compute agreement metrics for. "
            "Choose from 'Llama-8B', 'Llama-70B', 'R1-8B', 'R1-70B'. Default is all models.",
        ),
    ] = ["Llama-8B", "Llama-70B", "R1-8B", "R1-70B"],
    author_type: Annotated[list[str], typer.Option(help="List of author types to include.")] = [
        "llm",
        "human",
    ],
    proposition_type: Annotated[
        list[str], typer.Option(help="List of proposition types to include.")
    ] = ["claim", "sentence"],
    binarize_labels: Annotated[
        bool,
        typer.Option(
            help="Binarize labels to combine 'Not Addressed' and 'Not Supported'. "
            "as 'Not Addressed or Supported'. Default is False."
        ),
    ] = False,
    workers: Annotated[
        int,
        typer.Option(
            help="Number of workers computing bootstrap iterations for each comparison. "
            "Default is 8."
        ),
    ] = 20,
    num_parallel_raters: Annotated[
        int,
        typer.Option(
            help="Number of concurrent comparisons between VeriFact Raters vs. Ground Truth. "
            "Default is 8."
        ),
    ] = 4,
    bootstrap_iterations: Annotated[
        int,
        typer.Option(help="Number of bootstrap iterations for CI estimation. Default is 1000."),
    ] = 1000,
    save_dir: Annotated[
        str,
        typer.Option(help="Directory to save computed metrics. "),
    ] = None,
    force_recompute: Annotated[
        bool,
        typer.Option(
            help="Force recompute of inter-rater agreement metrics even if already computed. "
            "If forced, will overwrite saved metrics. Default is False."
        ),
    ] = False,
    niceness: Annotated[
        int,
        typer.Option(
            help="Nice value for the process. Integer value between -20 to 19 with "
            "lower values corresponding to higher job priority. Default is 10.",
        ),
    ] = 10,
    run_name: Annotated[
        str,
        typer.Option(help="Name of the run. Default is 'VeriFact Agreement'."),
    ] = "VeriFact Agreement",
) -> None:
    load_environment()
    start_utc_time = get_utc_time(output_format="str")
    start_local_time = get_local_time(output_format="str")

    if niceness not in list(range(-20, 20)):
        raise ValueError("Invalid niceness value. Choose an integer between -20 and 19.")
    os.nice(niceness)

    if metric not in ["percent_agreement", "gwet"]:
        raise ValueError("Invalid metric. Choose from 'percent_agreement' or 'gwet'")

    for model in models:
        if model not in [
            "Llama-8B",
            "Llama-70B",
            "R1-8B",
            "R1-70B",
        ]:
            raise ValueError(
                "Invalid model. Choose from 'Llama-8B', 'Llama-70B', 'R1-8B', 'R1-70B'."
            )
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
    # Load VeriFact Labels for Different Models
    model_data_dict: dict[str, pd.DataFrame] = {}
    for model in models:
        model_dir = model_dir_map[model]
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
        rater_name=ai_verdicts.apply(
            lambda row: f"model={row.model},{row.rater_name}", axis="columns"
        )
    )

    # Subselect AI Verdicts for which we want to compute metrics
    if author_type:
        author_type = [author_type] if isinstance(author_type, str) else author_type
        ai_verdicts = ai_verdicts.query("author_type in @author_type")
    if proposition_type:
        proposition_type = (
            [proposition_type] if isinstance(proposition_type, str) else proposition_type
        )
        ai_verdicts = ai_verdicts.query("proposition_type in @proposition_type")

    # Optionally binarize labels to combine 'Not Addressed' and 'Not Supported'
    # as 'Not Addressed or Supported'
    if binarize_labels:
        human_gt = binarize_verdicts_in_df(human_gt, verdict_name="verdict")
        ai_verdicts = binarize_verdicts_in_df(ai_verdicts, verdict_name="verdict")

    # Compute Metric Bunch, reloading computed metrics from cache if available
    name = f"ai_rater_{metric}_ci"
    if binarize_labels:
        name = f"{name}_binarized"
    if save_dir:
        save_dir = Path(save_dir)
    else:
        save_dir = (
            Path(os.environ["PROJECT_DIR"])
            / "scripts"
            / "analysis_verifact"
            / "2_compute_verifact_agreement"
        )
    MetricBunch.from_defaults(
        name=name,
        rater_verdicts=ai_verdicts,
        ground_truth=human_gt,
        metric=metric,
        rater_type="ai",
        rater_id_col="stratum",
        stratify_cols=[
            "model",
            "author_type",
            "proposition_type",
            "fact_type",
            "top_n",
            "retrieval_method",
            "reference_format",
            "reference_only_admission",
        ],
        workers=workers,
        num_parallel_raters=num_parallel_raters,
        bootstrap_iterations=bootstrap_iterations,
        cache_dir=save_dir,
        force_recompute=force_recompute,
    )

    # Create Notification Message
    msg = get_function_status_string(
        filename=__file__, start_utc_time=start_utc_time, start_local_time=start_local_time
    )
    send_notification(
        title=f"Completed Run: {run_name}", message=msg, url=os.environ["NOTIFY_WEBHOOK_URL"]
    )


if __name__ == "__main__":
    typer.run(main)
