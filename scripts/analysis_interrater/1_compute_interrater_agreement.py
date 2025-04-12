# %% [markdown]
# ## Human Clinician Inter-rater Agreement Analysis
#
# This is a script to compute inter-rater agreement metrics for human clinician verdicts on:
# 1. all propositions
# 2. propositions with at least one "Supported" verdict
# 3. propositions with at least one "Not Supported" verdict
# 4. propositions with at least one "Not Addressed" verdict.
#
# The metrics computed include:
# 1. Percent Agreement
# 2. Gwet's AC1
# %%
import os
from pathlib import Path
from typing import Annotated

import typer
from irr_metrics import (
    BINARIZED_VERDICT_LABELS,
    NOT_ADDRESSED,
    NOT_SUPPORTED,
    SUPPORTED,
    VERDICT_LABELS,
    MetricBunch,
    binarize_verdicts_in_df,
    coerce_types,
    melt_wide_to_long,
)
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
    proposition_strata: Annotated[
        str,
        typer.Option(
            help="Proposition strata for which to compute inter-rater agreement metrics "
            "('all', 'supported', 'not_supported', 'not_addressed', "
            "'all_binarized', 'supported_binarized', 'not_supported_or_addressed'). "
            "The 'all_binarized', 'supported_binarized' and 'not_supported_or_addressed' first "
            "applies binarization to the labels to combine 'Not Supported' and 'Not Addressed' "
            "before computing metrics. Default is 'all'."
        ),
    ] = "all",
    workers: Annotated[
        int,
        typer.Option(
            help="Number of workers computing bootstrap iterations for each comparison. "
            "Default is 8."
        ),
    ] = 8,
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
        typer.Option(help="Name of the run. Default is 'Human Clinician Interrater Agreement'."),
    ] = "Human Clinician Interrater Agreement",
) -> None:
    """Compute inter-rater agreement metrics for human clinician verdicts."""

    load_environment()
    start_utc_time = get_utc_time(output_format="str")
    start_local_time = get_local_time(output_format="str")

    if niceness not in list(range(-20, 20)):
        raise ValueError("Invalid niceness value. Choose an integer between -20 and 19.")
    os.nice(niceness)

    if metric not in ["percent_agreement", "gwet"]:
        raise ValueError("Invalid metric. Choose from 'percent_agreement' or 'gwet'")

    # Annotated Dataset Directories
    propositions_dir = Path(os.environ["VERIFACTBHC_PROPOSITIONS_DIR"])
    save_dir = Path.cwd() / "1_compute_interrater_agreement"
    save_dir.mkdir(exist_ok=True, parents=True)
    # Load Human Clinician Verdict Labels (One Row Per Proposition)
    human_verdicts = load_pandas(propositions_dir / "human_verdicts.csv.gz")
    human_verdicts = coerce_types(human_verdicts)

    match proposition_strata:
        case "all":
            ## Compute Inter-rater Agreement Metrics for All Propositions]
            all_propositions = melt_wide_to_long(human_verdicts)
            rater_verdicts = all_propositions
            name = f"human_interrater_{metric}_ci_all"

        case "supported":
            ## Compute Inter-rater Agreement Metrics for Propositions
            # with at least one "Supported" Verdict in First Round of annotation
            supported = human_verdicts.query(
                f"verdict1 == '{SUPPORTED}' or "
                f"verdict2 == '{SUPPORTED}' or "
                f"verdict3 == '{SUPPORTED}'"
            )
            rater_verdicts = melt_wide_to_long(supported)
            verdict_labels = VERDICT_LABELS
            name = f"human_interrater_{metric}_ci_supported"

        case "not_supported":
            ## Compute Inter-rater Agreement Metrics for Propositions
            # with at least one "Not Supported" Verdict in First Round of annotation
            not_supported = human_verdicts.query(
                f"verdict1 == '{NOT_SUPPORTED}' or "
                f"verdict2 == '{NOT_SUPPORTED}' or "
                f"verdict3 == '{NOT_SUPPORTED}'"
            )
            rater_verdicts = melt_wide_to_long(not_supported)
            verdict_labels = VERDICT_LABELS
            name = f"human_interrater_{metric}_ci_not_supported"

        case "not_addressed":
            ## Compute Inter-rater Agreement Metrics for Propositions
            # with at least one "Not Addressed" Verdict in First Round of annotation
            not_addressed = human_verdicts.query(
                f"verdict1 == '{NOT_ADDRESSED}' or "
                f"verdict2 == '{NOT_ADDRESSED}' or "
                f"verdict3 == '{NOT_ADDRESSED}'"
            )
            rater_verdicts = melt_wide_to_long(not_addressed)
            verdict_labels = VERDICT_LABELS
            name = f"human_interrater_{metric}_ci_not_addressed"

        case "all_binarized":
            ## Compute Inter-rater Agreement Metrics for Binarized Propositions
            binarized_human_verdicts = human_verdicts.copy()
            for col in ["verdict1", "verdict2", "verdict3"]:
                binarized_human_verdicts = binarize_verdicts_in_df(
                    binarized_human_verdicts, verdict_name=col
                )
            rater_verdicts = melt_wide_to_long(binarized_human_verdicts)
            verdict_labels = BINARIZED_VERDICT_LABELS
            name = f"human_interrater_{metric}_ci_all_binarized"

        case "supported_binarized":
            ## Compute Inter-rater Agreement Metrics for Binarized Propositions
            # with at least one "Supported" Verdict in First Round of annotation
            binarized_human_verdicts = human_verdicts.copy()
            for col in ["verdict1", "verdict2", "verdict3"]:
                binarized_human_verdicts = binarize_verdicts_in_df(
                    binarized_human_verdicts, verdict_name=col
                )
            binarized_supported = binarized_human_verdicts.query(
                f"verdict1 == '{SUPPORTED}' or "
                f"verdict2 == '{SUPPORTED}' or "
                f"verdict3 == '{SUPPORTED}'"
            )
            rater_verdicts = melt_wide_to_long(binarized_supported)
            verdict_labels = BINARIZED_VERDICT_LABELS
            name = f"human_interrater_{metric}_ci_supported_binarized"

        case "not_supported_or_addressed":
            ## Compute Inter-rater Agreement Metrics for Binarized Propositions with at least one
            # "Not Supported or Addressed" Verdict in First Round of annotation
            binarized_human_verdicts = human_verdicts.copy()
            for col in ["verdict1", "verdict2", "verdict3"]:
                binarized_human_verdicts = binarize_verdicts_in_df(
                    binarized_human_verdicts, verdict_name=col
                )
            not_supported_or_addressed = human_verdicts.query(
                f"verdict1 == '{NOT_SUPPORTED}' or "
                f"verdict1 == '{NOT_ADDRESSED}' or "
                f"verdict2 == '{NOT_SUPPORTED}' or "
                f"verdict2 == '{NOT_ADDRESSED}' or "
                f"verdict3 == '{NOT_SUPPORTED}' or "
                f"verdict3 == '{NOT_ADDRESSED}'"
            )
            rater_verdicts = melt_wide_to_long(not_supported_or_addressed)
            verdict_labels = BINARIZED_VERDICT_LABELS
            name = f"human_interrater_{metric}_ci_not_supported_or_addressed"

        case _:
            raise ValueError(
                "Invalid proposition strata. "
                "Choose from 'all', 'supported', 'not_supported', 'not_addressed', "
                "'supported_binarized', 'not_supported_or_addressed'. "
            )

    # Compute Metric Bunch, reloading computed metrics from cache if available
    MetricBunch.from_defaults(
        name=name,
        rater_verdicts=rater_verdicts,
        ground_truth=None,
        verdict_labels=verdict_labels,
        metric=metric,
        rater_type="human",
        rater_id_col="rater_name",
        stratify_cols=["author_type", "proposition_type"],
        workers=workers,
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
