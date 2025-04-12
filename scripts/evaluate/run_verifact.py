# %% [markdown]
# ## Run Verifact Evaluation Pipeline on Dataset of Propositions
# %%
import logging
import os
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from llm_judge import JudgeCohort, JudgeConfig, JudgeSingleSubject, ScoreReport
from rag import (
    ABSOLUTE_TIME,
    CLAIM_NODE,
    DENSE,
    HUMAN_AUTHOR,
    HYBRID,
    LLM_AUTHOR,
    RELATIVE_TIME,
    RERANK,
    SCORE,
    SENTENCE_NODE,
    SPARSE,
)
from rq import Queue, Retry
from rq_utils import (
    block_and_accumulate_results,
    enqueue_jobs,
    get_queue,
    get_redis,
    shutdown_all_workers,
    start_workers,
)
from tqdm.auto import tqdm
from utils import (
    LazyFileLogger,
    get_function_status_string,
    get_local_time,
    get_utc_time,
    load_pandas,
    save_pandas,
    send_notification,
)


def evaluate_judge(
    judge: JudgeSingleSubject,
    collection_name: str,
    query_timeout: float = 1200.0,
    save_results: bool = True,
    skip_existing: bool = True,
    job_log_file: str | Path | None = None,
) -> ScoreReport:
    from utils import load_environment

    load_environment()
    logger = LazyFileLogger(__name__, log_file=job_log_file, level=logging.DEBUG)
    judge_config_str = judge.config.to_config_str()

    try:
        # Judge exists on disk.  Load & skip re-evaluation.
        filepath = judge.get_judge_save_filepath()
        if skip_existing and filepath.exists():
            logger.info(f"Skipping existing judge: {judge_config_str}. File: {filepath}")
            judge = JudgeSingleSubject.load(filepath=filepath)
            score_report = judge.score_report
            return score_report
        # Judge does not exist on disk.  Evaluate & save to disk.
        else:
            from rag.components import get_vectorstore

            logger.info(f"Evaluating judge: {judge_config_str}")
            # Get Vectorstore
            vector_store = get_vectorstore(collection_name=collection_name, timeout=query_timeout)
            # Evaluate
            judge.setup_logger_and_vectorstore(vectorstore=vector_store)
            score_report = judge.evaluate(show_progress=False, vector_store=vector_store)
            # Persist Judge Object and Evaluation Results to Disk
            if save_results:
                judge.save(omit_input_data=True)
                judge.save_score_report()
            return score_report
    except Exception:
        logger.critical(
            f"Error occurred for judge: {judge_config_str}",
            exc_info=True,
            stack_info=True,
            stacklevel=5,
        )


def gather_score_reports(score_reports_dir: str = "") -> pd.DataFrame:
    """Load all ScoreReports from disk and transform them into a DataFrame.
    A ScoreReport is the aggregate bundle of ratings for all propositions from a single BHC
    for a single Subject ID made by a single rater.

    Args:
        score_reports_dir (str, optional): Directory to score reports.

    Returns:
        pd.DataFrame: DataFrame with each row containing columns from
            JudgeConfig and ScoreReport.
    """
    from utils import load_environment

    load_environment()

    # Get All ScoreReport Paths
    filepaths = list(score_reports_dir.glob("*/*.feather"))

    # Unpack ScoreReports and join with corresponding JudgeConfig
    score_reports = []
    for filepath in tqdm(filepaths, total=len(filepaths), desc="Aggregate Score Reports"):
        filename = filepath.stem
        rater_name = ",".join(filename.split(",")[3:])
        score_reports.append(
            {"rater_name": rater_name, "filename": filename}
            | JudgeConfig.from_config_str(filepath.stem).model_dump()
            | ScoreReport.load(filepath).model_dump()
        )
    score_report_df = pd.DataFrame(score_reports).drop(
        columns=[
            "default_temperature",
            "retry_temperature_increment",
            "retry_temperature_max",
            "num_workers",
            "num_invalid_output_retries",
            "system_prompt_template",
            "judge_config",
        ]
    )
    return score_report_df


def score_report_to_verdicts(score_report_df: pd.DataFrame) -> pd.DataFrame:
    """Transforms ScoreReport DataFrame into a DataFrame of Proposition Verdicts.

    Args:
        score_report_df (pd.DataFrame): DataFrame with each row corresponding to a ScoreReport
            (one rater rating all the propositions from a single BHC from a Subject ID).

    Returns:
        pd.DataFrame: DataFrame containing Proposition Verdicts.
    """
    # Explode ScoreReport dataframe so that each row is a single Proposition Verdict
    verdicts_df = score_report_df.loc[
        :,
        [
            "rater_name",
            "filename",
            "subject_id",
            "author_type",
            "proposition_type",
            "fact_type",
            "retrieval_method",
            "top_n",
            "reference_format",
            "reference_only_admission",
            "deduplicate_text",
            "verdicts",
        ],
    ].explode(column=["verdicts"], ignore_index=True)
    # Convert Proposition Verdict JSON into DataFrame
    new_cols = pd.json_normalize(verdicts_df.verdicts).drop(columns="metadata_")
    # Combine Proposition Verdict DataFrame with Rater Metadata from ScoreReport DataFrame
    verdicts_df = (
        pd.concat([verdicts_df.drop(columns="verdicts"), new_cols], axis=1)
        .rename(columns={"node_id": "proposition_id"})
        .set_index("proposition_id")
    )
    return verdicts_df


def main(
    run_name: Annotated[str, typer.Option(help="Name of the run.")] = "run_verifact",
    dataset_dir: Annotated[
        str, typer.Option(help="Directory containing VeriFact-BHC dataset.")
    ] = None,
    propositions: Annotated[str, typer.Option(help="File containing propositions data.")] = None,
    admissions: Annotated[str, typer.Option(help="File containing admissions data.")] = None,
    output_dir: Annotated[str, typer.Option(help="Directory to output results.")] = None,
    output_subdir: Annotated[
        str, typer.Option(help="Subdirectory to output results. Defaults to results.")
    ] = "results",
    is_reasoning_model: Annotated[
        bool, typer.Option(help="Set to True if LLM is a reasoning model.")
    ] = False,
    subject_id: Annotated[
        list[int], typer.Option(help="Subject ID to evaluate.  If not provided, evaluate all.")
    ] = [],
    author_type: Annotated[
        list[str], typer.Option(help=f"Input Text Author type: {LLM_AUTHOR}, {HUMAN_AUTHOR}.")
    ] = [LLM_AUTHOR, HUMAN_AUTHOR],
    proposition_type: Annotated[
        list[str],
        typer.Option(help=f"Proposition type (node kind): {CLAIM_NODE}, {SENTENCE_NODE}."),
    ] = [CLAIM_NODE, SENTENCE_NODE],
    fact_type: Annotated[
        list[str],
        typer.Option(
            help=f"EHR Fact type (node kind): {CLAIM_NODE}, {SENTENCE_NODE}."
            "If `None`, will default to using the same as `proposition_type`."
        ),
    ] = None,
    reference_collection_name: Annotated[
        str, typer.Option(help="Vector database collection name for EHR Facts.")
    ] = "",
    retrieval_method: Annotated[
        list[str],
        typer.Option(help=f"Retrieval method: {DENSE}, {SPARSE}, {HYBRID}, {RERANK}."),
    ] = [DENSE, SPARSE, HYBRID, RERANK],
    top_n: Annotated[list[int], typer.Option(help="Number of top results to consider.")] = [
        5,
        10,
        25,
        50,
        75,
        100,
        125,
        150,
    ],
    reference_format: Annotated[
        list[str],
        typer.Option(
            help=f"Reference Context Format for LLM-as-a-Judge "
            f"{SCORE}, {ABSOLUTE_TIME}, {RELATIVE_TIME}."
        ),
    ] = [SCORE, ABSOLUTE_TIME, RELATIVE_TIME],
    reference_only_admission: Annotated[
        list[bool],
        typer.Option(
            help="Limit retrieved information to current hospital admission (vs. entire EHR). "
            "{True, False}."
        ),
    ] = [True, False],
    job_timeout: Annotated[int, typer.Option(help="Timeout for each job in seconds.")] = 3600 * 24,
    job_result_ttl: Annotated[
        int, typer.Option(help="Time-to-live for the job result in seconds.")
    ] = 3600 * 72,
    job_polling_interval: Annotated[
        int, typer.Option(help="Frequency of polling for job updates in seconds.")
    ] = 5,
    queue_name: Annotated[str, typer.Option(help="Name of the queue to use.")] = "run_verifact",
    parallel_judges: Annotated[
        int, typer.Option(help="Number of parallel judge evaluations.")
    ] = 20,
    workers_in_judge: Annotated[
        int, typer.Option(help="Number of parallel workers in each judge.")
    ] = 10,
    query_timeout: Annotated[
        float, typer.Option(help="Timeout for each reference context query in seconds.")
    ] = 600.0,
    save_results: Annotated[
        bool,
        typer.Option(
            help="If True, save judge evaluation results to disk.  If False, do not save results."
        ),
    ] = True,
    skip_existing: Annotated[
        bool,
        typer.Option(
            help="If existing judge saved to disk, skip re-evaluation and use saved results.",
        ),
    ] = True,
) -> None:
    from utils import load_environment

    load_environment()
    start_utc_time = get_utc_time(output_format="str")
    start_local_time = get_local_time(output_format="str")

    # Load Propositions & Admissions
    dataset_dir = Path(dataset_dir or os.environ["VERIFACTBHC_DATASET_DIR"])
    propositions_filepath = (
        Path(propositions) if propositions else dataset_dir / "propositions" / "propositions.csv.gz"
    )
    admissions_filepath = (
        Path(admissions) if admissions else dataset_dir / "reference_ehr" / "admissions.csv.gz"
    )
    try:
        propositions_df = load_pandas(propositions_filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Propositions file not found: {propositions_filepath}")
    try:
        admissions_df = load_pandas(admissions_filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Admissions file not found: {admissions_filepath}")

    # Define Output & Log Directories
    output_dir = Path(output_dir or os.environ["VERIFACT_RESULTS_DIR"]) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    utc_time = get_utc_time()
    log_dir = output_dir / "logs" / f"{utc_time}"
    log_dir.mkdir(parents=True, exist_ok=True)
    job_log_file = log_dir / "_jobs.log"

    # Define Cohort of Judges w/ Different Hyperparameters
    judge_cohort = JudgeCohort.from_defaults(
        proposition_df=propositions_df,
        admission_df=admissions_df,
        output_dir=output_dir,
        log_dir=str(log_dir),
        is_reasoning_model=is_reasoning_model,
        subject_id=subject_id,
        author_type=author_type,
        proposition_type=proposition_type,
        fact_type=fact_type,
        retrieval_method=retrieval_method,
        reference_format=reference_format,
        reference_only_admission=reference_only_admission,
        top_n=top_n,
        num_parallel_workers_in_judge=workers_in_judge,
    )
    # Create Queue
    redis = get_redis()
    q = get_queue(connection=redis, queue=queue_name)

    # Create Workers to Process Jobs
    start_workers(queues=[q], num_workers=parallel_judges, name=f"{queue_name}-worker")

    # Create Judge Evaluation Jobs & Place on Queue
    job_creation_str_timestamp = get_utc_time()
    jobs = []
    for judge in judge_cohort.judge_generator():
        # Use Judge Configuration to Identify Judge
        judge_config_str = judge.config.to_config_str()
        judge.remove_logger_and_vectorstore()  # Logger cannot be pickled by RQ
        # Create Job Definition
        job_data = Queue.prepare_data(
            func=evaluate_judge,
            kwargs={
                "judge": judge,
                "collection_name": reference_collection_name
                or os.environ["MIMIC3_EHR_COLLECTION_NAME"],
                "query_timeout": query_timeout,
                "save_results": save_results,
                "skip_existing": skip_existing,
                "job_log_file": job_log_file,
            },
            timeout=job_timeout,
            result_ttl=job_result_ttl,
            job_id=f"judge-{judge_config_str}-{job_creation_str_timestamp}",
            description=f"Evaluate Judge: {judge_config_str}",
            retry=Retry(max=10, interval=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
        )
        job = enqueue_jobs(job_datas=[job_data], connection=redis, queue=q)[0]
        jobs.append(job)

    # Block Main Thread until Queue is empty and all jobs are completed
    block_and_accumulate_results(
        jobs=jobs, polling_interval=job_polling_interval, show_progress=True
    )

    # Shut Down Workers
    shutdown_all_workers(connection=redis, queue=q)

    # Gather All Score Reports into DataFrame
    sr_save_dir = judge_cohort.score_report_save_dir
    score_report_df = gather_score_reports(score_reports_dir=sr_save_dir)
    save_pandas(df=score_report_df, filepath=sr_save_dir / "score_reports.feather")
    # Transform Score Report DataFrame into DataFrame of Proposition Verdicts
    verdicts_df = score_report_to_verdicts(score_report_df)
    save_pandas(df=verdicts_df, filepath=sr_save_dir / "verdicts.feather")

    # Log Completion & Notify
    msg = get_function_status_string(
        filename=__file__, start_utc_time=start_utc_time, start_local_time=start_local_time
    )
    logger = LazyFileLogger(__name__, log_file=job_log_file, level=logging.DEBUG)
    logger.info(msg)
    send_notification(
        title=f"Completed Run: {run_name}", message=msg, url=os.environ["NOTIFY_WEBHOOK_URL"]
    )


if __name__ == "__main__":
    typer.run(main)
