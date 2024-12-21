# %% [markdown]
# ## Use LLM to Generate Hospital Course section of Discharge Summary
#
# This note-writing process simulates a physician who is reviewing all the notes
# in a hospital admission and summarizing the key points in each note before synthesizing
# it into a larger narrative that spans the entire hospital course.
#
# At a high-level, the following procedure is executed:
# 1. Select only notes in the target hospital admission. Split note text into chunks.
# 2. For each note chunk, generate a 1-4 sentence summary. Then combine these summaries
#    for each note, yielding a 1-4 sentence summary for each note in the hospital admission.
# 3. Sort note summaries in chronological order, then add note metadata context which
#    includes note datetime, note type, and description.
# 4. Split the note summaries into batches. Generate an initial "Brief Hospital course"
#    from the first chronological batch. Use the subsquent batches to performing a rolling
#    update of the Brief Hospital Course.
# 5. If the Brief Hospital Course narrative exceeds a length limit, use the LLM to
#    compress the summary.
#
# The following script places jobs on a queue and executes them in parallel using RQ.
# Each RQ worker will generate a Brief Hospital Course for a single hospital admission,
# following the above procedure which involves multiple LLM invocations.
# %%
import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from llm_writer.hospital_course import HospitalCourseSummarizer
from rag.components import get_llm
from rq import Queue, Retry
from rq.job import Job
from rq_utils import (
    block_and_accumulate_results,
    enqueue_jobs,
    get_queue,
    get_redis,
    shutdown_all_workers,
    start_workers,
)
from utils import (
    get_local_time,
    get_utc_time,
    load_environment,
    load_pandas,
    load_text,
    save_pandas,
    save_text,
)

load_environment()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def summarize_hospital_admission(
    notes_df: pd.DataFrame,
    llm_temperature: float = 0.5,
    llm_top_p: float = 1.0,
    summarizer_parallel_notes: int = 8,
    summarizer_max_chunk_size: int = 5000,
    summarizer_max_bhc_size: int = 1000,
    save_path: os.PathLike[str] | Path | str | None = None,
    show_progress: bool = True,
) -> str:
    """Write a Brief Hospital Course (BHC) section for a patient's discharge summary,
    which summarizes an entire hospital admission for the patient.

    If the Hospital Course has already been written, it will be loaded from the save_path.
    Otherwise, it will be generated using the LLM and saved to the save_path.
    """
    # Check if Hospital Course already exists
    if save_path is None:
        raise ValueError("Must provide save path.")
    save_path = Path(save_path)
    if Path(save_path).exists():
        bhc = load_text(save_path)
        return bhc
    else:
        # Have LLM Write Hospital Course
        loop = asyncio.get_event_loop()
        llm = get_llm(temperature=llm_temperature, top_p=llm_top_p)
        summarizer = HospitalCourseSummarizer.from_defaults(
            llm=llm,
            num_workers=summarizer_parallel_notes,
            max_chunk_size=summarizer_max_chunk_size,
            max_hospital_course_size=summarizer_max_bhc_size,
        )
        bhc = loop.run_until_complete(summarizer.acall(df=notes_df, show_progress=show_progress))
        # Save to Text File
        if save_path:
            save_text(text=bhc, filepath=save_path)
        return bhc


def main(
    dataset_dir: str = typer.Option(default=None, help="Path to the dataset directory"),
    admissions_path: str = typer.Option(
        default=None, help="Path to admissions table which maps SUBJECT_ID to HADM_ID."
    ),
    dc_sum_notes_path: str = typer.Option(
        default=None, help="Path to the discharge summary notes (NOTEEVENTS table)."
    ),
    ehr_notes_path: str = typer.Option(default=None, help="Path to the other EHR notes"),
    save_dir: str = typer.Option(default=None, help="Path to directory to save the output files"),
    table_save_name: str = typer.Option(
        default="llm_bhc_noteevents",
        help="Name of the output file (will be saved as .feather and .csv) in save_dir.",
    ),
    text_save_dir_name: str = typer.Option(
        default="llm_bhc_note_text",
        help="Name of the directory to save a copy of raw text files for "
        "LLM-written Hospital Course. This will be created as a subdirectory in save_dir.",
    ),
    llm_temperature: float = typer.Option(default=0.5, help="Temperature for LLM sampling"),
    llm_top_p: float = typer.Option(default=1.0, help="Top-p sampling threshold for LLM"),
    parallel_bhc: int = typer.Option(
        default=100, help="Number of parallel Brief Hospital Courses (BHC) being written."
    ),
    summarizer_parallel_notes: int = typer.Option(
        default=24, help="Parallel notes being summarized for each BHC generation."
    ),
    summarizer_max_chunk_size: int = typer.Option(
        default=5000, help="Max token chunk size for summarizer"
    ),
    summarizer_max_bhc_size: int = typer.Option(
        default=1000, help="Max token length for output Brief Hospital Course"
    ),
    job_timeout: int = typer.Option(
        default=3600 * 24, help="Max duration for each job in seconds."
    ),
    job_result_ttl: int = typer.Option(
        default=3600 * 72, help="How long result is available for each job in seconds."
    ),
    job_polling_interval: int = typer.Option(
        default=5, help="Frequency of polling for job updates in seconds."
    ),
    queue_name: str = typer.Option(default="BHC-Writer", help="Name of the queue to use."),
) -> None:
    # Data Paths
    if dataset_dir is None:
        dataset_dir = Path(os.environ["DATA_DIR"]) / "dataset"
    if admissions_path is None:
        admissions_path = Path(dataset_dir) / "admissions.feather"
    if ehr_notes_path is None:
        ehr_notes_path = Path(dataset_dir) / "ehr_noteevents.feather"
    if dc_sum_notes_path is None:
        dc_sum_notes_path = Path(dataset_dir) / "dc_noteevents.feather"
    if save_dir is None:
        save_dir = Path(dataset_dir)
    text_save_dir = Path(dataset_dir) / text_save_dir_name
    text_save_dir.mkdir(exist_ok=True, parents=True)

    # Load Data
    admissions_df = load_pandas(admissions_path)
    ehr_notes_df = load_pandas(ehr_notes_path)

    # Sort Subject IDs by number of notes (from most to least)
    ehr_note_counts = ehr_notes_df.groupby("SUBJECT_ID").size().sort_values(ascending=False)
    subject_ids = ehr_note_counts.index.to_list()

    ## Generate "Brief Hospital Course" section for each patient
    # Create Job Definitions
    job_creation_str_timestamp = get_utc_time()
    job_datas = []
    for subject_id in subject_ids:
        # Get Hospital Admission ID & Note ID for Subject
        subj_adm = admissions_df.query(f"SUBJECT_ID == {subject_id}").iloc[0]
        hadm_id = int(subj_adm.HADM_ID)
        row_id = subj_adm.ROW_ID
        # Get all notes for only that hospital admission
        subject_hadm_notes = ehr_notes_df.query("HADM_ID == @subj_adm.HADM_ID")
        # Configure save path
        save_path = text_save_dir / f"{subject_id}_{hadm_id}_{row_id}.txt"

        # Prepare Jobs
        job_data = Queue.prepare_data(
            summarize_hospital_admission,
            kwargs={
                "notes_df": subject_hadm_notes,
                "llm_temperature": llm_temperature,
                "llm_top_p": llm_top_p,
                "summarizer_parallel_notes": summarizer_parallel_notes,
                "summarizer_max_chunk_size": summarizer_max_chunk_size,
                "summarizer_max_bhc_size": summarizer_max_bhc_size,
                "save_path": save_path,
            },
            timeout=job_timeout,
            result_ttl=job_result_ttl,
            job_id=f"BHC-{subject_id}-{hadm_id}_{job_creation_str_timestamp}",
            description=f"Write Hospital Course: {subject_id} - {hadm_id}",
            retry=Retry(max=10, interval=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
        )
        job_datas.append(job_data)

    # Place Jobs on Queue
    redis = get_redis()
    q = get_queue(connection=redis, queue=queue_name)
    jobs = enqueue_jobs(job_datas=job_datas, connection=redis, queue=q)

    # Create Workers to Process Jobs
    start_workers(
        queues=[q],
        num_workers=parallel_bhc,
        name="BHC-Worker",
    )

    # Block Main Thread until Queue is empty and all jobs are completed
    results: list[str] = block_and_accumulate_results(
        jobs=jobs, polling_interval=job_polling_interval, show_progress=True
    )

    # Shut Down Workers
    shutdown_all_workers(connection=redis, queue=q)

    # Save Results to DataFrame using MIMIC-III NOTEEVENTS table format
    dc_sum_notes_df = load_pandas(dc_sum_notes_path)
    save_results_to_df(
        jobs=jobs,
        results=results,
        noteevents_df=dc_sum_notes_df,
        save_dir=save_dir,
        table_save_name=table_save_name,
    )


def save_results_to_df(
    jobs: list[Job],
    results: list[Any],
    noteevents_df: pd.DataFrame,
    save_dir: Path | str,
    table_save_name: str,
) -> None:
    # Get Mapping of Subject IDs to Hospital Course Text
    hc_text = {int(job.id.split("-")[1]): result for job, result in zip(jobs, results, strict=True)}
    # Format as a Pandas Series
    hosp_course_mapping = pd.Series(hc_text, name="TEXT")
    # Create a DataFrame with the LLM-generated "Brief Hospital Course" text for each patient
    # Using the discharge summary notes as the base table
    hosp_course_df = noteevents_df.copy(deep=True).set_index("SUBJECT_ID").drop(columns=["TEXT"])
    hosp_course_df = (
        hosp_course_df.join(hosp_course_mapping)
        .reset_index(drop=False)
        .sort_values(by=["SUBJECT_ID", "CHARTDATE", "CHARTTIME", "STORETIME"], ascending=True)
        .reset_index(drop=True)
    )
    # Rearrange columns to original order
    hosp_course_df = hosp_course_df.loc[
        :,
        [
            "ROW_ID",
            "SUBJECT_ID",
            "HADM_ID",
            "CHARTDATE",
            "CHARTTIME",
            "STORETIME",
            "CATEGORY",
            "DESCRIPTION",
            "CGID",
            "ISERROR",
            "TEXT",
        ],
    ]
    # Save Hospital Course Outputs as MIMIC-III NOTEEVENTS table
    save_path_feather = Path(save_dir) / f"{table_save_name}.feather"
    save_pandas(hosp_course_df, save_path_feather)

    utc_timestamp = get_utc_time(output_format="str")
    local_timestamp = get_local_time(output_format="str")
    logger.info(f"Completed Job at: {utc_timestamp} UTC Time ({local_timestamp} Local Time)")


if __name__ == "__main__":
    typer.run(main)

# %%
