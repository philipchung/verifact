import itertools
import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Self

import pandas as pd
from llama_index.core.bridge.pydantic import BaseModel, ConfigDict, Field
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
from rag.components import get_vectorstore
from tqdm.auto import tqdm
from utils import LazyFileLogger, get_utc_time

from llm_judge.judge_subject import JudgeSingleSubject
from llm_judge.score_report import ScoreReport


class JudgeCohort(BaseModel):
    """Factory object to create a cohort of judges with different hyperparameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Input Data
    proposition_df: pd.DataFrame | None = Field(
        default=None, description="Dataframe of propositions for the patients."
    )
    admission_df: pd.DataFrame | None = Field(
        default=None, description="Dataframe of admissions data for the patients."
    )
    # Save Directories
    output_dir: str | Path | None = Field(
        default=None,
        description="Directory to save the Judge and Score Report evaluation results",
    )
    judge_save_dir: Path | None = Field(default=None, description="Directory to save the judges.")
    score_report_save_dir: Path | None = Field(
        default=None, description="Directory to save the score reports."
    )
    # Logger
    log_dir: str | Path | None = Field(default=None, description="Directory to save logs.")
    logger: logging.Logger | None = Field(
        default=None, description="Logger object.", exclude=True, repr=False
    )
    # R1-style reasoning model hosted by vLLM?
    is_reasoning_model: bool = Field(
        default=False, description="Whether the model is a R1-style reasoning model."
    )
    # Grid Search Hyperparameters to Iterate Through
    subject_id: list[int] | None = Field(
        default=None, description="List of Subject IDs for the patients."
    )
    author_type: list[str] | None = Field(
        default=None, description=f"Input Text Author type: {LLM_AUTHOR}, {HUMAN_AUTHOR}."
    )
    proposition_type: list[str] | None = Field(
        default=None,
        description=f"Proposition Type: {CLAIM_NODE}, {SENTENCE_NODE}.",
    )
    fact_type: list[str] | None = Field(
        default=None,
        description=f"EHR Fact type (node kind): {CLAIM_NODE}, {SENTENCE_NODE}."
        "If `None`, will default to using the same as `proposition_type`.",
    )
    retrieval_method: list[str] | None = Field(
        default=None,
        description=f"Retrieval method: {DENSE}, {SPARSE}, {HYBRID}, {RERANK}.",
    )
    top_n: list[int] | None = Field(default=None, description="Top N results to evaluate.")
    reference_format: list[str] | None = Field(
        default=None,
        description=f"Format for retrieved reference context: "
        f"{SCORE}, {ABSOLUTE_TIME}, {RELATIVE_TIME}.",
    )
    reference_only_admission: list[bool] | None = Field(
        default=None, description="Whether to use only admissions for reference context."
    )
    deduplicate_text: list[bool] | None = Field(
        default=None, description="Whether to deduplicate text in reference context."
    )
    # Judge Evaluation Settings
    num_parallel_workers_in_judge: int = Field(
        default=32, description="Number of parallel workers in each judge."
    )

    @classmethod
    def class_name(cls) -> str:
        return "JudgeCohort"

    @classmethod
    def from_defaults(
        cls,
        proposition_df: pd.DataFrame,
        admission_df: pd.DataFrame,
        output_dir: str | None = None,
        log_dir: str | None = None,
        log_level: str = logging.DEBUG,
        is_reasoning_model: bool = False,
        subject_id: Sequence[int] | int | None = None,
        author_type: Sequence[str] | str = [LLM_AUTHOR, HUMAN_AUTHOR],
        proposition_type: Sequence[str] | str = [CLAIM_NODE, SENTENCE_NODE],
        fact_type: Sequence[str] | str | None = None,
        retrieval_method: Sequence[str] | str = ["dense", "hybrid", "rerank"],
        top_n: Sequence[int] | int = [10, 25, 50, 100],
        reference_format: Sequence[str] | str = ["score", "absolute_time", "relative_time"],
        reference_only_admission: Sequence[bool] | bool = [True, False],
        deduplicate_text: Sequence[bool] | bool = True,
        num_parallel_workers_in_judge: int = 32,
    ) -> Self:
        """Initialize CohortJudge factory."""
        from utils import load_environment

        load_environment()

        # Ensure hyperparameters are lists
        if isinstance(subject_id, int):
            subject_id = [subject_id]
        elif not subject_id:
            if "subject_id" in proposition_df.columns:
                subject_id = proposition_df.subject_id.unique()
            elif "SUBJECT_ID" in admission_df.columns:
                subject_id = proposition_df.SUBJECT_ID.unique()
            else:
                raise ValueError("Subject ID must be specified in JudgeCohort.")
        if isinstance(author_type, str):
            author_type = [author_type]
        if isinstance(proposition_type, str):
            proposition_type = [proposition_type]
        if isinstance(retrieval_method, str):
            retrieval_method = [retrieval_method]
        if isinstance(top_n, int):
            top_n = [top_n]
        if isinstance(reference_format, str):
            reference_format = [reference_format]
        if isinstance(reference_only_admission, bool):
            reference_only_admission = [reference_only_admission]
        if isinstance(deduplicate_text, bool):
            deduplicate_text = [deduplicate_text]

        # Set Output Save Paths
        if output_dir is None:
            raise ValueError("Output directory must be specified in JudgeCohort.")
        judge_save_dir = output_dir / "judges"
        judge_save_dir.mkdir(parents=True, exist_ok=True)
        sr_save_dir = output_dir / "score_reports"
        sr_save_dir.mkdir(parents=True, exist_ok=True)
        # Configure Logger
        log_dir = Path(log_dir) if log_dir else None
        if log_dir is None:
            utc_time = get_utc_time()
            log_dir = Path(output_dir) / "logs" / f"{utc_time}"
        log_filepath = log_dir / "_judge_cohort.log"
        logger = LazyFileLogger(name=__name__, level=log_level, log_file=log_filepath)

        return cls(
            is_reasoning_model=is_reasoning_model,
            subject_id=subject_id,
            author_type=author_type,
            proposition_type=proposition_type,
            fact_type=fact_type,
            retrieval_method=retrieval_method,
            top_n=top_n,
            reference_format=reference_format,
            reference_only_admission=reference_only_admission,
            deduplicate_text=deduplicate_text,
            proposition_df=proposition_df,
            admission_df=admission_df,
            output_dir=str(output_dir),
            judge_save_dir=judge_save_dir,
            score_report_save_dir=sr_save_dir,
            log_dir=str(log_dir),
            logger=logger,
            num_parallel_workers_in_judge=num_parallel_workers_in_judge,
        )

    def setup_logger(
        self, level: str = logging.DEBUG, log_file: str | Path | None = None
    ) -> LazyFileLogger:
        log_file = log_file or self.log_filepath
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        self.logger = LazyFileLogger(name=__name__, level=level, log_file=log_file)
        return self.logger

    def judge_generator(self, subject_ids: list[int] | None = None) -> Iterator[JudgeSingleSubject]:
        """Generator function yielding a single judge at a time with different hyperparameters."""
        subject_ids = subject_ids or self.subject_id
        # Create judge cohort for each Subject ID
        for subject_id in subject_ids:
            # Select dataframes for input text kind & node kind
            for author_type, proposition_type in itertools.product(
                self.author_type, self.proposition_type
            ):
                df = self.proposition_df.query(
                    f"subject_id == {subject_id} & "
                    f"author_type == '{author_type}' & "
                    f"proposition_type == '{proposition_type}'"
                )
                # Combination of retrieval & reference formatting hyperparameters
                for (
                    top_n,
                    retrieval_method,
                    reference_format,
                    reference_only_admission,
                    deduplicate_text,
                ) in itertools.product(
                    self.top_n,
                    self.retrieval_method,
                    self.reference_format,
                    self.reference_only_admission,
                    self.deduplicate_text,
                ):
                    # If fact_type is not specified, use proposition_type
                    if not self.fact_type:
                        self.fact_type = [proposition_type]
                    for fact_type in self.fact_type:
                        judge = JudgeSingleSubject.from_defaults(
                            proposition_df=df,
                            admission_df=self.admission_df,
                            subject_id=subject_id,
                            author_type=author_type,
                            proposition_type=proposition_type,
                            fact_type=fact_type,
                            retrieval_method=retrieval_method,
                            top_n=top_n,
                            reference_format=reference_format,
                            reference_only_admission=reference_only_admission,
                            deduplicate_text=deduplicate_text,
                            is_reasoning_model=self.is_reasoning_model,
                            judge_save_dir=self.judge_save_dir,
                            score_report_save_dir=self.score_report_save_dir,
                            log_dir=self.log_dir,
                            num_judge_workers=self.num_parallel_workers_in_judge,
                        )
                        yield judge

    def create(
        self, subject_ids: list[int] | None = None, show_progress: bool = True
    ) -> list[JudgeSingleSubject]:
        """Create a cohort of judges with different hyperparameters."""
        subject_ids = self.subject_ids if subject_ids is None else subject_ids
        judges = []
        for judge in (
            pbar := tqdm(
                self.judge_generator(subject_ids=subject_ids),
                desc="Creating Judges",
                disable=not show_progress,
            )
        ):
            judge_conf_str = judge.config.to_config_str()
            pbar.set_description(desc=f"Created Judge: {judge_conf_str}")
            judges.append(judge)
        return judges

    def evaluate(
        self,
        judges: list[JudgeSingleSubject] | None = None,
        collection_name: str | None = None,
        timeout: float = 60.0,
        show_progress: bool = True,
        save_results: bool = True,
        skip_existing: bool = True,
    ) -> list[ScoreReport]:
        """Evaluate a list of judges."""
        from utils import load_environment

        load_environment()

        if judges is None:
            judges = self.judges
        kwargs = {}
        if collection_name is None:
            collection_name = os.environ["MIMIC3_EHR_COLLECTION_NAME"]
            kwargs["vector_store"] = get_vectorstore(
                collection_name=collection_name, timeout=timeout
            )

        # Evaluate Each Judge
        score_reports = []
        for judge in tqdm(judges, desc="Evaluating Judges", disable=not show_progress):
            # Judge exists on disk. Load & skip re-evaluation.
            if skip_existing and judge.save_filepath.exists():
                self.logger.info(f"Skipping existing judge: {judge.save_filepath}")
                judge = JudgeSingleSubject.load(filepath=judge.save_filepath)
                score_report = judge.score_report
            # Judge does not exist on disk. Evaluate & save to disk.
            else:
                score_report = judge.evaluate(show_progress=False, **kwargs)
                score_reports.append(score_report)
                # Persist Judge Object and Evaluation Results to Disk
                if save_results:
                    judge.save(omit_input_data=True)
        return score_reports
