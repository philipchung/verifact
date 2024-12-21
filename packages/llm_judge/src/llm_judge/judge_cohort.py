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
)
from rag.components import get_vectorstore
from tqdm.auto import tqdm
from utils import LazyFileLogger, get_utc_time, load_pandas

from llm_judge.judge_subject import JudgeSingleSubject
from llm_judge.score_report import ScoreReport


class JudgeCohort(BaseModel):
    """Create a cohort of judges with different hyperparameters.

    This class facilitates parallel evaluation of judges as different jobs using RQ.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Grid Hyperparameters for Judge Cohort
    subject_ids: list[int] | None = Field(
        default=None, description="List of Subject IDs for the patients."
    )
    author_types: list[str] | None = Field(
        default=None, description=f"Input Text Author type: {LLM_AUTHOR}, {HUMAN_AUTHOR}."
    )
    node_kinds: list[str] | None = Field(
        default=None,
        description=f"Node kind: {CLAIM_NODE}, {SENTENCE_NODE}.",
    )
    retrieval_methods: list[str] | None = Field(
        default=None,
        description=f"Retrieval method: {DENSE}, {HYBRID}, {RERANK}.",
    )
    top_ns: list[int] | None = Field(default=None, description="Top N results to evaluate.")
    reference_formats: list[str] | None = Field(
        default=None,
        description=f"Format for retrieved reference context: "
        f"{SCORE}, {ABSOLUTE_TIME}, {RELATIVE_TIME}.",
    )
    reference_only_admissions: list[bool] | None = Field(
        default=None, description="Whether to use only admissions for reference context."
    )
    deduplicate_text: list[bool] | None = Field(
        default=None, description="Whether to deduplicate text in reference context."
    )
    # Input Data
    input_dataset_dir: str | Path | None = Field(
        default=None,
        description="Directory containing the input dataset for the judges.",
    )
    judge_save_dir: str | Path | None = Field(
        default=None,
        description="Directory to save the evaluation results for the judges. "
        "The whole judge object is serialized and saved to disk.",
    )
    admissions_df: pd.DataFrame | None = Field(
        default=None, description="Dataframe of admissions data for the patients."
    )
    mimic_hc_claims_df: pd.DataFrame | None = Field(
        default=None, description="Dataframe of claims for original MIMIC-III hospital course."
    )
    mimic_hc_sentences_df: pd.DataFrame | None = Field(
        default=None, description="Dataframe of sentences for original MIMIC-III hospital course."
    )
    llm_hc_claims_df: pd.DataFrame | None = Field(
        default=None, description="Dataframe of claims for LLM-generated hospital course."
    )
    llm_hc_sentences_df: pd.DataFrame | None = Field(
        default=None, description="Dataframe of sentences for LLM-generated hospital course."
    )
    # Judge Cohort Outputs
    judges: list[JudgeSingleSubject] | None = Field(
        default=None, description="List of judge objects that comprise the cohort."
    )
    # Judge Evaluation Settings
    num_parallel_judge_evaluations: int = Field(
        default=64, description="Number of parallel judge evaluations."
    )
    num_parallel_workers_in_judge: int = Field(
        default=32, description="Number of parallel workers in each judge."
    )
    # Logger
    log_dir: str | Path | None = Field(default=None, description="Directory to save logs.")
    logger: logging.Logger | None = Field(
        default=None, description="Logger object.", exclude=True, repr=False
    )

    @classmethod
    def class_name(cls) -> str:
        return "JudgeCohort"

    @classmethod
    def from_defaults(
        cls,
        subject_ids: Sequence[int] | int | None = None,
        author_types: Sequence[str] | str = [LLM_AUTHOR, HUMAN_AUTHOR],
        node_kinds: Sequence[str] | str = [CLAIM_NODE, SENTENCE_NODE],
        retrieval_methods: Sequence[str] | str = ["dense", "hybrid", "rerank"],
        top_ns: Sequence[int] | int = [10, 25, 50, 100],
        reference_formats: Sequence[str] | str = ["score", "absolute_time", "relative_time"],
        reference_only_admissions: Sequence[bool] | bool = [True, False],
        deduplicate_text: Sequence[bool] | bool = True,
        input_dataset_dir: str | None = None,
        mimic_hc_nodes_dir_name: str = "bhc_nodes",
        llm_hc_nodes_dir_name: str = "llm_bhc_nodes",
        judge_save_dir: str | None = None,
        log_dir: str | None = None,
        log_level: str = logging.DEBUG,
        num_parallel_judge_evaluations: int = 64,
        num_parallel_workers_in_judge: int = 32,
    ) -> Self:
        """Create a cohort of judges with different hyperparameters."""
        from utils import load_environment

        load_environment()

        if isinstance(subject_ids, int):
            subject_ids = [subject_ids]
        if isinstance(author_types, str):
            author_types = [author_types]
        if isinstance(node_kinds, str):
            node_kinds = [node_kinds]
        if isinstance(retrieval_methods, str):
            retrieval_methods = [retrieval_methods]
        if isinstance(top_ns, int):
            top_ns = [top_ns]
        if isinstance(reference_formats, str):
            reference_formats = [reference_formats]
        if isinstance(reference_only_admissions, bool):
            reference_only_admissions = [reference_only_admissions]
        if isinstance(deduplicate_text, bool):
            deduplicate_text = [deduplicate_text]

        # Paths to Datasets & Outputs
        if input_dataset_dir is None:
            input_dataset_dir = Path(os.environ["MIMIC3_DATA_DIR"])
        if judge_save_dir is None:
            judge_save_dir = input_dataset_dir / "judges" / "judges"
            judge_save_dir.mkdir(parents=True, exist_ok=True)
        if log_dir is None:
            utc_time = get_utc_time()
            log_dir = Path(judge_save_dir).parent / "logs" / f"{utc_time}"
            log_dir.mkdir(parents=True, exist_ok=True)
        log_filepath = Path(log_dir) / "_judge_cohort.log"
        logger = LazyFileLogger(name=__name__, level=log_level, log_file=log_filepath)

        # Load Admissions Data
        admissions_df = load_pandas(input_dataset_dir / "admissions.feather")

        # Load Input Text
        mimic_hc_claims_df = load_pandas(
            filepath=input_dataset_dir / mimic_hc_nodes_dir_name / "claims.feather"
        )
        mimic_hc_sentences_df = load_pandas(
            filepath=input_dataset_dir / mimic_hc_nodes_dir_name / "sentences.feather"
        )

        llm_hc_claims_df = load_pandas(
            filepath=input_dataset_dir / llm_hc_nodes_dir_name / "claims.feather"
        )
        llm_hc_sentences_df = load_pandas(
            filepath=input_dataset_dir / llm_hc_nodes_dir_name / "sentences.feather"
        )

        # If no subject_ids specified, use all unique subject_ids in input text dataframes
        if not subject_ids:
            subject_ids = list(
                set(
                    mimic_hc_claims_df.SUBJECT_ID.unique().tolist()
                    + mimic_hc_sentences_df.SUBJECT_ID.unique().tolist()
                    + llm_hc_claims_df.SUBJECT_ID.unique().tolist()
                    + llm_hc_sentences_df.SUBJECT_ID.unique().tolist()
                )
            )
            logger.info(
                "Subject IDs not provided. Using all unique subject IDs in input data. "
                f"Subject IDs: {subject_ids}"
            )
        else:
            logger.info(f"Creating Judges for Subject IDs: {subject_ids}")

        return cls(
            subject_ids=subject_ids,
            author_types=author_types,
            node_kinds=node_kinds,
            retrieval_methods=retrieval_methods,
            top_ns=top_ns,
            reference_formats=reference_formats,
            reference_only_admissions=reference_only_admissions,
            deduplicate_text=deduplicate_text,
            input_dataset_dir=str(input_dataset_dir),
            judge_save_dir=str(judge_save_dir),
            admissions_df=admissions_df,
            mimic_hc_claims_df=mimic_hc_claims_df,
            mimic_hc_sentences_df=mimic_hc_sentences_df,
            llm_hc_claims_df=llm_hc_claims_df,
            llm_hc_sentences_df=llm_hc_sentences_df,
            num_parallel_judge_evaluations=num_parallel_judge_evaluations,
            num_parallel_workers_in_judge=num_parallel_workers_in_judge,
            log_dir=str(log_dir),
            logger=logger,
        )

    def setup_logger(
        self, level: str = logging.DEBUG, log_file: str | Path | None = None
    ) -> LazyFileLogger:
        log_file = log_file or self.log_filepath
        self.logger = LazyFileLogger(name=__name__, level=level, log_file=log_file)
        return self.logger

    @classmethod
    def load(cls, filepaths: Sequence[str | Path]) -> list[Self]:
        """Load a list of judge cohorts from disk.

        NOTE: this only loads the cohort but not hyperparameter settings
        for creating a cohort of judges.
        """
        judges = []
        for filepath in filepaths:
            try:
                judge = JudgeSingleSubject.load(filepath=filepath)
                judges.append(judge)
            except Exception as e:
                raise FileNotFoundError(f"Error loading judge cohort from {filepath}: {e}") from e
        judge_save_dir = Path(judges[0].save_dir) if judges else None
        return cls(judges=judges, judge_save_dir=judge_save_dir)

    def save(self, save_dir: str | Path | None = None, omit_input_data: bool = True) -> None:
        """Save the judge cohort to disk."""
        save_dir = Path(self.judge_save_dir) if save_dir is None else Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save each judge in the cohort
        for judge in self.judges:
            judge.save(save_dir=save_dir, omit_input_data=omit_input_data)

    def judge_generator(self, subject_ids: list[int] | None = None) -> Iterator[JudgeSingleSubject]:
        """Generator function yielding a single judge at a time."""
        subject_ids = self.subject_ids if subject_ids is None else subject_ids
        # Create judge cohort for each Subject ID
        for subject_id in subject_ids:
            # Select dataframes for input text kind & node kind
            for author_type, node_kind in itertools.product(self.author_types, self.node_kinds):
                # Load nodes for input text kind
                if author_type == LLM_AUTHOR:
                    if node_kind == CLAIM_NODE:
                        input_df = self.llm_hc_claims_df
                    elif node_kind == SENTENCE_NODE:
                        input_df = self.llm_hc_sentences_df
                    else:
                        raise ValueError(f"Invalid node_kind: {node_kind}")
                elif author_type == HUMAN_AUTHOR:
                    if node_kind == CLAIM_NODE:
                        input_df = self.mimic_hc_claims_df
                    elif node_kind == SENTENCE_NODE:
                        input_df = self.mimic_hc_sentences_df
                    else:
                        raise ValueError(f"Invalid node_kind: {node_kind}")
                else:
                    raise ValueError(f"Invalid author_type: {author_type}")

                # Combination of retrieval & reference formatting hyperparameters
                for (
                    top_n,
                    retrieval_method,
                    reference_format,
                    reference_only_admission,
                    deduplicate_text,
                ) in itertools.product(
                    self.top_ns,
                    self.retrieval_methods,
                    self.reference_formats,
                    self.reference_only_admissions,
                    self.deduplicate_text,
                ):
                    judge = JudgeSingleSubject.from_defaults(
                        input_df=input_df,
                        admission_df=self.admissions_df,
                        subject_id=subject_id,
                        author_type=author_type,
                        node_kind=node_kind,
                        retrieval_method=retrieval_method,
                        top_n=top_n,
                        reference_format=reference_format,
                        reference_only_admission=reference_only_admission,
                        deduplicate_text=deduplicate_text,
                        save_dir=self.judge_save_dir,
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
        self.judges = judges
        return self.judges

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
