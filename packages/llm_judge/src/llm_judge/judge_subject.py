import asyncio
import logging
import os
import types
from copy import deepcopy
from pathlib import Path
from typing import Self

import pandas as pd
from llama_index.core.bridge.pydantic import BaseModel, ConfigDict, Field
from rag import (
    CLAIM_NODE,
    DENSE,
    HUMAN_AUTHOR,
    HYBRID,
    LLM_AUTHOR,
    RELATIVE_TIME,
    RERANK,
    SENTENCE_NODE,
)
from rag.components import get_llm
from utils import LazyFileLogger, load_pickle, save_pickle

from llm_judge.judge import Judge
from llm_judge.reference_context import ReferenceContextMaker
from llm_judge.schema import InputTextsAndReferenceContexts
from llm_judge.score_report import ScoreReport


class JudgeConfig(BaseModel):
    subject_id: int
    author_type: str
    node_kind: str
    retrieval_method: str
    top_n: int
    reference_format: str
    reference_only_admission: bool
    deduplicate_text: bool

    def to_config_str(self) -> str:
        """Serialize the JudgeConfig object as a string.
        Assumes JudgeConfig properties do not have underscores."""
        cfg_strs = []
        for key, value in self.dict().items():
            # Convert value to string; replace spaces with underscores
            value = str(value).replace(" ", "_")
            # Package key-value pair as a string
            cfg_strs.append(f"{key}={value}")
        cfg_str = ",".join(cfg_strs)
        return cfg_str

    @classmethod
    def from_config_str(cls, cfg_str: str) -> Self:
        """Deserialize the JudgeConfig object from a string.
        Assumes JudgeConfig properties do not have underscores."""
        cfg_dict = {}
        for cfg_pair in cfg_str.split(","):
            key, value = cfg_pair.split("=")
            # Convert underscores to spaces
            value = value.replace("_", " ")
            cfg_dict[key] = value
        return cls(**cfg_dict)


class JudgeSingleSubject(BaseModel):
    """Run the Judge on a single patient with a set of hyperparameters for the evaluation system.

    The Judge will take a discharge summary that has been broken into propositions,
    query the EHR (Vector Store) for relevant information to compose a
    reference context for each claim.

    Each example corresponds to a single SUBJECT_ID and their hospital admission (HADM_ID)
    for which there is a discharge summary which we will judge based on the
    patient's prior data in the EHR.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subject_id: int = Field(description="Subject ID for the patient.")
    hadm_id: int | None = Field(default=None, description="Hospital Admission ID for the patient.")
    hadm_start: pd.Timestamp | None = Field(
        default=None, description="Hospital Admission Start Timestamp."
    )
    hadm_end: pd.Timestamp | None = Field(
        default=None, description="Hospital Admission End Timestamp."
    )
    input_df: pd.DataFrame | None = Field(description="Dataframe of propositions for Subject ID.")
    author_type: str = Field(description=f"Input Text Author type: {LLM_AUTHOR}, {HUMAN_AUTHOR}.")
    node_kind: str = Field(description=f"Node Kind: {CLAIM_NODE}, {SENTENCE_NODE}.")
    retrieval_method: str = Field(description=f"Retrieval method: {DENSE}, {HYBRID}, {RERANK}.")
    rcm: ReferenceContextMaker = Field(description="Reference Context Maker object.")
    judge: Judge = Field(description="Judge object.")
    async_mode: bool = Field(
        default=True,
        description="Whether to use async when " "querying vectorstore and LLM generation.",
    )
    contexts: InputTextsAndReferenceContexts | None = Field(
        default=None, description="Input Texts and Reference Contexts for the Subject ID."
    )
    score_report: ScoreReport | None = Field(
        default=None, description="Score Report for the Subject ID."
    )
    save_dir: str | Path | None = Field(
        default=None, description="Directory to persist evaluation results."
    )
    save_filepath: str | Path | None = Field(
        default=None, description="Filepath to persist evaluation results."
    )
    log_filepath: str | Path | None = Field(
        default=None, description="Filepath to persist log messages."
    )
    logger: logging.Logger | None = Field(
        default=None, description="Logger object.", exclude=True, repr=False
    )

    @classmethod
    def class_name(cls) -> str:
        return "JudgeSingleSubject"

    @property
    def config(self) -> JudgeConfig:
        return JudgeConfig(
            subject_id=self.subject_id,
            author_type=self.author_type,
            node_kind=self.rcm.node_kind,
            retrieval_method=self.retrieval_method,
            top_n=self.rcm.top_n,
            reference_format=self.rcm.reference_format,
            reference_only_admission=self.rcm.reference_only_admission,
            deduplicate_text=self.rcm.deduplicate_text,
        )

    @classmethod
    def from_defaults(
        cls,
        input_df: pd.DataFrame,
        admission_df: pd.DataFrame | None = None,
        subject_id: int | None = None,
        hadm_id: int | None = None,
        hadm_start: str | pd.Timestamp | None = None,
        hadm_end: str | pd.Timestamp | None = None,
        reference_collection_name: str | None = None,
        author_type: str = LLM_AUTHOR,
        node_kind: str = CLAIM_NODE,
        retrieval_method: str | None = None,
        query_mode: str = "hybrid",
        top_k: int = 10,
        use_rerank: bool = False,
        top_n: int = 5,
        reference_format: str = RELATIVE_TIME,
        reference_only_admission: bool = False,
        deduplicate_text: bool = True,
        async_mode: bool = True,
        timeout: int = 300,
        temperature: float = 0.1,
        top_p: float = 0.9,
        num_judge_workers: int = 16,
        num_invalid_output_retries: int = 5,
        save_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        log_level: str = logging.DEBUG,
    ) -> Self:
        """Default constructor for JudgeSingleSubject object."""
        from utils import load_environment

        load_environment()

        # Default Vectorstore Collection
        if reference_collection_name is None:
            reference_collection_name = os.environ["MIMIC3_EHR_COLLECTION_NAME"]

        # Subset Notes Dataframe to specific SUBJECT_ID or HADM_ID
        if subject_id and hadm_id:
            input_df = input_df.query(f"SUBJECT_ID == {subject_id} & HADM_ID == {hadm_id}")
        elif subject_id:
            if "SUBJECT_ID" in input_df.columns:
                input_df = input_df.query(f"SUBJECT_ID == {subject_id}")
            if "HADM_ID" in input_df.columns:
                hadm_id = input_df.HADM_ID.iloc[0] if not input_df.empty else None
        elif hadm_id:
            if "HADM_ID" in input_df.columns:
                input_df = input_df.query(f"HADM_ID == {hadm_id}")
            if "SUBJECT_ID" in input_df.columns:
                subject_id = input_df.SUBJECT_ID.iloc[0] if not input_df.empty else None
        else:
            raise ValueError("Must provide either `subject_id` or `hadm_id` or both.")

        # Get Admission Start and End Times
        if admission_df is None and hadm_start is None and hadm_end is None:
            raise ValueError("Must provide either `admission_df` or `hadm_start` and `hadm_end`.")
        if admission_df is not None and "SUBJECT_ID" in input_df.columns:
            hadm_start = admission_df.query(f"SUBJECT_ID == {subject_id}").ADMITTIME.iloc[0]
            hadm_end = admission_df.query(f"SUBJECT_ID == {subject_id}").DISCHTIME.iloc[0]

        # Format Timestamps
        if isinstance(hadm_start, str):
            hadm_start = pd.Timestamp(hadm_start)
        if isinstance(hadm_end, str):
            hadm_end = pd.Timestamp(hadm_end)

        # If Retrieval Method specified, overrides query_mode, top_k, use_rerank
        # and use top_n to specify number of final items to use in reference context
        const = types.SimpleNamespace(DENSE=DENSE, HYBRID=HYBRID, RERANK=RERANK)
        if retrieval_method and top_n:
            match retrieval_method:
                case const.DENSE:
                    # Top K is the same as Top N
                    query_mode = "dense"
                    use_rerank = False
                    top_k = top_n
                case const.HYBRID:
                    # We retrieve N items for each retriever and after fusion,
                    # we have more than N items. We take the top N and drop
                    query_mode = "hybrid"
                    use_rerank = False
                    top_k = top_n
                case const.RERANK:
                    query_mode = "hybrid"
                    use_rerank = True
                    top_k = top_n
                case _:
                    raise ValueError(f"Invalid retrieval_method: {retrieval_method}")

        # Create Reference Context Maker
        rcm = ReferenceContextMaker.from_defaults(
            subject_id=subject_id,
            admission_start=hadm_start,
            admission_end=hadm_end,
            collection_name=reference_collection_name,
            node_kind=node_kind,
            query_mode=query_mode,
            top_k=top_k,
            use_rerank=use_rerank,
            top_n=top_n,
            reference_format=reference_format,
            reference_only_admission=reference_only_admission,
            deduplicate_text=deduplicate_text,
            async_mode=async_mode,
            timeout=timeout,
        )

        # Create Judge
        llm = get_llm(temperature=temperature, top_p=top_p)
        judge = Judge(
            llm=llm,
            num_workers=num_judge_workers,
            num_invalid_output_retries=num_invalid_output_retries,
        )

        # Define Save Filepaths
        judge_config = JudgeConfig(
            subject_id=subject_id,
            author_type=author_type,
            node_kind=rcm.node_kind,
            retrieval_method=retrieval_method,
            top_n=rcm.top_n,
            reference_format=rcm.reference_format,
            reference_only_admission=rcm.reference_only_admission,
            deduplicate_text=rcm.deduplicate_text,
        )
        config_str = judge_config.to_config_str()
        if save_dir:
            save_dir = Path(save_dir) / f"{subject_id}"
            save_dir.mkdir(parents=True, exist_ok=True)
        save_filename = f"{config_str}.pkl"
        save_filepath = Path(save_dir) / save_filename if save_dir else None
        # Define Log Filepaths
        if log_dir is None:
            log_dir = Path(save_dir) / "logs" if save_dir else None
        log_filename = f"{config_str}.log"
        log_filepath = Path(log_dir) / log_filename if log_dir else None
        # Setup Loggers
        logger = LazyFileLogger(name=__name__, level=log_level, log_file=log_filepath)
        rcm.setup_logger(level=log_level, log_file=log_filepath)
        judge.setup_logger(level=log_level, log_file=log_filepath)

        return cls(
            subject_id=subject_id,
            hadm_id=hadm_id,
            hadm_start=hadm_start,
            hadm_end=hadm_end,
            input_df=input_df,
            author_type=author_type,
            node_kind=node_kind,
            retrieval_method=retrieval_method,
            rcm=rcm,
            judge=judge,
            save_dir=str(save_dir),
            save_filepath=str(save_filepath),
            log_filepath=str(log_filepath),
            logger=logger,
        )

    def remove_logger(self) -> None:
        self.logger = None
        self.rcm.logger = None
        self.judge.logger = None

    def setup_logger(
        self, level: str = logging.DEBUG, log_file: str | Path | None = None
    ) -> LazyFileLogger:
        log_file = log_file or self.log_filepath
        self.logger = LazyFileLogger(name=__name__, level=level, log_file=log_file)
        self.rcm.setup_logger(level=level, log_file=log_file)
        self.judge.setup_logger(level=level, log_file=log_file)
        return self.logger

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        input_df: pd.DataFrame | None = None,
        setup_logger: bool = True,
        **kwargs,
    ) -> Self:
        """Load JudgeSingleSubject from a serialized file."""
        filepath = Path(filepath)
        obj: Self = load_pickle(filepath=filepath)
        if input_df is not None:
            obj.input_df = input_df
        if setup_logger:
            obj.logger = obj.setup_logger(level=kwargs.get("log_level", logging.DEBUG))
        return obj

    def load_from_save_filepath(
        self, filepath: str | Path | None = None, setup_logger: bool = True, **kwargs
    ) -> Self:
        """Load JudgeSingleSubject from a serialized file.
        This method returns a new SingleSubjectJudge object."""
        filepath = Path(self.save_filepath) if filepath is None else Path(filepath)
        return self.load(filepath=filepath, setup_logger=setup_logger, **kwargs)

    def save(self, save_dir: str | Path | None = None, omit_input_data: bool = True) -> None:
        """Serialize JudgeSingleSubject object to a file."""
        save_dir = Path(self.save_dir) if save_dir is None else Path(save_dir)
        filename = f"{self.config.to_config_str()}.pkl"
        self.remove_logger()
        obj: Self = deepcopy(self)
        if omit_input_data:
            obj.input_df = None
        save_pickle(obj=obj, filepath=save_dir / filename)

    def build_contexts(
        self, texts: list[str] | None = None, show_progress: bool = True, **kwargs
    ) -> InputTextsAndReferenceContexts:
        """Query vector store to build a reference context for each text."""
        texts = self.input_df.TEXT.tolist() if texts is None else texts
        self.contexts: InputTextsAndReferenceContexts = self.rcm.build(
            texts=texts, show_progress=show_progress, **kwargs
        )
        return self.contexts

    def make_score_report(
        self,
        texts: list[str],
        references: list[str],
        node_ids: list[str],
        show_progress: bool = True,
    ) -> ScoreReport:
        if self.async_mode:
            loop = asyncio.get_event_loop()
            self.score_report = loop.run_until_complete(
                self.judge.acall(
                    texts=texts,
                    references=references,
                    node_ids=node_ids,
                    show_progress=show_progress,
                )
            )
        else:
            self.score_report = self.judge(
                texts=texts,
                references=references,
                node_ids=node_ids,
                show_progress=show_progress,
            )
        return self.score_report

    def evaluate(
        self,
        texts: list[str] | None = None,
        node_ids: list[str] | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> ScoreReport:
        # Get default inputs
        texts = self.input_df.TEXT.tolist() if texts is None else texts
        node_ids = self.input_df.node_id.tolist() if node_ids is None else node_ids
        # Build Reference contexts
        contexts = self.build_contexts(texts=texts, show_progress=show_progress, **kwargs)
        # Generate Score Reports
        score_report = self.make_score_report(
            texts=contexts.texts,
            references=contexts.references,
            node_ids=node_ids,
            show_progress=show_progress,
        )
        return score_report
