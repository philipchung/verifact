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
    SPARSE,
)
from rag.components import get_aux_llm, get_llm
from rag.vector_stores.qdrant import QdrantVectorStore
from utils import LazyFileLogger, load_pickle, save_pickle

from llm_judge.judge import Judge
from llm_judge.reference_context import ReferenceContextMaker
from llm_judge.schema import InputTextsAndReferenceContexts
from llm_judge.score_report import ScoreReport


class JudgeConfig(BaseModel):
    subject_id: int
    author_type: str
    proposition_type: str
    fact_type: str
    retrieval_method: str
    top_n: int
    reference_format: str
    reference_only_admission: bool
    deduplicate_text: bool

    def to_config_str(self) -> str:
        """Serialize the JudgeConfig object as a string.
        Assumes JudgeConfig properties do not have underscores."""
        cfg_strs = []
        for key, value in self.model_dump().items():
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
    """Run LLM-as-a-Judge on a single patient with a set of hyperparameters.

    The Judge takes long-form narrative text that has been broken into propositions in the
    form of `proposition_df` (1 proposition per row). For each proposition, the Judge will
    query the EHR (Vector Store) for relevant information to compose a reference context
    and then will pass both proposition and reference context to a LLM-as-a-Judge.

    The ReferenceContextMaker object is responsible for querying the Vector Store and
    building the reference context.
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
    proposition_df: pd.DataFrame | None = Field(
        description="Dataframe of propositions for Subject ID."
    )
    author_type: str = Field(description=f"Input Text Author type: {LLM_AUTHOR}, {HUMAN_AUTHOR}.")
    proposition_type: str = Field(description=f"Proposition Type: {CLAIM_NODE}, {SENTENCE_NODE}.")
    fact_type: str | None = Field(
        default=None,
        description=f"EHR Fact type (node kind): {CLAIM_NODE}, {SENTENCE_NODE}."
        "If `None`, will default to using the same as `proposition_type`.",
    )
    retrieval_method: str = Field(
        description=f"Retrieval method: {DENSE}, {SPARSE}, {HYBRID}, {RERANK}."
    )
    rcm: ReferenceContextMaker = Field(description="Reference Context Maker object.")
    judge: Judge = Field(description="Judge object.")
    is_reasoning_model: bool = Field(
        default=False, description="Whether the Judge is a reasoning model."
    )
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
    judge_save_dir: str | Path | None = Field(
        default=None, description="Directory to persist SingleSubjectJudge object."
    )
    score_report_save_dir: str | Path | None = Field(
        default=None, description="Directory to persist ScoreReport evaluation results."
    )
    log_dir: str | Path | None = Field(
        default=None, description="Directory to persist log messages."
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
            proposition_type=self.proposition_type,
            fact_type=self.fact_type,
            retrieval_method=self.retrieval_method,
            top_n=self.rcm.top_n,
            reference_format=self.rcm.reference_format,
            reference_only_admission=self.rcm.reference_only_admission,
            deduplicate_text=self.rcm.deduplicate_text,
        )

    @classmethod
    def from_defaults(
        cls,
        proposition_df: pd.DataFrame,
        admission_df: pd.DataFrame | None = None,
        subject_id: int | None = None,
        hadm_id: int | None = None,
        hadm_start: str | pd.Timestamp | None = None,
        hadm_end: str | pd.Timestamp | None = None,
        reference_collection_name: str | None = None,
        author_type: str = LLM_AUTHOR,
        proposition_type: str = CLAIM_NODE,
        fact_type: str | None = None,
        retrieval_method: str | None = None,
        query_mode: str = HYBRID,
        top_k: int = 10,
        use_rerank: bool = False,
        top_n: int = 5,
        reference_format: str = RELATIVE_TIME,
        reference_only_admission: bool = False,
        deduplicate_text: bool = True,
        is_reasoning_model: bool = False,
        async_mode: bool = True,
        timeout: int = 300,
        temperature: float = 0.1,
        top_p: float = 1.0,
        num_judge_workers: int = 16,
        num_invalid_output_retries: int = 5,
        judge_save_dir: str | Path | None = None,
        score_report_save_dir: str | Path | None = None,
        log_dir: str | Path | None = None,
        log_level: str = logging.DEBUG,
    ) -> Self:
        """Default constructor for JudgeSingleSubject object."""
        from utils import load_environment

        load_environment()

        # Default Vectorstore Collection
        reference_collection_name = (
            reference_collection_name or os.environ["MIMIC3_EHR_COLLECTION_NAME"]
        )

        # Subset Propositions Dataframe to specific subject_id or hadm_id
        if not subject_id and not hadm_id:
            raise ValueError("Must provide either `subject_id` or `hadm_id` or both.")
        if subject_id and "subject_id" in proposition_df.columns:
            proposition_df = proposition_df.query(f"subject_id == {subject_id}")
        if hadm_id and "hadm_id" in proposition_df.columns:
            proposition_df = proposition_df.query(f"hadm_id == {hadm_id}")
        subject_id = proposition_df.subject_id.iloc[0] if not proposition_df.empty else None
        hadm_id = proposition_df.hadm_id.iloc[0] if not proposition_df.empty else None

        # Get Admission Start and End Times
        if admission_df is None and hadm_start is None and hadm_end is None:
            raise ValueError("Must provide either `admission_df` or `hadm_start` and `hadm_end`.")
        if admission_df is not None and "SUBJECT_ID" in admission_df.columns:
            hadm_start = admission_df.query(f"SUBJECT_ID == {subject_id}").ADMITTIME.iloc[0]
            hadm_end = admission_df.query(f"SUBJECT_ID == {subject_id}").DISCHTIME.iloc[0]

        # Format Timestamps
        if isinstance(hadm_start, str):
            hadm_start = pd.Timestamp(hadm_start)
        if isinstance(hadm_end, str):
            hadm_end = pd.Timestamp(hadm_end)

        # If Retrieval Method specified, overrides query_mode, top_k, use_rerank
        # and use top_n to specify number of final items to use in reference context
        const = types.SimpleNamespace(DENSE=DENSE, SPARSE=SPARSE, HYBRID=HYBRID, RERANK=RERANK)
        if retrieval_method and top_n:
            match retrieval_method:
                case const.DENSE:
                    # Top K is the same as Top N
                    query_mode = "dense"
                    use_rerank = False
                    top_k = top_n
                case const.SPARSE:
                    # Top K is the same as Top N
                    query_mode = "sparse"
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

        # If EHR Fact Type not specified, default to Proposition Type
        fact_type = fact_type or proposition_type

        # Create Reference Context Maker
        rcm = ReferenceContextMaker.from_defaults(
            subject_id=subject_id,
            admission_start=hadm_start,
            admission_end=hadm_end,
            collection_name=reference_collection_name,
            fact_type=fact_type,
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
        aux_llm = get_aux_llm(temperature=temperature, top_p=top_p) if is_reasoning_model else None
        judge = Judge.from_defaults(
            llm=llm,
            aux_llm=aux_llm,
            is_reasoning_model=is_reasoning_model,
            num_workers=num_judge_workers,
            num_invalid_output_retries=num_invalid_output_retries,
        )

        # Define Save Filepaths
        judge_config = JudgeConfig(
            subject_id=subject_id,
            author_type=author_type,
            proposition_type=proposition_type,
            fact_type=fact_type,
            retrieval_method=retrieval_method,
            top_n=rcm.top_n,
            reference_format=rcm.reference_format,
            reference_only_admission=rcm.reference_only_admission,
            deduplicate_text=rcm.deduplicate_text,
        )
        config_str = judge_config.to_config_str()

        # Define Log Filepaths
        log_dir = Path(log_dir) if log_dir else None
        if log_dir is None:
            log_dir = Path(judge_save_dir) / "logs" / f"{subject_id}" if judge_save_dir else None
        log_dir.mkdir(parents=True, exist_ok=True) if log_dir else None
        log_filename = f"{config_str}.log"
        log_filepath = log_dir / log_filename if log_dir else None
        # Setup Loggers
        logger = LazyFileLogger(name=__name__, level=log_level, log_file=log_filepath)
        rcm.setup_logger(level=log_level, log_file=log_filepath)
        judge.setup_logger(level=log_level, log_file=log_filepath)

        return cls(
            subject_id=subject_id,
            hadm_id=hadm_id,
            hadm_start=hadm_start,
            hadm_end=hadm_end,
            proposition_df=proposition_df,
            author_type=author_type,
            proposition_type=proposition_type,
            fact_type=fact_type,
            retrieval_method=retrieval_method,
            rcm=rcm,
            judge=judge,
            is_reasoning_model=is_reasoning_model,
            judge_save_dir=judge_save_dir,
            score_report_save_dir=score_report_save_dir,
            log_dir=str(log_dir),
            log_filepath=log_filepath,
            logger=logger,
        )

    def remove_logger_and_vectorstore(self) -> None:
        self.logger = None
        self.rcm.logger = None
        self.rcm.vector_store = None
        self.judge.logger = None

    def setup_logger_and_vectorstore(
        self,
        level: str = logging.DEBUG,
        log_file: str | Path | None = None,
        vectorstore: QdrantVectorStore | None = None,
        collection_name: str = None,
        timeout: int = 300,
    ) -> LazyFileLogger:
        log_file = log_file or self.log_filepath
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        self.logger = LazyFileLogger(name=__name__, level=level, log_file=log_file)
        self.rcm.setup_logger(level=level, log_file=log_file)
        self.rcm.setup_vectorstore(
            vectorstore=vectorstore, collection_name=collection_name, timeout=timeout
        )
        self.judge.setup_logger(level=level, log_file=log_file)
        return self.logger

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        proposition_df: pd.DataFrame | None = None,
        setup_logger_and_vectorstore: bool = True,
        **kwargs,
    ) -> Self:
        """Load JudgeSingleSubject from a serialized file."""
        filepath = Path(filepath)
        obj: Self = load_pickle(filepath=filepath)
        if proposition_df is not None:
            obj.proposition_df = proposition_df
        if setup_logger_and_vectorstore:
            obj.logger = obj.setup_logger_and_vectorstore(
                level=kwargs.get("log_level", logging.DEBUG)
            )
        return obj

    def load_judge_from_save_filepath(
        self,
        filepath: str | Path | None = None,
        setup_logger_and_vectorstore: bool = True,
        **kwargs,
    ) -> Self:
        """Load JudgeSingleSubject from a serialized file.
        This method returns a new SingleSubjectJudge object."""
        filepath = self.get_judge_save_filepath() if filepath is None else Path(filepath)
        return self.load(
            filepath=filepath, setup_logger_and_vectorstore=setup_logger_and_vectorstore, **kwargs
        )

    def save(self, save_dir: str | Path | None = None, omit_input_data: bool = True) -> None:
        """Serialize JudgeSingleSubject object to a file."""
        # Remove logger and vectorstore before saving
        self.remove_logger_and_vectorstore()
        obj: Self = deepcopy(self)
        # Remove Input Data to reduce file size
        if omit_input_data:
            obj.proposition_df = None
        # Save Judge object to file
        filepath = self.get_judge_save_filepath(save_dir=save_dir)
        save_pickle(obj=obj, filepath=filepath)

    def evaluate(
        self,
        texts: list[str] | None = None,
        proposition_ids: list[str] | None = None,
        proposition_type: str | None = None,
        fact_type: str | None = None,
        show_progress: bool = True,
        **kwargs,
    ) -> ScoreReport:
        # Get default inputs
        texts = texts or self.proposition_df.text.tolist()
        proposition_ids = proposition_ids or self.proposition_df.proposition_id.tolist()
        # Build Reference contexts
        contexts = self.build_contexts(texts=texts, show_progress=show_progress, **kwargs)
        # Generate Score Reports
        score_report = self.make_score_report(
            texts=contexts.texts,
            references=contexts.references,
            proposition_ids=proposition_ids,
            proposition_type=proposition_type or self.proposition_type,
            fact_type=fact_type or self.fact_type,
            show_progress=show_progress,
        )
        return score_report

    def build_contexts(
        self, texts: list[str] | None = None, show_progress: bool = True, **kwargs
    ) -> InputTextsAndReferenceContexts:
        """Query vector store to build a reference context for each text."""
        texts = self.proposition_df.TEXT.tolist() if texts is None else texts
        self.contexts: InputTextsAndReferenceContexts = self.rcm.build(
            texts=texts, show_progress=show_progress, **kwargs
        )
        return self.contexts

    def make_score_report(
        self,
        texts: list[str],
        references: list[str],
        proposition_ids: list[str],
        proposition_type: str | None = None,
        fact_type: str | None = None,
        show_progress: bool = True,
    ) -> ScoreReport:
        if self.async_mode:
            loop = asyncio.get_event_loop()
            self.score_report = loop.run_until_complete(
                self.judge.acall(
                    texts=texts,
                    references=references,
                    proposition_ids=proposition_ids,
                    proposition_type=proposition_type,
                    fact_type=fact_type,
                    judge_config=self.config.model_dump(),
                    show_progress=show_progress,
                )
            )
        else:
            self.score_report = self.judge(
                texts=texts,
                references=references,
                proposition_ids=proposition_ids,
                proposition_type=proposition_type,
                fact_type=fact_type,
                judge_config=self.config.model_dump(),
                show_progress=show_progress,
            )
        return self.score_report

    def save_score_report(self, save_dir: str | Path | None = None) -> None:
        filepath = self.get_score_report_save_filepath(save_dir=save_dir)
        self.score_report.save(filepath=filepath)

    def get_judge_save_filepath(self, save_dir: str | Path | None = None) -> Path:
        save_dir = Path(save_dir or self.judge_save_dir)
        filename = f"{self.config.to_config_str()}.pkl"
        return save_dir / f"{self.subject_id}" / filename

    def get_score_report_save_filepath(self, save_dir: str | Path | None = None) -> Path:
        save_dir = Path(save_dir or self.score_report_save_dir)
        filename = f"{self.config.to_config_str()}.feather"
        return save_dir / f"{self.subject_id}" / filename
