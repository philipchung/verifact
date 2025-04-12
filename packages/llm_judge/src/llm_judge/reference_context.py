import asyncio
import logging
import types
from pathlib import Path
from typing import Self

import pandas as pd
from llama_index.core.bridge.pydantic import BaseModel, ConfigDict, Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from qdrant_client.http.models import DatetimeRange, FieldCondition, Filter
from rag import ABSOLUTE_TIME, CLAIM_NODE, RELATIVE_TIME, SCORE
from rag.components import get_rerank_model, get_vectorstore
from rag.schema import NodeWithScore
from rag.vector_stores import (
    QdrantVectorStore,
    nodes_to_dataframe,
    vector_store_query_result_to_node_with_score,
)
from utils import LazyFileLogger

from llm_judge.schema import InputTextsAndReferenceContexts


class ReferenceContextMaker(BaseModel):
    """Create a reference context for each claim for a given subject ID.

    A set of claims is retrieved from the VectorStore representation of the EHR
    for each claim. The retrieved results are used to format a reference context.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    subject_id: int | None = Field(default=None, description="MIMIC-III Subject ID")
    admission_start: pd.Timestamp | None = Field(
        default=None, description="Hospital Admission Start Time"
    )
    admission_end: pd.Timestamp | None = Field(
        default=None, description="Hospital Admission End Time"
    )
    fact_type: str | None = Field(
        default=CLAIM_NODE, description="EHR Fact Type (Node Kind) to Query"
    )
    query_mode: str = Field(default="hybrid", description="Query Mode for VectorStore")
    top_k: int = Field(default=5, description="Top K results to retrieve from each retriever.")
    dense_vector_name: str = Field(default="dense", description="Dense Vector Name")
    sparse_vector_name: str = Field(default="sparse", description="Sparse Vector Name")
    collection_name: str | None = Field(default=None, description="Vector Store Collection Name")
    vector_store: QdrantVectorStore | None = Field(default=None, description="Qdrant Vector Store")
    timeout: float = Field(default=60.0, description="Timeout for VectorStore Query in seconds")
    async_mode: bool = Field(default=True, description="Whether to run queries asynchronously")
    num_workers: int = Field(default=16, description="Number of workers for async queries")
    use_rerank: bool = Field(
        default=True, description="Whether to use rerank model for postprocessing."
    )
    rerank_model: BaseNodePostprocessor | None = Field(default=None, description="Rerank Model.")
    top_n: int | None = Field(
        default=None,
        description="Final number of top N results to use in reference context after "
        "fusing all retrieved results & reranking.",
    )
    reference_format: str = Field(
        default=RELATIVE_TIME,
        description=f"Format for retrieved reference context: "
        f"{SCORE}, {ABSOLUTE_TIME}, {RELATIVE_TIME}.",
    )
    deduplicate_text: bool = Field(
        default=True, description="Deduplicate exact text matches in reference context"
    )
    reference_only_admission: bool = Field(
        default=False,
        description="Limit reference context to current hospital admission."
        "Otherwise, reference can extend infinitely into the past.",
    )
    logger: logging.Logger | None = Field(
        default=None, description="Logger object.", exclude=True, repr=False
    )

    @classmethod
    def class_name(cls) -> str:
        return "ReferenceContextMaker"

    @classmethod
    def from_defaults(
        cls,
        subject_id: int | None = None,
        admission_start: pd.Timestamp | str | None = None,
        admission_end: pd.Timestamp | str | None = None,
        fact_type: str | None = CLAIM_NODE,
        query_mode: str = "hybrid",
        top_k: int = 5,
        collection_name: str | None = None,
        vector_store: QdrantVectorStore | None = None,
        timeout: float = 60.0,
        async_mode: bool = True,
        num_workers: int = 16,
        use_rerank: bool = True,
        rerank_model: BaseNodePostprocessor | None = None,
        top_n: int | None = None,
        reference_format: str = "time",
        deduplicate_text: bool = True,
        reference_only_admission: bool = False,
        log_filepath: str | Path | None = None,
        log_level: str = logging.DEBUG,
    ) -> Self:
        from utils import load_environment

        load_environment()
        log_filepath = Path(log_filepath) if log_filepath else None
        if log_filepath:
            logger = LazyFileLogger(name=__name__, level=log_level, log_file=log_filepath)
        else:
            logger = None

        # Format Timestamps
        if isinstance(admission_start, str):
            admission_start = pd.Timestamp(admission_start)
        if isinstance(admission_end, str):
            admission_end = pd.Timestamp(admission_end)

        # Vector Store Config
        if vector_store is None and collection_name is None:
            raise ValueError("Must provide either collection_name or vector_store.")
        if vector_store is None:
            vector_store = get_vectorstore(collection_name=collection_name, timeout=timeout)

        # Rerank Model Config
        if rerank_model is None:
            rerank_model = get_rerank_model()

        # NOTE: By default, query k results from each retriever and return n=k results.
        # If hybrid retrieval is used, the top k results are returned from each retriever.
        # so the number of final results n returned is less than the total number
        # of retrieved results.
        if top_n is None:
            top_n = top_k

        return cls(
            subject_id=subject_id,
            admission_start=admission_start,
            admission_end=admission_end,
            fact_type=fact_type,
            query_mode=query_mode,
            top_k=top_k,
            collection_name=collection_name,
            vector_store=vector_store,
            timeout=timeout,
            async_mode=async_mode,
            num_workers=num_workers,
            use_rerank=use_rerank,
            rerank_model=rerank_model,
            top_n=top_n,
            reference_format=reference_format,
            deduplicate_text=deduplicate_text,
            reference_only_admission=reference_only_admission,
            logger=logger,
        )

    def setup_logger(
        self, level: str = logging.DEBUG, log_file: str | Path | None = None
    ) -> LazyFileLogger:
        self.logger = LazyFileLogger(name=__name__, level=level, log_file=log_file)
        return self.logger

    def setup_vectorstore(
        self,
        vectorstore: QdrantVectorStore | None,
        collection_name: str | None = None,
        timeout: float = 60.0,
    ) -> QdrantVectorStore:
        if vectorstore:
            self.vector_store = vectorstore
        else:
            collection_name = collection_name or self.collection_name
            timeout = timeout or self.timeout
            self.vector_store = get_vectorstore(collection_name=collection_name, timeout=timeout)
        return self.vector_store

    def build(
        self, texts: list[str], show_progress: bool = False, **kwargs
    ) -> InputTextsAndReferenceContexts:
        """For each text, retrieve reference context from VectorStore and format."""
        vector_store = kwargs.get("vector_store", self.vector_store)
        dense_vector_name = kwargs.get("dense_vector_name", self.dense_vector_name)
        sparse_vector_name = kwargs.get("sparse_vector_name", self.sparse_vector_name)
        results = self._query(
            texts,
            show_progress=show_progress,
            vector_store=vector_store,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
        )
        references = [
            self._format_reference_context(retrieved_context=result, text=text)
            for text, result in zip(texts, results)
        ]
        return InputTextsAndReferenceContexts(
            texts=texts, references=references, raw_references=results
        )

    def _query(
        self, texts: list[str], show_progress: bool = False, desc: str = "Querying", **kwargs
    ) -> list[VectorStoreQuery]:
        """Retrieve information from VectorStore for each text in `texts`."""
        ## Filters for Subject ID and Node Kind
        criteria = []
        if self.subject_id:
            subject_id_criteria = {
                "key": "SUBJECT_ID",
                "value": self.subject_id,
                "operator": "==",
            }
            criteria.append(subject_id_criteria)
        if self.fact_type:
            fact_type_criteria = {
                "key": "node_kind",
                "value": self.fact_type,
                "operator": "==",
            }
            criteria.append(fact_type_criteria)
        metadata_filter = MetadataFilters.from_dicts(criteria)
        if self.reference_only_admission:
            if not self.admission_start or not self.admission_end:
                raise ValueError(
                    "Admission start and end times must be provided for `reference_only_admission`."
                )
            # Qdrant Vector Store performs datetime range filtering using ISO8601 strings
            # In MIMIC-III, CHARTDATE is always available. CHARTTIME is mostly available
            # but NaT for some rows. For consistency, we use CHARTDATE for filtering
            # and consider the admission start time boundary to be at midnight of the
            # admission start date & the admission end time boundary to be at 23:59 of
            # the admission end date.
            admission_start = pd.Timestamp(
                year=self.admission_start.year,
                month=self.admission_start.month,
                day=self.admission_start.day,
                hour=0,
                minute=0,
                second=0,
            )
            admission_end = pd.Timestamp(
                year=self.admission_end.year,
                month=self.admission_end.month,
                day=self.admission_end.day,
                hour=23,
                minute=59,
                second=59,
            )
            qdrant_filters = Filter(
                must=[
                    FieldCondition(
                        key="CHARTDATE",
                        range=DatetimeRange(
                            gt=None,
                            gte=admission_start.isoformat(),
                            lt=None,
                            lte=admission_end.isoformat(),
                        ),
                    )
                ]
            )
        else:
            qdrant_filters = None
        # Query Mode
        match self.query_mode:
            case "dense":
                query_mode = VectorStoreQueryMode.DEFAULT
            case "sparse":
                query_mode = VectorStoreQueryMode.SPARSE
            case "hybrid":
                query_mode = VectorStoreQueryMode.HYBRID
            case _:
                raise ValueError(f"Invalid query mode: {self.query_mode}")
        # Form Queries
        queries = [
            VectorStoreQuery(
                query_str=text,
                mode=query_mode,
                filters=metadata_filter,
                similarity_top_k=self.top_k,
            )
            for text in texts
        ]
        # Run Queries
        vector_store = kwargs.get("vector_store", self.vector_store)
        dense_vector_name = kwargs.get("dense_vector_name", self.dense_vector_name)
        sparse_vector_name = kwargs.get("sparse_vector_name", self.sparse_vector_name)
        if self.async_mode:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                vector_store.aquery(
                    queries,
                    qdrant_filters=qdrant_filters,
                    dense_vector_name=dense_vector_name,
                    sparse_vector_name=sparse_vector_name,
                    show_progress=show_progress,
                )
            )
        else:
            results = vector_store.query(
                queries,
                qdrant_filters=qdrant_filters,
                dense_vector_name=dense_vector_name,
                sparse_vector_name=sparse_vector_name,
                show_progress=show_progress,
            )
        # Convert Query Results to Nodes with Scores
        # NOTE: each input `text` corresponds to a retrieved list[NodeWithScore]
        results_nws: list[list[NodeWithScore]] = [
            vector_store_query_result_to_node_with_score(result) for result in results
        ]

        # Deduplicate Identical Text Results, Keeping First Occurrence
        if self.deduplicate_text:
            deduplicated_results_nws = []
            for result_nws in results_nws:
                result_nws = sorted(result_nws, key=lambda x: x.score, reverse=True)
                # Identify exact text duplicates, keeping first one
                df = nodes_to_dataframe(result_nws)
                is_duplicated = df.duplicated(subset=["TEXT"], keep="first")
                if is_duplicated.any():
                    # Get only duplicated node ids & filter out duplicated nodes
                    duplicated_rows = df[is_duplicated]
                    duplicated_node_ids = duplicated_rows.node_id.tolist()
                    deduplicated_result_nws = [
                        node for node in result_nws if node.node_id not in duplicated_node_ids
                    ]
                    deduplicated_results_nws.append(deduplicated_result_nws)
                else:
                    deduplicated_results_nws.append(result_nws)
            results_nws = deduplicated_results_nws

        # Rerank Results
        if self.use_rerank:
            # Persist Pre-Rerank (Post-Fusion) Embedding Scores in Metadata
            for text, result_nws in zip(texts, results_nws, strict=False):
                for node in result_nws:
                    node.metadata["embedding_score"] = node.score
                # Rerank Results
                result_nws = self.rerank_model.postprocess_nodes(nodes=result_nws, query_str=text)
                # Persist Rerank Scores in Metadata
                for node in result_nws:
                    node.metadata["rerank_score"] = node.score
                # Sort by Reranker Score
                result_nws = sorted(result_nws, key=lambda x: x.score, reverse=True)

        # Limit Results to Top N
        if self.top_n:
            retrieval_outputs = []
            for result_nws in results_nws:
                if len(result_nws) > self.top_n:
                    result_nws = result_nws[: self.top_n]
                retrieval_outputs.append(result_nws)
        else:
            retrieval_outputs = results_nws
        return retrieval_outputs

    def _format_reference_context(
        self,
        retrieved_context: list[NodeWithScore],
        text: str | None = None,
    ) -> str:
        """Format a single retrieved context (from one query) into a string."""
        # Reformat retrieved context into DataFrame
        df = nodes_to_dataframe(retrieved_context)
        if df.empty:
            self.logger.warning(
                f"Proposition: {text}, "
                f"Empty DataFrame for Subject ID: {self.subject_id}, "
                f"EHR Fact Type: {self.fact_type}, Query Mode: {self.query_mode}, "
                f"Top N: {self.top_n}, Reference Format: {self.reference_format}, "
                f"Deduplicate Text: {self.deduplicate_text}, "
                f"Reference Only Admission: {self.reference_only_admission}, "
            )
            # Compose output string for No EHR Context
            output_str = (
                "Electronic Health Record Context\n"
                "Ordered by Relevance Score (Highest to Lowest):\n"
                "No Electronic Health Record Context Found."
            )
            return output_str
        else:
            const = types.SimpleNamespace(
                SCORE=SCORE, ABSOLUTE_TIME=ABSOLUTE_TIME, RELATIVE_TIME=RELATIVE_TIME
            )
            match self.reference_format:
                case const.SCORE:
                    return self._format_reference_context_by_score(df)
                case const.ABSOLUTE_TIME:
                    return self._format_reference_context_by_absolute_time(df)
                case const.RELATIVE_TIME:
                    return self._format_reference_context_by_relative_time(df)
                case _:
                    raise ValueError(f"Invalid reference format: {self.reference_format}")

    def _format_reference_context_by_score(self, df: pd.DataFrame) -> str:
        # Ensure retrieved context is ordered by score
        df.sort_values(by="score", ascending=False, inplace=True)
        # Compose output string
        output_str = (
            "Electronic Health Record Context\n" "Ordered by Relevance Score (Highest to Lowest):\n"
        )
        for i, row in enumerate(df.itertuples()):
            date, time, cat, desc = get_date_time_category_description(row)
            output_str += (
                f"{i+1}. Score: {row.score:.2f}, "
                f"Note Category: {cat}, "
                f"Note Description: {desc} "
                f"| Text: {row.TEXT}\n"
            )
        return output_str

    def _format_reference_context_by_absolute_time(self, df: pd.DataFrame) -> str:
        # Ensure retrieved context is ordered by time
        df = sort_reference_context_by_time(df)
        # Compose output string
        output_str = (
            f"Hospital Admission Start: {self.admission_start}\n"
            f"Hospital Admission End: {self.admission_end}\n"
            ""
            "Electronic Health Record Context\n"
            "Ordered by Time (Earliest to Latest):\n"
        )
        for i, row in enumerate(df.itertuples()):
            date, time, cat, desc = get_date_time_category_description(row)
            output_str += (
                f"{i+1}. Date: {date}, Time: {time}, "
                f"Note Category: {cat}, "
                f"Note Description: {desc} "
                f"| Text: {row.TEXT}\n"
            )
        return output_str

    def _format_reference_context_by_relative_time(self, df: pd.DataFrame) -> str:
        # Ensure retrieved context is ordered by time
        df = sort_reference_context_by_time(df)
        df = df.assign(TIMEDELTA=self.admission_end - df.CHARTDATE)
        # Get Relative Times (days & hours ago from admission end)
        admission_start_timedelta = self.admission_end - self.admission_start
        admission_days = admission_start_timedelta.components.days
        admission_hrs = admission_start_timedelta.components.hours
        # Compose output string
        output_str = (
            f"Hospital Admission Start: {admission_days} days {admission_hrs} hours ago\n"
            f"Hospital Admission End: Now\n"
            ""
            "Electronic Health Record Context\n"
            "Ordered by Time (Earliest to Latest):\n"
        )
        for i, row in enumerate(df.itertuples()):
            note_timedelta = self.admission_end - row.CHARTDATE
            note_td_days = note_timedelta.components.days
            note_td_hours = note_timedelta.components.hours

            _, _, cat, desc = get_date_time_category_description(row)
            output_str += (
                f"{i+1}. When: {note_td_days} days {note_td_hours} hours ago, "
                f"Note Category: {cat}, "
                f"Note Description: {desc} "
                f"| Text: {row.TEXT}\n"
            )
        return output_str


def sort_reference_context_by_time(df: pd.DataFrame) -> pd.DataFrame:
    # Timestamps stored in VectorStore are ISO8601 strings. Convert to Timestamps.
    df = df.assign(
        CHARTDATE_TS=df.CHARTDATE.apply(pd.Timestamp),
        CHARTTIME_TS=df.CHARTTIME.apply(pd.Timestamp),
        STORETIME_TS=df.STORETIME.apply(pd.Timestamp),
    ).sort_values(
        by=["CHARTDATE_TS", "CHARTTIME_TS", "STORETIME_TS"],
        ascending=True,
    )
    return df


def get_date_time_category_description(row: pd.Series) -> tuple:
    date_str = row.CHARTDATE.strftime("%Y-%m-%d")  # All notes have Date
    time_str = (
        row.CHARTTIME.strftime("%X") if pd.notna(row.CHARTTIME) else "Unknown"
    )  # Not all notes have DateTime
    category = row.CATEGORY  # Note Category
    description = row.DESCRIPTION  # Note Description
    return date_str, time_str, category, description
