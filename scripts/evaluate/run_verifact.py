# %% [markdown]
# ## Run VeriFact on any text written about a patient.
#
# This script combines functionality in `dataset/decompose_text.py` and `dataset/run_verifact.py`.
#
# It performs the following:
# 1. Decomposes the input text into either atomic claim or sentence propositions.
# 2. Runs VeriFact on the decomposed propositions, which retrieves facts from the patient's
#    EHR Vector Database to generate reference contexts for each proposition and then
#    evaluates the propositions using VeriFact's LLM-as-a-Judge.
#
# Since this script is with respect to patient-specific facts, you must specify the patient's
# Subject ID along with the input text file. This script also assumes that the EHR Vector Database
# has been constructed for this Subject ID. If this has not yet been performed, please follow
# instructions to run `dataset/decompose_text.py` on the EHR clinical notes.
# %%
import logging
import os
from pathlib import Path
from typing import Annotated, Literal

import nest_asyncio
import pandas as pd
import typer
from llama_index.core.schema import MetadataMode
from llm_judge import JudgeSingleSubject, ScoreReport
from qdrant_client.http.models import FieldCondition, Filter, MatchValue
from rag import MIMIC_NODE
from rag.components import (
    get_atomic_claim_node_parser,
    get_embed_model,
    get_llm,
    get_semantic_node_parser,
    get_single_sentence_node_parser,
    get_vectorstore,
)
from rag.embedding import M3Embedding
from rag.node_parser import (
    AtomicClaimNodeParser,
    SemanticSplitterNodeParser,
    SingleSentenceNodeParser,
)
from rag.schema import BaseNode, Document
from rag.vector_stores import nodes_to_dataframe
from rag.vector_stores.qdrant import QdrantVectorStore
from utils import (
    create_uuid_from_string,
    get_event_loop,
    get_utc_time,
    load_environment,
    num_tokens_from_string,
)
from utils.file_utils import save_pickle

load_environment()
nest_asyncio.apply()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


collection_name = os.environ["MIMIC3_EHR_COLLECTION_NAME"]
vectorstore = get_vectorstore(collection_name=collection_name)


def check_subject_id_exists(subject_id: int, vectorstore: QdrantVectorStore) -> bool:
    """Check if Subject ID exists in the Vectorstore."""
    subject_id_filter = Filter(
        must=[
            FieldCondition(key="node_kind", match=MatchValue(value=MIMIC_NODE)),
            FieldCondition(key="SUBJECT_ID", match=MatchValue(value=subject_id)),
        ]
    )
    result = vectorstore.scroll(qdrant_filters=subject_id_filter, num_items=1)
    return bool(result)


def decompose_text(
    text: str,
    proposition_type: Literal["sentence", "claim"] = "claim",
    embed_batch_size: int = 32,
    embed_n_jobs: int = 8,
    semantic_threshold: int = 90,
    semantic_chunk_size: int = 128,
    llm_temperature: float = 0.1,
    llm_top_p: float = 1.0,
    llm_n_jobs: int = 8,
    show_progress: bool = True,
) -> list[BaseNode]:
    """Transform text into propositions."""
    # Wrap Text into Document
    node_id = str(create_uuid_from_string(text))
    tokens_ct = num_tokens_from_string(text)
    char_ct = len(text)
    start_char_idx = 0
    end_char_idx = char_ct
    document = Document(
        doc_id=node_id,
        text=text,
        metadata={
            "num_tokens": tokens_ct,
            "created_at": get_utc_time(),
        },
        start_char_idx=start_char_idx,
        end_char_idx=end_char_idx,
    )

    # Get Embedding Model
    embed_model: M3Embedding = get_embed_model(
        embed_batch_size=embed_batch_size,
        num_workers=embed_n_jobs,
        metadata_mode=MetadataMode.NONE,
    )

    # Document --> Semantic Nodes
    semantic_node_parser: SemanticSplitterNodeParser = get_semantic_node_parser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=semantic_threshold,
        max_chunk_size=semantic_chunk_size,
    )
    semantic_nodes = semantic_node_parser([document], show_progress=show_progress)
    semantic_nodes = sorted(semantic_nodes, key=lambda x: x.metadata["created_at"])
    semantic_nodes = embed_model(semantic_nodes, show_progress=show_progress)
    logger.debug(f"Num Semantic Nodes: {len(semantic_nodes)}")

    if proposition_type == "sentence":
        # Semantic Nodes --> Sentence Nodes
        single_sentence_parser: SingleSentenceNodeParser = get_single_sentence_node_parser()
        sentence_nodes = single_sentence_parser(semantic_nodes, show_progress=show_progress)
        sentence_nodes = sorted(sentence_nodes, key=lambda x: x.metadata["created_at"])
        sentence_nodes = embed_model(sentence_nodes, show_progress=show_progress)
        logger.debug(f"Num Sentence Nodes: {len(sentence_nodes)}")
        return sentence_nodes
    elif proposition_type == "claim":
        # Semantic Nodes --> Atomic Claim Nodes
        loop = get_event_loop()
        llm = get_llm(temperature=llm_temperature, top_p=llm_top_p)
        atomic_claim_node_parser: AtomicClaimNodeParser = get_atomic_claim_node_parser(
            num_workers=llm_n_jobs, llm=llm
        )
        claim_nodes = loop.run_until_complete(
            atomic_claim_node_parser.acall(semantic_nodes, show_progress=show_progress)
        )
        claim_nodes = sorted(claim_nodes, key=lambda x: x.metadata["created_at"])
        claim_nodes = embed_model(claim_nodes, show_progress=show_progress)
        logger.debug("Num Claim Nodes: ", len(claim_nodes))
        return claim_nodes
    else:
        raise ValueError(
            "Invalid proposition type. Please choose either 'sentence' or 'atomic_claim'."
        )


def run_verifact(
    text: str = "",
    text_file: Path | str | os.PathLike = "",
    subject_id: int = 1084,
    author_type: Literal["llm", "human"] = "llm",
    proposition_type: Literal["sentence", "claim"] = "claim",
    reference_collection_name: str = "",
    retrieval_method: Literal["dense", "hybrid", "rerank"] = "rerank",
    top_n: int = 50,
    reference_format: Literal["score", "absolute_time", "relative_time"] = "absolute_time",
    hadm_start: pd.Timestamp | str = "2100-01-01",
    hadm_end: pd.Timestamp | str = "2200-01-01",
    apply_time_range: bool = True,
    embed_batch_size: int = 32,
    embed_workers: int = 8,
    semantic_threshold: int = 90,
    semantic_chunk_size: int = 128,
    llm_temperature: float = 0.1,
    llm_top_p: float = 1.0,
    llm_workers: int = 8,
    show_progress: bool = True,
) -> ScoreReport:
    """Run VeriFact on input text by decompose text into propositions, retrieving relevant
    facts and then evaluating using LLM-as-a-Judge."""
    # Ensure only one text input is provided
    if not text and not text_file:
        raise ValueError("Please provide either `text` or `text_file`.")
    elif text and text_file:
        raise ValueError("Please provide only one of `text` or `text_file`.")
    elif text_file:
        text = Path(text_file).read_text()

    # Vector Database
    if not reference_collection_name:
        reference_collection_name = os.environ["MIMIC3_EHR_COLLECTION_NAME"]
    vectorstore = get_vectorstore(collection_name=collection_name)

    # Check that Subject ID argument is provided
    if not subject_id:
        raise ValueError("Please provide a Subject ID.")
    # Check if Subject ID exists in Vectorstore
    if not check_subject_id_exists(subject_id, vectorstore):
        raise ValueError(f"Subject ID {subject_id} does not exist in Vectorstore.")

    # Decompose Text into Proposition Nodes
    text_nodes = decompose_text(
        text=text,
        proposition_type=proposition_type,
        embed_batch_size=embed_batch_size,
        embed_n_jobs=embed_workers,
        semantic_threshold=semantic_threshold,
        semantic_chunk_size=semantic_chunk_size,
        llm_temperature=llm_temperature,
        llm_top_p=llm_top_p,
        llm_n_jobs=llm_workers,
        show_progress=show_progress,
    )
    # Associate Subject ID to each Node (Required by Judge)
    for n in text_nodes:
        n.metadata["SUBJECT_ID"] = subject_id
    texts_df = nodes_to_dataframe(text_nodes)

    # Create LLM-as-a-Judge
    judge = JudgeSingleSubject.from_defaults(
        input_df=texts_df,
        subject_id=subject_id,
        hadm_start=hadm_start,
        hadm_end=hadm_end,
        author_type=author_type,
        node_kind=proposition_type,
        reference_collection_name=reference_collection_name,
        retrieval_method=retrieval_method,
        top_n=top_n,
        reference_format=reference_format,
        reference_only_admission=apply_time_range,
        deduplicate_text=True,
        async_mode=True,
        timeout=300,
    )
    # Evaluate Proposition Node Text
    score_report: ScoreReport = judge.evaluate(show_progress=True)

    # Add Judge Rater Name and Subject ID to Score Report verdict metadata
    judge_config_str = judge.config.to_config_str()
    rater_name = ",".join(judge_config_str.split(",")[3:])
    for v in score_report.verdicts:
        v.metadata_ = {
            "judge_config_str": judge_config_str,
            "rater_name": rater_name,
            **judge.config.model_dump(),
        }
    return score_report


def main(
    text: Annotated[
        str, typer.Option(help="Input text to evaluate (choose either `text` or `text_file`).")
    ] = "",
    text_file: Annotated[
        str, typer.Option(help="Input text file (choose either `text` or `text_file`).")
    ] = "",
    subject_id: Annotated[int, typer.Option(help="Subject ID identifying patient.")] = 1084,
    author_type: Annotated[
        str, typer.Option(help="Author type for input text ('llm', 'human').")
    ] = "llm",
    proposition_type: Annotated[
        str,
        typer.Option(
            help="Proposition type to use for decomposing text for evaluation "
            "and to retrieve from EHR database ('claim', 'sentence')."
        ),
    ] = "claim",
    reference_collection_name: Annotated[
        str, typer.Option(help="Vector database collection name for EHR Facts.")
    ] = "",
    retrieval_method: Annotated[
        str, typer.Option(help="Retrieval Method ('dense', 'hybrid', 'rerank')")
    ] = "rerank",
    top_n: Annotated[
        int, typer.Option(help="Number of EHR Facts to retrieve from vector database.")
    ] = 50,
    reference_format: Annotated[
        str,
        typer.Option(
            help="Reference Context Format for LLM-as-a-Judge "
            "('score', 'absolute_time', 'relative_time')."
        ),
    ] = "absolute_time",
    hadm_start: Annotated[
        str, typer.Option(help="Hospital Admission Start (YYYY-MM-DD)")
    ] = "2100-01-01",
    hadm_end: Annotated[
        str, typer.Option(help="Hospital Admission End (YYYY-MM-DD)")
    ] = "2200-01-01",
    apply_time_range: Annotated[
        bool,
        typer.Option(
            help="Whether to limit retrieval of EHR Facts to the time range defined by "
            "`hadm_start` and `hadm_end`. If False, no time range filter is applied."
        ),
    ] = True,
    embed_batch_size: Annotated[int, typer.Option(help="Embedding batch size.")] = 32,
    embed_workers: Annotated[
        int,
        typer.Option(
            help="Number of async workers dispatching embedding jobs to embedding service."
        ),
    ] = 8,
    semantic_threshold: Annotated[
        int, typer.Option(help="Threshold for determining when to generate a text chunk split.")
    ] = 90,
    semantic_chunk_size: Annotated[
        int, typer.Option(help="Maximum token size of text chunks allowed.")
    ] = 128,
    llm_temperature: Annotated[float, typer.Option(help="LLM temperature parameter.")] = 0.1,
    llm_top_p: Annotated[float, typer.Option(help="LLM top_p parameter.")] = 1.0,
    llm_workers: Annotated[
        int, typer.Option(help="Number of async workers dispatching LLM jobs to LLM service.")
    ] = 8,
    show_progress: Annotated[bool, typer.Option(help="Whether to show progress bars.")] = True,
    output_file: Annotated[
        str,
        typer.Option(
            help="Path to save result. Valid file extensions are '.pkl', '.feather', or '.csv'. "
            "Specifying '.pkl' will save the raw ScoreReport object to disk. "
            "Specifying '.feather' or '.csv' will transform the ScoreReport to a DataFrame and "
            "save it to the disk."
        ),
    ] = "",
) -> None:
    # Generate Score Report
    score_report: ScoreReport = run_verifact(
        text=text,
        text_file=text_file,
        subject_id=subject_id,
        author_type=author_type,
        proposition_type=proposition_type,
        reference_collection_name=reference_collection_name,
        retrieval_method=retrieval_method,
        top_n=top_n,
        reference_format=reference_format,
        hadm_start=hadm_start,
        hadm_end=hadm_end,
        apply_time_range=apply_time_range,
        embed_batch_size=embed_batch_size,
        embed_workers=embed_workers,
        semantic_threshold=semantic_threshold,
        semantic_chunk_size=semantic_chunk_size,
        llm_temperature=llm_temperature,
        llm_top_p=llm_top_p,
        llm_workers=llm_workers,
        show_progress=show_progress,
    )
    # Save Score Report to Disk
    output_file = Path(output_file)
    if output_file.suffix == ".pkl":
        save_pickle(score_report, filepath=output_file)
    elif output_file.suffix == ".feather" or output_file.suffix == ".csv":
        score_report.save(output_file)
    else:
        raise ValueError(
            "Invalid output file extension. Please specify either '.pkl', '.feather', or '.csv'."
        )


if __name__ == "__main__":
    typer.run(main)

# %%
