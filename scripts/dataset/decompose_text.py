# %% [markdown]
# ## Transform Text into LlamaIndex Nodes (+/- ingest to VectorDB)
#
# Running this script requires the following resources to be available:
# * LLM Inference Service
# * Embedding Inference Service
# * Redis Service (to back Redis Queue)
# * Qdrant Vector Database Service
#
# Depending on script settings, you can specify ingestion of nodes to VectorDB
# or you can save the nodes as serialized python pickle files, or both.
#
# NOTE: If there duplicate clinical notes (exact text match) in the dataset to ingest,
# the script will ingest the first note and skip the rest. Ths is because the script
# checks the VectorDB to see if the note's document ID is already present, and the
# note's document ID is deterministically generated as a hash of the clinical note text.
# %%
import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import typer
from llama_index.core.schema import MetadataMode
from rag import CLAIM_NODE, MIMIC_NODE, SEMANTIC_NODE, SENTENCE_NODE
from rag.components import get_embed_model, get_vectorstore
from rag.embedding import M3Embedding
from rag.node_parser import (
    AtomicClaimNodeParser,
    SemanticSplitterNodeParser,
    SingleSentenceNodeParser,
)
from rag.readers import MIMICIIINoteReader
from rag.schema import Document, NodeWithScore, TextNode
from rag.schema.base_node import EmbeddingItem
from rag.vector_stores import nodes_to_dataframe
from rag.vector_stores.qdrant import QdrantVectorStore
from rq import Queue, Retry
from rq_utils import (
    block_and_accumulate_results,
    enqueue_jobs,
    get_queue,
    get_redis,
    shutdown_all_workers,
    start_workers,
)
from utils import flatten_list_of_list, get_local_time, get_utc_time, load_environment, save_pandas
from utils import save_pickle as _save_pickle

load_environment()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class DocumentResult:
    document_nodes: list[Document]
    sentence_nodes: list[TextNode]
    semantic_nodes: list[TextNode]
    claim_nodes: list[TextNode]

    def __str__(self) -> str:
        return (
            f"Num Document Nodes: {len(self.document_nodes)}\n"
            f"Num Sentence Nodes: {len(self.sentence_nodes)}\n"
            f"Num Semantic Nodes: {len(self.semantic_nodes)}\n"
            f"Num Claim Nodes: {len(self.claim_nodes)}\n"
            f"Document Node: {self.document_nodes}"
        )

    def __repr__(self) -> str:
        return self.__str__()


def build_nodes_from_document(
    document: Document,
    upsert_db: bool = True,
    collection_name: str = "default",
    embed_batch_size: int = 200,
    semantic_threshold: int = 90,
    semantic_chunk_size: int = 256,
    embed_n_jobs: int = 32,
    llm_n_jobs: int = 16,
    llm_temperature: float = 0.1,
    llm_top_p: float = 1.0,
    show_progress: bool = False,
    load_nodes_from_vectorstore_if_exists: bool = False,
) -> DocumentResult:
    """Pipeline to process each document.
    Extracts and builds sentence, semantic, and claim nodes from a single document.
    Then embed each node and upsert them into Vector Store.
    """
    from qdrant_client.http.models import FieldCondition, Filter, MatchValue
    from rag.components import (
        get_atomic_claim_node_parser,
        get_embed_model,
        get_llm,
        get_semantic_node_parser,
        get_single_sentence_node_parser,
        get_vectorstore,
    )
    from rag.vector_stores.qdrant import QdrantVectorStore
    from utils import load_environment

    load_environment()
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ## Create/Get Vectorstore Collection
    vs: QdrantVectorStore = get_vectorstore(collection_name=collection_name)

    ## Check if Document exists in Vectorstore
    document_qdrant_filters = Filter(
        must=[
            FieldCondition(key="node_kind", match=MatchValue(value=MIMIC_NODE)),
            FieldCondition(key="node_id", match=MatchValue(value=document.node_id)),
        ]
    )
    result: list[NodeWithScore] = vs.scroll(
        query=document.text, qdrant_filters=document_qdrant_filters, num_items=100
    )
    # Document Exists in Vectorstore
    if result:
        logger.info(f"Document {document.node_id} already exists in VectorDB. Skipping Ingestion.")
        if load_nodes_from_vectorstore_if_exists:
            # Query & Retrieve all Sentence Nodes for this Document
            sentence_qdrant_filters = Filter(
                must=[
                    FieldCondition(key="node_kind", match=MatchValue(value=SENTENCE_NODE)),
                    FieldCondition(
                        key="source_document_id", match=MatchValue(value=document.node_id)
                    ),
                ]
            )
            sentence_nodes = vs.scroll(qdrant_filters=sentence_qdrant_filters)
            sentence_nodes = sorted(sentence_nodes, key=lambda x: x.metadata["created_at"])
            # Query & Retrieve all Semantic Nodes for this Document
            semantic_qdrant_filters = Filter(
                must=[
                    FieldCondition(key="node_kind", match=MatchValue(value=SEMANTIC_NODE)),
                    FieldCondition(
                        key="source_document_id", match=MatchValue(value=document.node_id)
                    ),
                ]
            )
            semantic_nodes = vs.scroll(qdrant_filters=semantic_qdrant_filters)
            semantic_nodes = sorted(semantic_nodes, key=lambda x: x.metadata["created_at"])
            # Query & Retrieve all Claim Nodes for this Document
            claim_qdrant_filters = Filter(
                must=[
                    FieldCondition(key="node_kind", match=MatchValue(value=CLAIM_NODE)),
                    FieldCondition(
                        key="source_document_id", match=MatchValue(value=document.node_id)
                    ),
                ]
            )
            claim_nodes = vs.scroll(qdrant_filters=claim_qdrant_filters)
            claim_nodes = sorted(claim_nodes, key=lambda x: x.metadata["created_at"])
        else:
            sentence_nodes = []
            semantic_nodes = []
            claim_nodes = []
        return DocumentResult(
            document_nodes=[document],
            sentence_nodes=sentence_nodes,
            semantic_nodes=semantic_nodes,
            claim_nodes=claim_nodes,
        )
    # Document Does Not Exist in Vectorstore
    else:
        ## Ingest into Vectorstore
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

        # Semantic Nodes --> Sentence Nodes
        single_sentence_parser: SingleSentenceNodeParser = get_single_sentence_node_parser()
        sentence_nodes = single_sentence_parser(semantic_nodes, show_progress=show_progress)
        sentence_nodes = sorted(sentence_nodes, key=lambda x: x.metadata["created_at"])
        sentence_nodes = embed_model(sentence_nodes, show_progress=show_progress)
        logger.debug(f"Num Sentence Nodes: {len(sentence_nodes)}")

        # Semantic Nodes --> Atomic Claim Nodes
        loop = asyncio.get_event_loop()
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

        # Upsert Parsed Nodes
        if upsert_db:
            vs.add(sentence_nodes)
            vs.add(semantic_nodes)
            vs.add(claim_nodes)
            vs.add([document])  # Upsert Original Note Text w/o Embeddings

        return DocumentResult(
            document_nodes=[document],
            sentence_nodes=sentence_nodes,
            semantic_nodes=semantic_nodes,
            claim_nodes=claim_nodes,
        )


def main(
    dataset_dir: str = typer.Option(default=None, help="Path to dataset directory."),
    input_file: str = typer.Option(
        default="ehr_noteevents.feather",
        help="Filename for input notes dataframe (.csv, .feather, or .parquet)",
    ),
    upsert_db: bool = typer.Option(default=True, help="Whether to upsert nodes into Vectorstore."),
    collection_name: str = typer.Option(default="default", help="Name of Vectorstore collection."),
    save_pickle: bool = typer.Option(default=False, help="Whether to save nodes as pickle files."),
    output_dir_name: str = typer.Option(
        default="output", help="Name of directory to save pickle files."
    ),
    load_nodes_from_vectorstore_if_exists: bool = typer.Option(
        default=False, help="Whether to load nodes from Vectorstore if already exists."
    ),
    num_parallel_pipelines: int = typer.Option(
        default=100, help="Number of notes to ingest in parallel."
    ),
    embed_batch_size: int = typer.Option(default=100, help="Batch size for embedding nodes."),
    semantic_threshold: int = typer.Option(
        default=90, help="Percentile threshold for semantic splitting."
    ),
    semantic_chunk_size: int = typer.Option(
        default=128, help="Max number of tokens allowed per semantic chunk."
    ),
    embed_n_jobs: int = typer.Option(
        default=32, help="Number of concurrent embedding jobs allowed per pipeline."
    ),
    llm_n_jobs: int = typer.Option(
        default=4, help="Number of concurrent LLM jobs allowed per pipeline."
    ),
    llm_temperature: float = typer.Option(default=0.1, help="Temperature for LLM sampling."),
    llm_top_p: float = typer.Option(default=1.0, help="Top-p threshold for LLM sampling."),
    job_timeout: int = typer.Option(
        default=3600 * 24, help="Max duration for each job in seconds."
    ),
    job_result_ttl: int = typer.Option(
        default=3600 * 72, help="How long result is available for each job in seconds."
    ),
    job_polling_interval: int = typer.Option(
        default=5, help="Frequency of polling for job updates in seconds."
    ),
    queue_name: str = typer.Option(default="text_decompose", help="Name of the queue to use."),
) -> None:
    """Decompose MIMIC-III Notes into Atomic Claim and Sentence Nodes
    and store into Vectorstore using parallel async processes."""
    ## Data Paths
    if dataset_dir is None:
        dataset_dir = Path(os.environ["MIMIC3_DATA_DIR"])
    input_path = Path(dataset_dir) / input_file

    ## Load Note Files
    mimic_note_reader = MIMICIIINoteReader()
    note_nodes = mimic_note_reader.load_data(file_path=input_path)
    logger.debug(f"Num MIMIC-III Notes: {len(note_nodes)}")

    ## Create/Get Vectorstore Collection
    vs: QdrantVectorStore = get_vectorstore(collection_name=collection_name)

    # Ensure Qdrant Collection is Created before Launching Multiple Processes
    if not vs._collection_exists(collection_name):
        embed_model: M3Embedding = get_embed_model(
            embed_batch_size=embed_batch_size, num_workers=embed_n_jobs
        )
        embeddings: dict[str, EmbeddingItem] = embed_model.get_text_embedding("test")
        vs._create_collection_from_embeddings(embeddings=embeddings)

    ## Generate Nodes for Each Document
    # Create Job Definitions
    job_creation_str_timestamp = get_utc_time()
    job_datas = []
    for document_node in note_nodes:
        subject_id = document_node.metadata.get("SUBJECT_ID")
        hadm_id = document_node.metadata.get("HADM_ID")
        row_id = document_node.metadata.get("ROW_ID")
        job_data = Queue.prepare_data(
            build_nodes_from_document,
            kwargs={
                "document": document_node,
                "upsert_db": upsert_db,
                "collection_name": collection_name,
                "embed_batch_size": embed_batch_size,
                "semantic_threshold": semantic_threshold,
                "semantic_chunk_size": semantic_chunk_size,
                "embed_n_jobs": embed_n_jobs,
                "llm_n_jobs": llm_n_jobs,
                "llm_temperature": llm_temperature,
                "llm_top_p": llm_top_p,
                "load_nodes_from_vectorstore_if_exists": load_nodes_from_vectorstore_if_exists,
            },
            timeout=job_timeout,
            result_ttl=job_result_ttl,
            job_id=f"ingest-{subject_id}-{hadm_id}-{row_id}-{job_creation_str_timestamp}",
            description=f"Ingest Documents: {subject_id} - {hadm_id}",
            retry=Retry(max=10, interval=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
        )
        job_datas.append(job_data)

    # Place Jobs on Queue
    redis = get_redis()
    q = get_queue(connection=redis, queue=queue_name)
    jobs = enqueue_jobs(job_datas=job_datas, connection=redis, queue=q)

    # Create Workers to Process Jobs
    start_workers(queues=[q], num_workers=num_parallel_pipelines, name=f"{queue_name}-worker")

    # Block Main Thread until Queue is empty and all jobs are completed
    results = block_and_accumulate_results(
        jobs=jobs, polling_interval=job_polling_interval, show_progress=True
    )

    # Shut Down Workers
    shutdown_all_workers(connection=redis, queue=q)

    ## Unpack Results
    all_sentence_nodes = flatten_list_of_list([r.sentence_nodes for r in results])
    all_semantic_nodes = flatten_list_of_list([r.semantic_nodes for r in results])
    all_claim_nodes = flatten_list_of_list([r.claim_nodes for r in results])
    all_document_nodes = flatten_list_of_list([r.document_nodes for r in results])

    ## Save Results to Pickle Files
    if save_pickle:
        if output_dir_name is None:
            output_dir_name = "output"
        output_dir = Path(dataset_dir) / output_dir_name
        output_dir.mkdir(exist_ok=True, parents=True)
        if all_sentence_nodes:
            _save_pickle(obj=all_sentence_nodes, filepath=output_dir / "sentences.pkl")
        if all_semantic_nodes:
            _save_pickle(obj=all_semantic_nodes, filepath=output_dir / "semantic_chunks.pkl")
        if all_claim_nodes:
            _save_pickle(obj=all_claim_nodes, filepath=output_dir / "claims.pkl")
        if all_document_nodes:
            _save_pickle(obj=all_document_nodes, filepath=output_dir / "documents.pkl")

        # Convert Nodes into DataFrame
        sentence_nodes_df = nodes_to_dataframe(all_sentence_nodes)
        save_pandas(sentence_nodes_df, output_dir / "sentences.feather")
        semantic_nodes_df = nodes_to_dataframe(all_semantic_nodes)
        save_pandas(semantic_nodes_df, output_dir / "semantic_chunks.feather")
        claim_nodes_df = nodes_to_dataframe(all_claim_nodes)
        save_pandas(claim_nodes_df, output_dir / "claims.feather")

    utc_timestamp = get_utc_time(output_format="str")
    local_timestamp = get_local_time(output_format="str")
    logger.info(f"Completed Job at: {utc_timestamp} UTC Time ({local_timestamp} Local Time)")


if __name__ == "__main__":
    typer.run(main)
