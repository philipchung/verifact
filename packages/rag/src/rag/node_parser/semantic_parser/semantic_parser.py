"""
Semantic Node Parser that also imposes a maximum chunk length constraint.
Chunks with size larger than the maximum length are recursively split into smaller chunks.

A Semantic Node Parser generates embeddings for each sentence in a document (or a group of
sentences specified by `buffer_size`) and then identifies topical changes in the document
using the embeddings to split the document into chunks.  This may result in chunks
of variable sizes, but each chunk is usually semantically coherent. The idea is that
topics are better parsed out from the document and when they are later embedded these
chunk embeddings are more specifically retrievable from the vectorstore. Also because
retrieved text centers around coherent topics, it is less likely to return text chunks
with irrelevant off-topic information in the chunks.

This is slower than using a naive splitter that splits based on chunk size or sentence
or paragraph boundaries.

Create a LlamaIndex Parser similar to LlamaIndex SemanticSplitter.
Original: llama_index.core.node_parser.semantic_splitter.SemanticSplitter

Here are additional features:
1. Recursive semantic splitting chunks that exceed maximum chunk size with fallback
to non-semantic sentence splitter.
2. Optionally return embedding distances across entire document
3. Computes and adds metadata to each node including character positions, tokens, and
page number from the original document.
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import MetadataMode, NodeRelationship
from llama_index.core.utils import get_tqdm_iterable
from tqdm.auto import tqdm
from utils import (
    convert_nan_nat_to_none,
    convert_timestamps_to_iso,
    flatten_list_of_list,
    get_utc_time,
    num_tokens_from_string,
)

from rag import MIMIC_NODE, SEMANTIC_NODE
from rag.node_parser.node_utils import (
    count_sentences,
    count_tokens,
    nltk_split_sentences,
    node_id_from_string,
)
from rag.node_parser.semantic_parser.semantic_utils import (
    SentenceCombination,
    SplitBundle,
    SplitResult,
    deduplicate_split_results,
)
from rag.schema import BaseNode, Document, EmbeddingItem, TextNode

logger = logging.getLogger(__name__)


class SemanticSplitterNodeParser(NodeParser):
    """Semantic node parser with a maximum chunk length constraint.
    This splits a document into Nodes, with each node being a group of semantically
    related sentences.

    Length constraint is imposed by recursively splitting chunks that
    exceed the maximum length. If recursive splitting fails and chunk size
    remains too large, then we back-off to using LlamaIndex SentenceSplitter
    and then back-off to using LlamaIndex TokenTextSplitter to split the text.

    This node parser uses the embeddings on the `embedding` attribute for each node.

    Original: llama_index.core.node_parser.semantic_splitter.SemanticSplitterNodeParser

    Args:
        sentence_splitter (Optional[Callable]): splits text into sentences. By default,
            this simply uses nltk tokenizer to create sentence units as the first step
            of the semantic splitting process.
        embed_model: (BaseEmbedding): embedding model to use
        buffer_size (int): number of consecutive sentences to group together when
            evaluating semantic similarity. For example, setting this value to 3 will
            generate embeddings and distances using a sliding window of 3 sentences
            across the entire text.
        breakpoint_percentile_threshold (int): percentile of cosine dissimilarity that must be
            exceeded for text to be split into new chunk.
        max_chunk_size (int): maximum number of tokens allowed in a chunk.
        tokenizer_method (str | Callable): Tokenizer to use or a callable function that
            takes a text string argument and returns a token count.
        backup_node_parser (NodeParser): Backup node parser to use when recursive splitting
            fails to decrease text chunk size rapidly. This is used as a fallback.
            Must be a NodeParser object that takes initialization arguments `chunk_size`
            and `chunk_overlap`. Defaults to LlamaIndex SentenceSplitter which splits
            text into chunks based on paragraph and sentence boundaries.
        backup_node_parser_threshold (float): Acceptable token length percentage reduction
            that should be achieved by each split chunk when using semantic splitting.
            If this threshold is exceeded then recursive semantic splitting is not
            chunking the text effectively and we switch to using the `backup_node_parser`
            to further break down the chunk.
        split_if_less_than_max_chunk_size (bool): Whether to split chunks that are less than
            the max chunk size. If True, will apply at least 1 level of semantic splitting
            to all input text nodes. If False, will only split text chunks that exceed
            the max chunk size; text chunks less than max chunks size are not split and
            passed into an output node.
        include_parent_text (bool): Flag for whether to include parent text in metadata.
        id_func (Callable): Function that generates a unique ID for each node.
        node_kind (str): Node category.
    """

    sentence_splitter: Callable[[str], list[str]] | None = Field(
        default_factory=nltk_split_sentences,
        description="The text splitter to use when splitting documents into sentence units "
        "prior to computing embeddings to determine semantic splits.",
        exclude=True,
    )
    embed_model: BaseEmbedding = Field(
        description="The embedding model to use to for semantic comparison",
    )
    buffer_size: int = Field(
        default=3,
        description="The number of sentences to group together when evaluating "
        "semantic similarity. Set to 1 to consider each sentence individually. "
        "Set to >1 to group consecutive sentences together.",
    )
    breakpoint_percentile_threshold: int = Field(
        default=90,
        description="The percentile of cosine dissimilarity that must be exceeded between a "
        "group of sentences and the next to form a node.  The smaller this "
        "number is, the more nodes will be generated. Value should be from 1-99.",
    )
    max_chunk_size: int = Field(
        default=1000,
        description="The maximum number of tokens allowed in a chunk.",
    )
    tokenizer_method: str | Callable = Field(
        default="o200k_base",
        description="Name of tokenizer (e.g. `o200k_base`, or huggingface with `namespace/name`",
    )
    backup_node_parser: NodeParser | None = Field(
        default_factory=lambda: SentenceSplitter(chunk_size=1000, chunk_overlap=0),
        description="Backup node parser to use when semantic parsing is not effective in "
        "generating chunks. Must be a NodeParser object that takes initialization arguments "
        "`chunk_size` and `chunk_overlap`. Defaults to LlamaIndex SentenceSplitter.",
        exclude=True,
    )
    backup_node_parser_threshold: float = Field(
        default=75.0,
        description="Acceptable token length percentage reduction that should be achieved "
        "by each split chunk when using semantic splitting. Value should be from 1-99.",
    )
    split_if_less_than_max_chunk_size: bool = Field(
        default=False,
        description="Whether to split chunks that are less than the max chunk size. "
        "If True, will apply at least 1 level of semantic splitting to the text chunk "
        "regardless of max_chunk_size. If False, text chunks less than max chunks size "
        "are not split and passed directly into an output text node.",
    )
    include_parent_text: bool = Field(
        default=False, description="Flag for whether to include parent text in metadata."
    )
    id_func: Callable[[int, Document], str] = Field(
        default=node_id_from_string,
        description="A function that generates a unique ID for each node."
        "Function must take 3 arguments `i` (int), `node` (BaseNode), and `text` (str)",
        exclude=True,
    )
    node_kind: str = Field(default=SEMANTIC_NODE, description="Node category.")

    @classmethod
    def class_name(cls) -> str:
        return "SemanticSplitterNodeParser"

    @classmethod
    def from_defaults(
        cls,
        sentence_splitter: Callable[[str], list[str]] = nltk_split_sentences,
        embed_model: BaseEmbedding | None = None,
        buffer_size: int = 3,
        breakpoint_percentile_threshold: int = 90,
        max_chunk_size: int = 1000,
        tokenizer_method: str | Callable = "o200k_base",
        backup_node_parser: NodeParser | None = None,
        backup_node_parser_threshold: float = 75.0,
        split_if_less_than_max_chunk_size: bool = False,
        include_parent_text: bool = True,
        node_kind: str = SEMANTIC_NODE,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: CallbackManager | None = None,
        id_func: Callable[[int, Document], str] | None = None,
    ) -> "SemanticSplitterNodeParser":
        # NOTE: Avoid overriding new node metadata with parent metadata in
        # _postprocess_parsed_nodes by setting include_metadata=False.
        # Also don't want to include metadata when considering splits.
        callback_manager = callback_manager or CallbackManager([])
        backup_node_parser = backup_node_parser or SentenceSplitter(
            chunk_size=max_chunk_size, chunk_overlap=0, include_metadata=False
        )
        if embed_model is None:
            try:
                from llama_index.embeddings.openai import (
                    OpenAIEmbedding,
                )

                embed_model = embed_model or OpenAIEmbedding()
            except ImportError as err:
                raise ImportError(
                    "`llama-index-embeddings-openai` package not found, "
                    "please run `pip install llama-index-embeddings-openai`"
                ) from err

        id_func = id_func or node_id_from_string

        return cls(
            sentence_splitter=sentence_splitter,
            embed_model=embed_model,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            max_chunk_size=max_chunk_size,
            split_if_less_than_max_chunk_size=split_if_less_than_max_chunk_size,
            tokenizer_method=tokenizer_method,
            backup_node_parser=backup_node_parser,
            backup_node_parser_threshold=backup_node_parser_threshold,
            include_parent_text=include_parent_text,
            node_kind=node_kind,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )

    def _count_tokens(self, text: str) -> int:
        """Count number of tokens in a text string."""
        return count_tokens(text, self.tokenizer_method)

    def _count_sentences(self, text: str) -> int:
        """Count number of sentences in a text string."""
        return count_sentences(text, self.sentence_splitter)

    def _build_nodes_from_semantic_splits(
        self,
        semantic_splits: list[str],
        original_node: BaseNode,
    ) -> list[BaseNode]:
        """Build nodes from semantic splits."""
        # Create New Node for Each Semantic Split
        output_nodes: list[TextNode] = []
        for i, split in enumerate(semantic_splits):
            node_text = split
            parent_text = original_node.get_content(MetadataMode.NONE)
            # Create Unique ID for New Node based on Source Node ID, New Node Text, and Index
            node_id = self.id_func(f"{original_node.node_id}-{parent_text}-{i}-{node_text}")
            metadata = {
                # Monotonically increasing index (relative position in original node)
                "node_monotonic_idx": i,
                # Node Kind
                "node_kind": self.node_kind,
                # Node Creation Time
                "created_at": get_utc_time(),
                # Metadata from Original Node
                "parent_text": parent_text if self.include_parent_text else None,
                "parent_chunk_id": original_node.node_id,
                "source_document_id": original_node.source_node.node_id
                if original_node.source_node
                else original_node.node_id,
            }
            # Set Parent Node Relationship
            relationships = {NodeRelationship.PARENT: original_node.as_related_node_info()}
            # Set Source Node Relationship
            if isinstance(original_node, Document):
                relationships |= {NodeRelationship.SOURCE: original_node.as_related_node_info()}
            elif original_node.relationships.get(NodeRelationship.SOURCE) is not None:
                relationships |= {
                    NodeRelationship.SOURCE: original_node.relationships[NodeRelationship.SOURCE]
                }

            new_node = TextNode(
                id=node_id,
                text=node_text,
                metadata=metadata,
                relationships=relationships,
                # Carry over properties from original node
                embeddings=original_node.embeddings,
                embedding=original_node.embedding,
                excluded_embed_metadata_keys=original_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=original_node.excluded_llm_metadata_keys,
                metadata_seperator=original_node.metadata_seperator,
                metadata_template=original_node.metadata_template,
                text_template=original_node.text_template,
            )
            output_nodes.append(new_node)
        return output_nodes

    def _parse_nodes(
        self,
        nodes: list[BaseNode],
        show_progress: bool = False,
        return_details: bool = False,
        **kwargs: Any,
    ) -> list[TextNode] | tuple[list[TextNode], list[SplitBundle]]:
        """Parse document into nodes.

        If split_if_less_than_max_chunk_size is True, then we will always split the text
        in each node. If False, we will only split the text if it exceeds the max_chunk_size.

        Args:
            nodes (list[BaseNode]): Input text nodes to parse.
            show_progress (bool, optional): Whether to show progress bar. Defaults to False.
            return_details (bool, optional): Whether to return full output dictionary
                generated by the recursive semantic splitter. Defaults to False.

        Returns:
            If `return_details` is False: list[TextNode]
                This is the list of final nodes generated by the recursive semantic splitter.
                This is the default behavior in LlamaIndex node parsers. When chaining node
                parsers in LlamaIndex the expected output is a list of nodes.

            If `return_details` is True: Tuple[list[TextNode], list[SplitBundle]]
                This is useful for debugging or inspecting the splits.
                The first element in the tuple is the list of final nodes generated by
                the recursive semantic splitter.
                The second element is a list of SplitBundle objects. There is one
                    SplitBundle object per input document.
        """
        output_nodes: list[TextNode] = []
        all_outputs: list[SplitBundle] = []

        # Iterate through each document and create semantic nodes
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")
        for doc_node in nodes_with_progress:
            # Check if Document is Small Enough to be a Single Node or needs to be Split
            doc_tokens_ct = num_tokens_from_string(doc_node.get_content(MetadataMode.NONE))
            if not self.split_if_less_than_max_chunk_size and doc_tokens_ct <= self.max_chunk_size:
                ## Document is Small Enough to be a Single Node
                semantic_node = self._build_nodes_from_semantic_splits(
                    [doc_node.get_content(MetadataMode.NONE)], doc_node
                )[0]
                output_nodes.append(semantic_node)
            else:
                ## Document is Too Large, Split into Semantic Chunks
                # Extract Semantic Text Chunks from Node Text
                sbs: list[SplitBundle] = self._create_semantic_splits([doc_node], show_progress)
                text_chunks = flatten_list_of_list([sb.text_chunks for sb in sbs])
                # Build a New Node for each Semantic Split
                semantic_nodes = self._build_nodes_from_semantic_splits(text_chunks, doc_node)
                output_nodes.extend(semantic_nodes)

            if return_details:
                all_outputs.append(sbs)
        if return_details:
            return output_nodes, all_outputs
        else:
            return output_nodes

    def _postprocess_parsed_nodes(
        self, nodes: list[BaseNode], parent_doc_map: dict[str, Document]
    ) -> list[BaseNode]:
        """Update metadata and relationships for each semantic node.

        Args:
            nodes (list[BaseNode]): List of nodes that have been created by node parser.
            parent_doc_map (dict[str, Document]): List of original documents.

        Returns:
            list[BaseNode]: List of nodes with updated metadata and relationships.
        """
        for i, node in enumerate(nodes):
            parent_doc = parent_doc_map.get(node.parent_node.node_id, None)

            if parent_doc is not None:
                self._update_node_start_end_char_idx(node, parent_doc)

                if self.include_metadata:
                    node = self._update_node_metadata(
                        node,
                        parent_doc,
                        start_char_idx=node.start_char_idx,
                        end_char_idx=node.end_char_idx,
                    )

            if self.include_prev_next_rel:
                self._update_prev_next_relationships(i, node, nodes)
        return nodes

    def _update_node_start_end_char_idx(
        self, node: BaseNode, parent_doc: Document, **kwargs: Any
    ) -> BaseNode:
        """Update the start and end character index for a node based on the parent document."""
        node_text = node.get_content(MetadataMode.NONE)
        start_char_idx = parent_doc.text.find(node_text)
        if start_char_idx >= 0:
            node.start_char_idx = start_char_idx
            node.end_char_idx = start_char_idx + len(node_text)
        return node

    def _update_node_metadata(self, node: BaseNode, parent_doc: Document, **kwargs) -> BaseNode:
        """Update the metadata a node. Page data obtained by referencing
        original document and the character positions for each page."""
        # Update node metadata with new character positions, tokens, pages
        node_text = node.get_content(MetadataMode.NONE)
        node.metadata |= {
            "num_characters": len(node_text),
            "num_tokens": self._count_tokens(node_text),
            "num_sentences": self._count_sentences(node_text),
            "start_char_idx": node.start_char_idx,
            "end_char_idx": node.end_char_idx,
        }

        # If original document is MIMIC-III note, include some original metadata
        source_node_relationship = node.relationships.get(NodeRelationship.SOURCE)
        if (
            source_node_relationship
            and source_node_relationship.metadata.get("node_kind") == MIMIC_NODE
        ):
            node.metadata |= {
                "ROW_ID": source_node_relationship.metadata.get("ROW_ID"),
                "SUBJECT_ID": source_node_relationship.metadata.get("SUBJECT_ID"),
                "HADM_ID": source_node_relationship.metadata.get("HADM_ID"),
                "CHARTDATE": source_node_relationship.metadata.get("CHARTDATE"),
                "CHARTTIME": source_node_relationship.metadata.get("CHARTTIME"),
                "STORETIME": source_node_relationship.metadata.get("STORETIME"),
                "CATEGORY": source_node_relationship.metadata.get("CATEGORY"),
                "DESCRIPTION": source_node_relationship.metadata.get("DESCRIPTION"),
                "CGID": source_node_relationship.metadata.get("CGID"),
            }
            node.metadata = {k: convert_nan_nat_to_none(v) for k, v in node.metadata.items()}
            node.metadata = {k: convert_timestamps_to_iso(v) for k, v in node.metadata.items()}
        return node

    def _update_prev_next_relationships(
        self, i: int, node: BaseNode, nodes: list[BaseNode], **kwargs: Any
    ) -> BaseNode:
        """Update prev/next relationships for a node."""
        # establish prev/next relationships if nodes share the same source_node
        if (
            i > 0
            and node.source_node
            and nodes[i - 1].source_node
            and nodes[i - 1].source_node.node_id == node.source_node.node_id
        ):
            node.relationships[NodeRelationship.PREVIOUS] = nodes[i - 1].as_related_node_info()
        if (
            i < len(nodes) - 1
            and node.source_node
            and nodes[i + 1].source_node
            and nodes[i + 1].source_node.node_id == node.source_node.node_id
        ):
            node.relationships[NodeRelationship.NEXT] = nodes[i + 1].as_related_node_info()
        return node

    def _create_semantic_splits(
        self,
        documents: list[Document],
        show_progress: bool = False,
    ) -> list[SplitBundle]:
        """Split documents using recurisve semantic splitter."""
        # One split bundle created per document
        split_bundles: list[SplitBundle] = []
        for doc in documents:
            # Split text into chunks based on semantic similarity
            split_bundle: SplitBundle = self._recursive_semantic_chunk_text(
                text=doc.text, show_progress=show_progress
            )
            split_bundles.append(split_bundle)
        return split_bundles

    def _recursive_semantic_chunk_text(
        self,
        text: str,
        show_progress: bool = False,
        recursion_depth: int = 0,
        chunk_idx: int = 0,
        split_results: list[SplitResult] | None = None,
    ) -> SplitBundle:
        """Transform text into list of text chunks based on semantic similarity.
        Recursively split large chunks exceeding `self.max_chunk_size` into smaller chunks.
        The resulting chunks of text are returned in-order relative to the original text."""
        # Split text into chunks
        result: SplitResult = self._semantic_chunk_text(text, show_progress=show_progress)
        result.recursion_depth = recursion_depth
        result.chunk_idx = chunk_idx
        result.splitter = "semantic"

        # Store result in split_results
        if split_results is None:
            split_results = []
        split_results.append(result)

        # Check each chunk size; if too large, recursively split into smaller chunks
        text_chunks: list[str] = result.text_chunks
        final_chunks: list = []
        final_chunks_recursion_depth: list = []
        for chunk_idx, text_chunk in tqdm(
            enumerate(text_chunks),
            total=len(text_chunks),
            disable=not show_progress,
            desc="Recursive Semantic Splitting",
        ):
            # Count Tokens in Chunk
            text_chunk_token_ct = self._count_tokens(text_chunk)
            # Base Case: Accept Chunk
            if text_chunk_token_ct < self.max_chunk_size:
                final_chunks.append(text_chunk)
                final_chunks_recursion_depth.append(recursion_depth)
            # Text Chunk Too Large, Recursively Split
            else:
                # Check if infinite recursion and/or chunk size is decreasing too slowly.
                # If yes, switch to backup splitter.
                input_text_token_ct = self._count_tokens(text)
                threshold = self.backup_node_parser_threshold / 100 * input_text_token_ct
                if not text_chunk_token_ct < threshold:
                    # Base Case: Backup Splitter to avoid Infinite Recursion
                    result: SplitResult = self._backup_split_text(text=text_chunk)
                    result.recursion_depth = recursion_depth + 1
                    result.chunk_idx = chunk_idx
                    result.splitter = "sentence"
                    split_results += [result]
                    final_chunks.extend(result.text_chunks)
                    final_chunks_recursion_depth.extend(
                        [recursion_depth + 1] * len(result.text_chunks)
                    )
                # We're not in infinite recursion, so we continue to recursively split
                else:
                    rr: SplitBundle = self._recursive_semantic_chunk_text(
                        text=text_chunk,
                        show_progress=False,  # Suppress recursive splits progress bar
                        recursion_depth=recursion_depth + 1,  # Track recursion depth
                        chunk_idx=chunk_idx,
                        split_results=split_results,
                    )
                    split_results.extend(rr.split_results)
                    final_chunks.extend(rr.text_chunks)
                    final_chunks_recursion_depth.extend(rr.recursion_depths)

        # Deduplicate split results
        # NOTE: One split result created for each recursive split
        split_results: list[SplitResult] = deduplicate_split_results(split_results)
        split_results = sorted(split_results, key=lambda x: (x.recursion_depth, x.chunk_idx))
        return SplitBundle(
            text_chunks=final_chunks,  # list of final text chunks
            recursion_depths=final_chunks_recursion_depth,  # one item per text chunk
            split_results=split_results,  # one item per recursive split
        )

    def _semantic_chunk_text(self, text: str, show_progress: bool = False) -> SplitResult:
        """Transform text into list of text chunks based on semantic similarity."""
        # First split all sentences in the document
        text_splits = self.sentence_splitter(text)
        # Group window of sentences together (defined by buffer_size)
        sentences = self._build_sentence_groups(text_splits)

        # Generate embeddings for each group of sentences
        def get_default_embedding(sentences: list[str]) -> list[list[float]]:
            """Get default embedding for each sentence."""
            embeddings = self.embed_model.get_text_embedding_batch(sentences, show_progress)
            # If multiple embeddings in a dict, get default embedding
            default_embeddings = []
            for embedding in embeddings:
                if isinstance(embedding, dict):
                    default_vector_name = self.embed_model.default_vector_name
                    default_embedding = embedding[default_vector_name]
                    default_embeddings.append(default_embedding)
            # If each embedding is wrapped in EmbeddingItem, unwrap it
            return [e.embedding if isinstance(e, EmbeddingItem) else e for e in default_embeddings]

        combined_sentence_embeddings = get_default_embedding(
            [s["combined_sentence"] for s in sentences]
        )

        for i, embedding in enumerate(combined_sentence_embeddings):
            sentences[i]["combined_sentence_embedding"] = embedding

        # Compute distances between adjacent sentence group embeddings
        distances: list[float] = self._calculate_distances_between_sentence_groups(sentences)
        # Split document into chunks based on semantic similarity
        text_chunks: list[str] = self._build_node_chunks(sentences, distances)
        return SplitResult(input_text=text, text_chunks=text_chunks, distances=distances)

    def _backup_split_text(self, text: str) -> SplitResult:
        """Transform text into list of text chunks using backup node parser."""
        text_chunks = self.backup_node_parser.split_text(text)
        # If backup node parser fails to split, then try again with SentenceSplitter
        # using half the max target chunk size to ensure a split.
        if len(text_chunks) == 1:
            backup_node_parser = SentenceSplitter(
                chunk_size=int(self.max_chunk_size / 2), chunk_overlap=0
            )
            text_chunks = backup_node_parser.split_text(text)
        return SplitResult(input_text=text, text_chunks=text_chunks)

    def _build_sentence_groups(self, text_splits: list[str]) -> list[SentenceCombination]:
        sentences: list[SentenceCombination] = [
            {
                "sentence": x,
                "index": i,
                "combined_sentence": "",
                "combined_sentence_embedding": [],
            }
            for i, x in enumerate(text_splits)
        ]

        # Group consecutive sentences defined by buffer_size
        for i in range(len(sentences)):
            combined_sentence = ""

            for j in range(i - self.buffer_size, i):
                if j >= 0:
                    combined_sentence += sentences[j]["sentence"]

            combined_sentence += sentences[i]["sentence"]

            for j in range(i + 1, i + 1 + self.buffer_size):
                if j < len(sentences):
                    combined_sentence += sentences[j]["sentence"]

            sentences[i]["combined_sentence"] = combined_sentence

        return sentences

    def _calculate_distances_between_sentence_groups(
        self, sentences: list[SentenceCombination]
    ) -> list[float]:
        distances = []
        for i in range(len(sentences) - 1):
            embedding_current = sentences[i]["combined_sentence_embedding"]
            embedding_next = sentences[i + 1]["combined_sentence_embedding"]
            similarity = self.embed_model.similarity(embedding_current, embedding_next)
            distance = 1 - similarity
            distances.append(distance)
        return distances

    def _build_node_chunks(
        self, sentences: list[SentenceCombination], distances: list[float]
    ) -> list[str]:
        "Create text chunks based on semantic dissimiliarity from embedding distances."
        chunks = []
        if len(distances) > 0:
            breakpoint_distance_threshold = np.percentile(
                distances, self.breakpoint_percentile_threshold
            )

            indices_above_threshold = [
                i for i, x in enumerate(distances) if x > breakpoint_distance_threshold
            ]

            # Chunk sentences into semantic groups based on percentile breakpoints
            start_index = 0

            for index in indices_above_threshold:
                group = sentences[start_index : index + 1]
                combined_text = "".join([d["sentence"] for d in group])
                chunks.append(combined_text)

                start_index = index + 1

            if start_index < len(sentences):
                combined_text = "".join([d["sentence"] for d in sentences[start_index:]])
                chunks.append(combined_text)
        else:
            # If, for some reason we didn't get any distances (i.e. very, very small documents) just
            # treat the whole document as a single node
            chunks = [" ".join([s["sentence"] for s in sentences])]

        return chunks
