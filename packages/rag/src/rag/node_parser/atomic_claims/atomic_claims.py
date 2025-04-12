"""
Atomic Claim node parser transforms nodes of text into nodes of atomic claims.

For each text node, LLM is applied to the text to extract atomic claims.
Each atomic claim is wrapped in a new node. New nodes are returned with
metadata derived from the original text node and relationship back to the text node.
"""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from llama_index.core.async_utils import run_jobs
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import MetadataMode, NodeRelationship
from llama_index.core.utils import get_tqdm_iterable
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_utils import LLMBaseModel, SimpleClaimList
from utils import convert_nan_nat_to_none, convert_timestamps_to_iso, get_utc_time

from rag import CLAIM_NODE, MIMIC_NODE
from rag.llms.openai_like import OpenAILike
from rag.node_parser.atomic_claims.prompts import (
    prompt_contains_atomic_claims,
    prompt_extract_atomic_claims,
    system_prompt_extract_atomic_claims,
)
from rag.node_parser.node_utils import (
    count_sentences,
    count_tokens,
    mimic_node_id_fn,
)
from rag.schema import BaseNode, Document, TextNode


class ContainsAtomicClaim(BaseModel):
    contains_atomic_claim: bool


class AtomicClaimNodeParser(NodeParser, LLMBaseModel):
    """Atomic Claim node parser transforms nodes of text into nodes of atomic claims.

    NOTE: Text length is limited by the LLM's output max length (this is the maximum
    tokens LLMs can generate and not the maximum input context window size).
    * For most close-source models like GPT, Claude, Gemini, the maximum output token
    length is 4k tokens.
    * For open-source models like Llama-3, there is no hard limit on maximum output token
    length: maximum output tokens = total context window - input tokens.

    In general, atomic claims that are generated from text are more verbose than the
    original text, so recommend input text length be 25-33% of the output token length.
    """

    llm: OpenAILike | OpenAI = Field(description="LLM model to use for atomic claim extraction.")
    num_workers: int = Field(
        default=16, description="Number of nodes to parse to claims in parallel."
    )
    num_invalid_output_retries: int = Field(
        default=3,
        description="Number of attempts to use LLM to correct output.",
    )
    system_prompt_template: Callable = Field(
        default=system_prompt_extract_atomic_claims,
        description="System prompt function for atomic claim extraction. "
        "Overrides default LLM system prompt.",
    )
    detect_claims_template: Callable = Field(
        default=prompt_contains_atomic_claims,
        description="Message prompt function for checking if atomic claim is in text. "
        "Accepts single argument `text`.",
    )
    extract_claims_template: Callable = Field(
        default=prompt_extract_atomic_claims,
        description="Message prompt function for atomic claim extraction. "
        "Accepts single argument `text`.",
    )
    include_parent_text: bool = Field(
        default=True, description="Flag for whether to include parent text in metadata."
    )
    node_kind: str = Field(default=CLAIM_NODE, description="Node category.")

    @classmethod
    def class_name(cls) -> str:
        return "AtomicClaimNodeParser"

    @classmethod
    def from_defaults(
        cls,
        llm: OpenAILike | OpenAI | None = None,
        system_prompt_template: Callable = system_prompt_extract_atomic_claims,
        detect_claims_template: Callable = prompt_contains_atomic_claims,
        extract_claims_template: Callable = prompt_extract_atomic_claims,
        include_parent_text: bool = True,
        num_workers: int = 8,
        num_invalid_output_retries: int = 3,
        node_kind: str = CLAIM_NODE,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: CallbackManager | None = None,
        id_func: Callable[[int, Document], str] | None = None,
    ) -> "AtomicClaimNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        id_func = id_func or mimic_node_id_fn
        return cls(
            llm=llm,
            system_prompt_template=system_prompt_template,
            detect_claims_template=detect_claims_template,
            extract_claims_template=extract_claims_template,
            include_parent_text=include_parent_text,
            num_workers=num_workers,
            num_invalid_output_retries=num_invalid_output_retries,
            node_kind=node_kind,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
            id_func=id_func,
        )

    def _count_tokens(self, text: str) -> int:
        """Count number of tokens in a text string."""
        return count_tokens(text)

    def _count_sentences(self, text: str) -> int:
        """Count number of sentences in a text string."""
        return count_sentences(text)

    def _determine_contains_atomic_claim(self, text: str) -> bool:
        """Determines if text contains atomic claims using LLM."""
        obj = self._generate(
            text, prompt=self.detect_claims_template, response_format=ContainsAtomicClaim
        )
        contains_atomic_claim: bool = obj.contains_atomic_claim
        return contains_atomic_claim

    def _extract_claims_list(self, text: str) -> list[str]:
        """Generates atomic claims from text using LLM."""
        obj = self._generate(
            text, prompt=self.extract_claims_template, response_format=SimpleClaimList
        )
        if obj is None or obj.items is None:
            claims: list[str] = []
        else:
            claims: list[str] = obj.items
        return claims

    async def _a_determine_contains_atomic_claim(self, text: str) -> Awaitable[bool]:
        """Determines if text contains atomic claims using LLM."""
        obj = await self._a_generate(
            text, prompt=self.detect_claims_template, response_format=ContainsAtomicClaim
        )
        contains_atomic_claim: bool = obj.contains_atomic_claim
        return contains_atomic_claim

    async def _a_extract_claims_list(self, text: str) -> Awaitable[list[str]]:
        """Generates atomic claims from text using LLM."""
        obj = await self._a_generate(
            text, prompt=self.extract_claims_template, response_format=SimpleClaimList
        )
        if obj is None or obj.items is None:
            claims: list[str] = []
        else:
            claims: list[str] = obj.items
        return claims

    def _build_nodes_from_claims_list(
        self, claims: list[str], original_node: BaseNode, **kwargs
    ) -> list[BaseNode]:
        """Builds nodes from claims list."""
        # Create New Node for Each Claim
        output_nodes = []
        for i, claim in enumerate(claims):
            node_text = claim
            parent_text = original_node.get_content(MetadataMode.NONE)
            # Create Unique ID for New Node - This will be unique for each patient note,
            # author_type, proposition_type/node_kind, and extracted text
            node_id = self.id_func(
                text=node_text,
                subject_id=kwargs.get("subject_id", ""),
                row_id=kwargs.get("row_id", ""),
                author_type=kwargs.get("author_type", ""),
                node_kind=getattr(self, "node_kind", ""),
            )
            metadata = {
                # Monotonically increasing index (relative position in original node)
                "node_monotonic_idx": i,
                # Node Kind
                "node_kind": self.node_kind,
                # Node Creation Time
                "created_at": get_utc_time(),
                # Info about LLM Generating Claim
                "claim_generation_llm": self.llm.model,
                "claim_generation_llm_config": {
                    "model_name": self.llm.model,
                    "context_window": self.llm.context_window,
                    "system_prompt": self.llm.system_prompt,
                    "temperature": self.llm.temperature,
                    **self.llm.additional_kwargs,
                },
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
        self, nodes: Sequence[TextNode], show_progress: bool = False, **kwargs: Any
    ) -> list[TextNode]:
        """Parses input nodes into atomic claims and creates a new node for each claim."""
        queue_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing Node to Claims")
        # Drop Empty Nodes
        nodes = [node for node in nodes if node.get_content(MetadataMode.NONE)]

        output_nodes = []
        for node in queue_with_progress:
            # Determine if Node Text has Atomic Claims
            node_text = node.get_content(MetadataMode.NONE)
            contains_atomic_claim: bool = self._determine_contains_atomic_claim(node_text)
            # Extract Claims from Node Text
            if contains_atomic_claim:
                claims: list[str] = self._extract_claims_list(node_text)
                # Build a New Node for each Claim
                claim_nodes = self._build_nodes_from_claims_list(claims, node, **kwargs)
                output_nodes.extend(claim_nodes)
        return output_nodes

    async def _aparse_nodes(
        self, nodes: Sequence[TextNode], show_progress: bool = False, **kwargs: Any
    ) -> Awaitable[list[TextNode]]:
        """Parses input nodes into atomic claims and creates a new node for each claim."""
        workers = kwargs.get("workers", self.num_workers)

        async def job(node) -> Awaitable[list[TextNode]]:
            # Determine if Node Text has Atomic Claims
            node_text = node.get_content(MetadataMode.NONE)
            contains_atomic_claim: bool = await self._a_determine_contains_atomic_claim(node_text)
            # Extract Claims from Node Text
            if contains_atomic_claim:
                claims: list[str] = await self._a_extract_claims_list(node_text)
                # Build a New Node for each Claim
                claim_nodes = self._build_nodes_from_claims_list(claims, node, **kwargs)
            else:
                claim_nodes = []
            return claim_nodes

        nodes = [node for node in nodes if node.get_content(MetadataMode.NONE)]
        jobs = [job(node) for node in nodes]
        results = await run_jobs(
            jobs, show_progress=show_progress, workers=workers, desc="Parsing Node to Claims"
        )
        output_nodes = [node for result in results for node in result]
        return output_nodes

    def _postprocess_parsed_nodes(
        self, nodes: list[BaseNode], parent_doc_map: dict[str, Document]
    ) -> list[BaseNode]:
        """Update metadata and relationships for each node.

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
        # NOTE: claims are generated and not split from original text, so usually
        # we will not have start/end char idx that matches.
        node.start_char_idx = None
        node.end_char_idx = None
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
            "current_page": None,
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
