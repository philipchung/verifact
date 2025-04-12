from typing import Self

import pandas as pd
from pydantic import BaseModel, ConfigDict

from rag.schema import BaseNode, NodeWithScore


class Proposition(BaseModel):
    """A simple unit of text (e.g. claim or sentence) parsed from larger body of text.

    Field names in uppercase are specific to the MIMIC-III dataset.
    Field names in lowercase are derived from LlamaIndex or metadata.
    Field names specific to scores are only available for scored nodes (e.g. retrieval)
        and will be None for nodes without scores.
    """

    model_config = ConfigDict(populate_by_name=True, arbitrary_types_allowed=True)
    # Node Parser-Generated Fields
    node_id: str
    node_monotonic_idx: int | None = None
    node_kind: str
    TEXT: str
    parent_text: str | None = None
    parent_chunk_id: str | None = None
    source_document_id: str | None = None
    num_characters: int | None = None
    num_tokens: int | None = None
    num_sentences: int | None = None
    relationships: dict | None = None
    # MIMIC Specific Fields
    ROW_ID: int | None = None
    SUBJECT_ID: int | None = None
    HADM_ID: float | None = None
    CHARTDATE: pd.Timestamp | None = None
    CHARTTIME: pd.Timestamp | None = None
    STORETIME: pd.Timestamp | None = None
    CATEGORY: str | None = None
    DESCRIPTION: str | None = None
    CGID: int | None = None
    # Scored/Retrieved Node Fields
    score: float | None = None
    dense_score: float | None = None
    dense_normalized_score: float | None = None
    sparse_score: float | None = None
    sparse_normalized_score: float | None = None
    fusion_score: float | None = None

    @classmethod
    def from_node(cls, node) -> Self:
        n = node.node if isinstance(node, NodeWithScore) else node
        return cls(
            # Node Parser-Generated Fields
            node_id=n.node_id,
            node_monotonic_idx=n.metadata.get("node_monotonic_idx"),
            node_kind=n.metadata.get("node_kind"),
            TEXT=n.text,
            parent_text=n.metadata.get("parent_text"),
            parent_chunk_id=n.metadata.get("parent_chunk_id"),
            source_document_id=n.metadata.get("source_document_id"),
            num_characters=n.metadata.get("num_characters"),
            num_tokens=n.metadata.get("num_tokens"),
            num_sentences=n.metadata.get("num_sentences"),
            relationships={k: v.to_dict() for k, v in n.relationships.items()},
            # MIMIC Specific Fields
            ROW_ID=n.metadata.get("ROW_ID"),
            SUBJECT_ID=n.metadata.get("SUBJECT_ID"),
            HADM_ID=n.metadata.get("HADM_ID"),
            CHARTDATE=pd.to_datetime(n.metadata.get("CHARTDATE")),
            CHARTTIME=pd.to_datetime(n.metadata.get("CHARTTIME")),
            STORETIME=pd.to_datetime(n.metadata.get("STORETIME")),
            CATEGORY=n.metadata.get("CATEGORY"),
            DESCRIPTION=n.metadata.get("DESCRIPTION"),
            CGID=n.metadata.get("CGID"),
            # Scored/Retrieved Node Fields
            score=node.get_score() if isinstance(node, NodeWithScore) else None,
            dense_score=n.metadata.get("dense_score"),
            dense_normalized_score=n.metadata.get("dense_normalized_score"),
            sparse_score=n.metadata.get("sparse_score"),
            sparse_normalized_score=n.metadata.get("sparse_normalized_score"),
            fusion_score=n.metadata.get("fusion_score"),
        )


def nodes_to_proposition(nodes: list[BaseNode]) -> list[Proposition]:
    return [Proposition.from_node(n) for n in nodes]


def nodes_to_dataframe(nodes: list[BaseNode]) -> pd.DataFrame:
    rudimentary_units = nodes_to_proposition(nodes)
    return pd.DataFrame(ru.model_dump() for ru in rudimentary_units)


def node_dataframe_to_propositions_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Formats Text Nodes DataFrame to Propositions DataFrame."""
    return df.rename(
        columns={
            "node_id": "proposition_id",
            "TEXT": "text",
            "parent_text": "parent_text_chunk",
            "SUBJECT_ID": "subject_id",
            "ROW_ID": "row_id",
            "HADM_ID": "hadm_id",
        }
    ).loc[
        :,
        [
            "proposition_id",
            "text",
            "parent_text_chunk",
            "subject_id",
            "row_id",
            "hadm_id",
        ],
    ]
