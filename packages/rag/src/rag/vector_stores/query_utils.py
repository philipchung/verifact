from typing import Any, cast

from llama_index.core.vector_stores.types import VectorStoreQueryResult
from qdrant_client import models as rest
from qdrant_client.http.models import Payload

from rag.schema import NodeWithScore
from rag.vector_stores.payload_utils import qdrant_payload_dict_to_node


def vector_store_query_result_to_node_with_score(
    query_result: VectorStoreQueryResult,
) -> list[NodeWithScore]:
    """
    Convert LlamaIndex VectorStoreQueryResult to LlamaIndex list[NodeWithScore].

    Args:
        query_result (VectorStoreQueryResult): VectorStoreQueryResult object.

    Returns:
        list[NodeWithScore]: List of NodeWithScore objects.
    """
    nodes = query_result.nodes
    scores = query_result.similarities
    # If search performed using qdrant scroll, scores will be None
    if scores is None:
        scores = [None] * len(nodes)

    node_with_scores = []
    for node, score in zip(nodes, scores, strict=False):
        node_with_scores += [NodeWithScore(node=node, score=score)]
    return node_with_scores


def qdrant_query_response_to_vector_store_query_result(
    response: list[Any],
) -> VectorStoreQueryResult:
    """
    Convert Qdrant client's vector store query response to
    LlamaIndex VectorStoreQueryResult. In this process, we unpack the Qdrant
    Point payload and convert it into a LlamaIndex Node object.

    The response object type depends on vector search vs. scroll.
    Response for vector search is a list of Qdrant ScoredPoint objects
    Response for scroll is a list of Qdrant Record objects

    Args:
        response: List[Any]: List of ScoredPoint or Records.

    Returns:
        VectorStoreQueryResult: VectorStoreQueryResult object. If response
            is a list of ScoredPoint objects, the similarities are populated.
            If response is a list of Record objects, the similarities are
            set to None.
    """
    nodes = []
    similarities = []
    ids = []

    for item in response:
        if isinstance(item, rest.ScoredPoint):
            payload = cast(Payload, item.payload)
            node = qdrant_payload_dict_to_node(payload)
            nodes.append(node)
            similarities.append(item.score)
            ids.append(str(item.id))
        elif isinstance(item, rest.Record):
            payload = cast(Payload, item.payload)
            node = qdrant_payload_dict_to_node(payload)
            nodes.append(node)
            similarities = None
            ids.append(str(item.id))
        else:
            raise ValueError(
                "Unable to parse response. Response type unknown and is not"
                "a list of ScoredPoint or Record objects."
            )

    return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)


def qdrant_query_response_to_node_with_score(
    response: list[rest.ScoredPoint | rest.Record],
) -> list[NodeWithScore]:
    """
    Convert qdrant query response (list of Qdrant ScoredPoints or Record)
    to list of LlamaIndex NodeWithScore.

    Args:
        response (list[Any]): List of Qdrant ScoredPoint or Record objects.

    Returns:
        list[NodeWithScore]: List of NodeWithScore objects.
    """
    query_result = qdrant_query_response_to_vector_store_query_result(response)
    return vector_store_query_result_to_node_with_score(query_result)
