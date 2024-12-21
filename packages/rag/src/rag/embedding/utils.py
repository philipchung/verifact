import warnings
from collections import defaultdict
from typing import Any, NamedTuple, TypeAlias

import numpy as np

from rag.schema import EmbeddingItem

# NOTE: NamedEmbedding supports regular un-named dense embedding (list of float)
# which is default for Llama-Index, but also supports named multivector embedding
# which is a dict with a string name key and an embedding.  The embedding can
# be dense (list of float) or it can be sparse (dict with token indices and lexical weight values),
# or it can be `None`, which means we disable the embedding or don't generate that embedding.
# The NamedEmbedding format is similar to qdrant's named vectors.
DenseEmbedding: TypeAlias = list[float]
SparseEmbedding: TypeAlias = dict[int, float]
NamedEmbedding: TypeAlias = (
    DenseEmbedding | dict[str, EmbeddingItem | DenseEmbedding | SparseEmbedding | None]
)


class FlatEmbedding(NamedTuple):
    name: str
    kind: str
    value: Any


def unpack_embedding(embedding: NamedEmbedding) -> list[FlatEmbedding]:
    """Unpacks all embeddings for item into flat list of tuples with name, kind and value."""
    unpacked_embeddings = []
    if isinstance(embedding, list):
        item = FlatEmbedding(name="dense", kind="dense", value=embedding)
        unpacked_embeddings.append(item)
    elif isinstance(embedding, dict):
        for key, value in embedding.items():
            if isinstance(value, EmbeddingItem):
                item = FlatEmbedding(name=value.name, kind=value.kind, value=value.embedding)
                unpacked_embeddings.append(item)
            elif isinstance(value, DenseEmbedding):
                item = FlatEmbedding(name=key, kind="dense", value=value)
                unpacked_embeddings.append(item)
            elif isinstance(value, SparseEmbedding):
                item = FlatEmbedding(name=key, kind="sparse", value=value)
                unpacked_embeddings.append(item)
            elif value is None:
                warnings.warn(
                    f"Embedding {key} has value of None. Embedding skipped.",
                    stacklevel=1,
                )
            else:
                warnings.warn(
                    f"Cannot unpack embedding with unknown type: {type(value)}. "
                    "Embedding skipped.",
                    stacklevel=1,
                )
    else:
        raise TypeError(f"Cannot unpack embeddings with unknown type: {type(embedding)}.")
    return unpacked_embeddings


def unpack_embeddings(embeddings: list[NamedEmbedding]) -> list[FlatEmbedding]:
    """Unpacks all embeddings into flat list of tuples with name, kind and value."""
    unpacked_embeddings = []
    for embedding in embeddings:
        unpacked_embeddings.extend(unpack_embedding(embedding))
    return unpacked_embeddings


def mean_agg_dense(embeddings: list[DenseEmbedding]) -> DenseEmbedding:
    "Aggregates many dense embeddings into one."
    return list(np.array(embeddings).mean(axis=0))


def mean_agg_sparse(embeddings: list[SparseEmbedding]) -> SparseEmbedding:
    "Aggregates many sparse embeddings into one."
    accumulate_embeddings = {}
    # Accumulate all sparse embedding key-value pairs
    for embedding in embeddings:
        for key, value in embedding.items():
            if key not in accumulate_embeddings:
                accumulate_embeddings[key] = [value]
            else:
                accumulate_embeddings[key].append(value)
    # Aggregate keys with duplicate values
    agg_embeddings = {}
    for key, value in accumulate_embeddings.items():
        if len(value) > 1:
            agg_embeddings[key] = np.array(value).mean(axis=0)
        else:
            agg_embeddings[key] = value[0]
    return agg_embeddings


def mean_agg(embeddings: list[NamedEmbedding]) -> NamedEmbedding:
    """Mean aggregation for named embeddings.

    NamedEmbedding is a dict[str, embedding value], where embedding value may be
    either a dense or sparse embedding. This may also be wrapped in an EmbeddingItem.
    This method first flattens all embeddings into a list and then aggregates them.
    """
    if len(embeddings) == 0:
        return embeddings
    elif isinstance(embeddings[0], list):  # embeddings is a list of float
        return mean_agg_dense(embeddings)
    elif isinstance(embeddings[0], dict):  # embeddings are NamedEmbedding list of dict[str, Any]
        # Unpack all embeddings for each item into flat list of tuples with name, kind and value
        unpacked_embeddings = unpack_embeddings(embeddings)
        # Aggregate embeddings by name, select aggregation function by kind
        agg_embeddings = defaultdict(list)
        agg_method_map = {}
        for flat_embedding in unpacked_embeddings:
            # Store embeddings with same name in a dict of list
            key = flat_embedding.name
            agg_embeddings[key].append(flat_embedding)
            # Select aggregation method for each embedding name
            if key not in agg_method_map:
                agg_method_map[key] = (
                    mean_agg_dense if flat_embedding.kind == "dense" else mean_agg_sparse
                )
        # Actually aggregate each embedding
        for key, value in agg_embeddings.items():
            agg_embeddings[key] = agg_method_map[key]([e.value for e in value])
        return dict(agg_embeddings)
    else:
        raise ValueError("Cannot aggregate embeddings with unknown type.")


def dict_with_list_to_list_of_dict(
    embeddings: dict[str, Any], length: int | None = None
) -> list[dict[str, Any]]:
    """Named embeddings from BGE-M3 are created as a dictionary of
    multiple embeddings e.g. {"dense": [...], "sparse": [...], "colbert": [...]}.

    This method will convert the dict of lists into a list of dict where each
    dict corresponds to a single example that is embedded.

    This method assumes values of dictionary are all lists with same length.
    """
    if isinstance(embeddings, dict):  # Multivector Embeddings in Dict
        if length is None:
            length = len(list(embeddings.values())[0])
        # Convert Dict of lists to List of dicts
        list_of_dict_embeddings = []
        for idx in range(length):
            d = {
                key: value[idx] if value is not None else None for key, value in embeddings.items()
            }
            list_of_dict_embeddings.append(d)
        return list_of_dict_embeddings
    else:
        return embeddings


def sparse_embedding_dict_to_index_vector_lists(
    sparse_embedding: dict[str, Any],
) -> tuple[list[int], list[float]]:
    """Converts sparse embedding mapping in dict format into a list of sparse indices and
    a list of sparse vectors and returns the 2 lists as a tuple.
    """
    sparse_indices, sparse_vectors = [], []
    for key, value in sparse_embedding.items():
        sparse_indices.append(key)
        sparse_vectors.append(value)
    return sparse_indices, sparse_vectors
