import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

from utils import create_uuid_from_string

logger = logging.getLogger(__name__)


class SentenceCombination(TypedDict):
    sentence: str
    index: int
    combined_sentence: str
    combined_sentence_embedding: list[float]


@dataclass
class SplitResult:
    """Result of a text split event.

    Each SplitResult object contains the following fields:
        id_ (str): The unique identifier for the split detail object, which is generated
            based on the input_text and text_chunks.
        input_text (str): The original text used to generate the chunk.
        text_chunks (Sequence[str]): The list of text chunks generated at the
            current recursion depth.
        distances (Sequence[float]): The list of cosine distances between adjacent text chunks.
        recursion_depth (int): The recursion depth at which the chunk was generated.
        chunk_idx (int): The index of the chunk within the parent input text after
            splitting at the current recursion depth.
        splitter (str): The method used to split the text. Either "semantic" or "sentence".
    """

    id_: str | None = None
    input_text: str | None = None
    text_chunks: Sequence[str] | None = None
    distances: Sequence[float] | None = None
    recursion_depth: int | None = None
    chunk_idx: int | None = None
    splitter: str | None = None

    def __post_init__(self) -> None:
        self.id_ = self._create_id()

    def _create_id(self) -> str:
        "Create a unique ID for the SplitResult object."
        numbered_text_chunks = [f"{i}_{chunk}" for i, chunk in enumerate(self.text_chunks)]
        str_to_hash = "-".join(numbered_text_chunks)
        return str(create_uuid_from_string(str_to_hash))

    def __repr__(self) -> str:
        "Print string representation of object."
        if self.input_text is not None:
            if len(self.input_text) <= 100:
                input_text_str = self.input_text[:100]
            else:
                input_text_str = (
                    f"{self.input_text[:100]}... and {len(self.input_text) - 100} more characters"
                )
        if self.text_chunks is not None:
            if len(self.text_chunks) <= 6:
                text_chunks_str = f'{[x[:30] + "..." for x in self.text_chunks]}'
            else:
                text_chunks_str = (
                    f'{[x[:30] + "..." for x in self.text_chunks][:6]} '
                    f'and {len(self.text_chunks) - 6} more items...'
                )
        else:
            text_chunks_str = "None"
        if self.distances is not None:
            if len(self.distances) <= 4:
                distances_str = self.distances[:4]
            else:
                distances_str = f"{self.distances[:4]} and {len(self.distances) - 4} more items..."
        else:
            distances_str = "None"
        return (
            "SplitResult("
            f"id_={self.id_},"
            f"input_text={input_text_str}, "
            f"text_chunks={text_chunks_str}, "
            f"distances={distances_str})"
            f"recursion_depth={self.recursion_depth}, "
            f"chunk_idx={self.chunk_idx}, "
            f"splitter={self.splitter}, "
            ")"
        )


def deduplicate_split_results(split_results: list[SplitResult]) -> list[SplitResult]:
    """Deduplicate a list of SplitResult objects based on their `id_`."""
    seen_ids = set()
    unique_items = []
    for item in split_results:
        if item.id_ not in seen_ids:
            unique_items.append(item)
            seen_ids.add(item.id_)
    return unique_items


@dataclass
class SplitBundle:
    """Container for nodes and detailed outputs of the recursive semantic splitting process.

    Each SplitBundle object contains the following fields:
        nodes (list[BaseNode]): list of nodes resulting from text splitting.
        text_chunks (list[str]): list of raw text used to construct the nodes.
        recursion_depths (list[int]): list of recursion depths for each chunk.
            For example `0` means the chunk was generated at the first pass
            with the semantic splitter. `1` means the initial chunk exceeded
            the maximum chunk size and was recursively split into smaller text_chunks
            using the semantic splitter.
        split_details (list[SplitResult]): list of SplitResult objects.
            A SplitResult object provides details of a text split event.
            In the recursive splitting process, it is generated each time a chunk
            exceeds maximum chunk size and is split either using semantic splitter
            or sentence splitter.

    NOTE: "nodes", "text_chunks", and "recursion_depths" are the same length and
        corresponding text are in the same order as the original text in the document.
    """

    text_chunks: list[str]
    recursion_depths: list[int]
    split_results: list[SplitResult]
