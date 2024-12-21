from collections.abc import Generator, Iterable
from itertools import islice
from typing import Any


def flatten_list_of_list(list_of_list: list[list[Any]]) -> list[Any]:
    return [x for lst in list_of_list for x in lst]


def flatten(xs) -> Generator[Any | str | bytes, Any, None]:
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, str | bytes):
            yield from flatten(x)
        else:
            yield x


def iter_batch(iterable: Iterable | Generator, batch_size: int) -> Iterable:
    """Iterate over an iterable in batches.

    >>> list(iter_batch([1, 2, 3, 4, 5], 3))
    [[1, 2, 3], [4, 5]]
    """
    source_iter = iter(iterable)
    while source_iter:
        b = list(islice(source_iter, batch_size))
        if len(b) == 0:
            break
        yield b
