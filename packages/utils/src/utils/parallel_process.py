import multiprocessing
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from multiprocessing import cpu_count
from multiprocessing.context import BaseContext
from threading import Semaphore
from typing import Any

from tqdm.auto import tqdm


class BoundedProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self, *args, max_pending_items, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._semaphore = Semaphore(value=max_pending_items)

    def submit(self, fn, /, *args, **kwargs) -> Future:
        # Acquire semaphore before submitting job
        self._semaphore.acquire()
        f = super().submit(fn, *args, **kwargs)
        # When future finishes, release semaphore
        f.add_done_callback(lambda _: self._semaphore.release())
        return f


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, *args, max_pending_items, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._semaphore = Semaphore(value=max_pending_items)

    def submit(self, fn, /, *args, **kwargs) -> Future:
        # Acquire semaphore before submitting job
        self._semaphore.acquire()
        f = super().submit(fn, *args, **kwargs)
        # When future finishes, release semaphore
        f.add_done_callback(lambda _: self._semaphore.release())
        return f


def parallel_process(
    iterable: list | Iterable,
    function: Callable | None = None,
    n_jobs: int | None = None,
    use_args: bool = False,
    use_kwargs: bool = False,
    desc: str | None = None,
    mp_context: str | BaseContext | None = "fork",
    **kwargs,
) -> list:
    """
    A parallel version of the map function with a progress bar.
    Args:
        iterable (collection): An array-like or dict-like to iterate over
        function (function): A python function to apply to the elements of array.
            A special requirement for this function is that it needs to be at
            top-level scope so python can pickle the function. Also all variables
            within the function must be pickle-able.  Note that only this function has
            this requirement and the calling scope for `parallel_process` can be
            embedded within another function or class.
            In some cases, we want to run through the iterable items in parallel,
            but we don't need to call a specific function on the iterable.  In these
            situations, set function=None.
        n_jobs (int): The number of cores to use
        use_args (boolean, default=False): Whether to consider the elements of array
            as tuples of arguments to function. Tuple elements are passed to function
            arguments by position.  Set this to True if function has multiple arguments
            and your tuple provides the arguments in-order.
        use_kwargs (boolean, default=False): Whether to consider the elements of array
            as dictionaries of keyword arguments to pass into function.  Set this to
            True if function has multiple arguments and you want to pass arguments to
            function by keyword (does not need to be in-order).
        desc (string, default=None): Description on progress bar.
        mp_context (str | None): Either `spawn`, `fork`, `forkserver`.
            NOTE: `spawn` creates a new python interpreter for each process and has more
            overhead, but avoids deadlock scenario in `fork` when parent owns a resource
            and the parent is forked, now creating a child that also competes with the
            resource.  `fork` is more lightweight but unsafe.
    Returns:
        [function(iterable[0]), function(iterable[1]), ...]
    """
    if function is None:
        function = return_self
    n_jobs = cpu_count() if n_jobs is None else n_jobs
    mp_context = multiprocessing.get_context(mp_context)
    desc = "task" if desc is None else desc

    # Get Iterable Length for pbar
    if isinstance(iterable, Sequence) and "total" not in kwargs:
        iterable = list(iterable)
        kwargs["total"] = len(iterable)

    # If we set n_jobs to 1, just run a list comprehension.
    # This is useful for benchmarking and debugging.
    if n_jobs == 1:
        outputs = []
        for a in (pbar := tqdm(iterable, desc=desc, **kwargs)):
            if use_kwargs:
                output = function(**a)
            elif use_args:
                output = function(*a)
            else:
                output = function(a)
            outputs.append(output)
        return outputs
    else:
        # If n_jobs > 1, assemble a pool of worker processes and submit each item as a job
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_context) as pool:
            futures = []
            for a in iterable:
                if use_kwargs:
                    future = pool.submit(function, **a)
                elif use_args:
                    future = pool.submit(function, *a)
                else:
                    future = pool.submit(function, a)
                futures.append(future)

        # Monitor the completion of futures with a progress bar
        kwargs["total"] = len(futures)
        for _ in (pbar := tqdm(as_completed(futures), desc=desc, **kwargs)):
            # Note: as_completed() iterates in the order of futures completed, not the
            # original order of items in the futures list
            pass

        # Get the results from the futures (in order of original tasks)
        outputs = []
        for i, future in enumerate(futures):
            try:
                outputs += [future.result()]
            except Exception as e:
                outputs += [
                    (
                        e,
                        f"Input element at index: {i}.  Error: {e}",
                    )
                ]
    return outputs


def return_self(x: Any) -> Any:
    "A no-op function that returns self."
    return x
