"""Async utils.

Original Source: llamaindex.core.async_utils
"""

import asyncio
from collections.abc import Coroutine, Iterable
from itertools import zip_longest
from typing import Any, TypeVar


def get_event_loop() -> asyncio.AbstractEventLoop:
    """Helper function to get existing event loop or create new one."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def asyncio_module(show_progress: bool = False) -> Any:
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        module = tqdm_asyncio
    else:
        module = asyncio

    return module


def asyncio_run(coro: Coroutine) -> Any:
    """Gets an existing event loop to run the coroutine.

    If there is no existing event loop, creates a new one.
    """
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


def run_async_tasks(
    tasks: list[Coroutine],
    show_progress: bool = False,
    progress_bar_desc: str = "Running async tasks",
) -> list[Any]:
    """Run a list of async tasks."""
    tasks_to_execute: list[Any] = tasks
    if show_progress:
        try:
            import nest_asyncio
            from tqdm.asyncio import tqdm

            # jupyter notebooks already have an event loop running
            # we need to reuse it instead of creating a new one
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()

            async def _tqdm_gather() -> list[Any]:
                return await tqdm.gather(*tasks_to_execute, desc=progress_bar_desc)

            tqdm_outputs: list[Any] = loop.run_until_complete(_tqdm_gather())
            return tqdm_outputs
        # run the operation w/o tqdm on hitting a fatal
        # may occur in some environments where tqdm.asyncio
        # is not supported
        except Exception:
            pass

    async def _gather() -> list[Any]:
        return await asyncio.gather(*tasks_to_execute)

    outputs: list[Any] = asyncio_run(_gather())
    return outputs


def chunks(iterable: Iterable, size: int) -> Iterable:
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=None)


async def batch_gather(
    tasks: list[Coroutine], batch_size: int = 10, verbose: bool = False
) -> list[Any]:
    output: list[Any] = []
    for task_chunk in chunks(tasks, batch_size):
        output_chunk = await asyncio.gather(*task_chunk)
        output.extend(output_chunk)
        if verbose:
            print(f"Completed {len(output)} out of {len(tasks)} tasks")
    return output


def get_asyncio_module(show_progress: bool = False) -> Any:
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        module = tqdm_asyncio
    else:
        module = asyncio

    return module


DEFAULT_NUM_WORKERS = 4

T = TypeVar("T")


async def run_jobs(
    jobs: list[Coroutine[Any, Any, T]],
    show_progress: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
    desc: str | None = None,
) -> list[T]:
    """Run jobs.

    Args:
        jobs (List[Coroutine]):
            List of jobs to run.
        show_progress (bool):
            Whether to show progress bar.

    Returns:
        List[Any]:
            List of results.
    """
    semaphore = asyncio.Semaphore(workers)

    async def worker(job: Coroutine) -> Any:
        async with semaphore:
            return await job

    pool_jobs = [worker(job) for job in jobs]

    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        results = await tqdm_asyncio.gather(*pool_jobs, desc=desc)
    else:
        results = await asyncio.gather(*pool_jobs)

    return results
