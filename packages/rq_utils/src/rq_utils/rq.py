"""Helper functions for rq (Redis Queue)."""

import logging
import multiprocessing
import os
import time
from typing import Any

from redis import ConnectionPool, Redis
from rq import Queue, Worker
from rq.command import (
    send_kill_horse_command,
    send_shutdown_command,
    send_stop_job_command,
)
from rq.job import Job
from rq.queue import EnqueueData
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_redis_connection_pool(
    host: str | None = None,
    port: int | None = None,
    db: int = 0,
    max_connections: int = 24,
) -> ConnectionPool:
    """Get a Redis connection pool."""
    from utils import load_environment

    load_environment()
    host = host or os.environ["REDIS_HOST"]
    port = port or os.environ["REDIS_HOST_PORT"]
    return ConnectionPool(host=host, port=port, db=db, max_connections=max_connections)


def get_redis(
    host: str | None = None,
    port: int | None = None,
    db: int = 0,
    connection_pool: ConnectionPool = None,
) -> Redis:
    """Get a Redis connection."""
    from utils import load_environment

    load_environment()
    if connection_pool is None:
        connection_pool = get_redis_connection_pool(host=host, port=port, db=db)
    return Redis(connection_pool=connection_pool)


def get_queue(
    queue: str | Queue = "default",
    connection: Redis | None = None,
) -> Queue:
    """Get a Queue object."""
    if isinstance(queue, Queue):
        return queue
    else:
        connection = connection or get_redis()
        return Queue(name=queue, connection=connection)


def delete_queue(queue: str, connection: Redis | None = None) -> None:
    """Delete a Queue object in RQ. Make sure queue is empty first."""
    connection = connection or get_redis()
    connection.srem("rq:queues", f"rq:queue:{queue}")


def enqueue_jobs(
    job_datas: list[EnqueueData] | None = None,
    connection: Redis | None = None,
    queue: str | Queue = "default",
) -> list[Job]:
    """Create and enqueue jobs using list of EnqueueData created by Queue.prepare_data()."""
    if job_datas is None:
        raise ValueError("Must provide at least one EnqueueData for job.")
    connection = connection or get_redis()
    q = get_queue(connection=connection, queue=queue)
    job_list = q.enqueue_many(job_datas=job_datas)
    return job_list


def get_jobs(
    jobs: list[str | Job] | str | Job,
    connection: Redis | None = None,
) -> list[Job]:
    """Get jobs for given Job or job id.
    If list of jobs, return list of Job objects.
    If individual job or job id, return a single Job object."""
    connection = connection or get_redis()
    if isinstance(jobs, str):
        return Job.fetch(id=jobs, connection=connection)
    elif isinstance(jobs, Job):
        return Job.fetch(id=jobs.id, connection=connection)
    elif isinstance(jobs, list):
        job_ids = []
        for job in jobs:
            if isinstance(job, str):
                job_ids.append(job)
            elif isinstance(job, Job):
                job_ids.append(job.id)
            else:
                raise ValueError(
                    "Argument `job` must be a list of Job objects or job ids "
                    "or an individual Job object or job id."
                )
        return Job.fetch_many(job_ids=job_ids, connection=connection)
    else:
        raise ValueError(
            "Argument `job` must be a list of Job objects or job ids "
            "or an individual Job object or job id."
        )


def has_job_finished(job: Job) -> bool:
    match job.get_status(refresh=True):
        case "finished" | "stopped" | "canceled" | "failed":
            return True
        case "queued" | "started" | "deferred" | "scheduled":
            return False
        case _:
            return False


def block_and_accumulate_results(
    jobs: list[Job], polling_interval: int = 2, show_progress: bool = True
) -> list[Any]:
    """Block thread until all jobs are done.  Accumulate results as jobs finish."""
    logger = logging.getLogger(__name__)

    pbar = tqdm(total=len(jobs), disable=not show_progress)
    all_jobs_done = False
    num_finished_jobs = 0
    job_finished: dict[str, Any] = {job.id: False for job in jobs}  # Track which jobs are finished
    job_results: dict[str, Any] = {}  # Store results of finished jobs
    while not all_jobs_done:
        # Frequency of polling for job updates
        time.sleep(polling_interval)
        # Fetch Jobs
        jobs = get_jobs(jobs=jobs)

        # job_status = {job.id: job.get_status(refresh=True) for job in jobs}
        # logger.debug(f"Job status: {job_status}")

        # Determine if each job is finished or not
        job_finished = {job.id: has_job_finished(job) for job in jobs}

        # If new jobs completed, add to results list
        for job in jobs:
            if job_finished[job.id] and job.id not in job_results:
                job_results[job.id] = job.result
                logger.debug(f"Got result: {str(job.result)}")
                num_finished_jobs = sum(job_finished.values())
                pbar.update(1)
                pbar.refresh()

        logger.debug(f"Jobs finished: {num_finished_jobs}/{len(jobs)}")

        # Check if all jobs are done
        if all(job_finished.values()):
            print("All jobs are done.")
            all_jobs_done = True

    # Reorder results to match original job order
    results = [job_results[job.id] for job in jobs]
    return results


def start_worker(
    queues: list[str],
    name: str | None,
    burst: bool = True,
    redis_host: str | None = None,
    redis_port: str | None = None,
    redis_db: int = 0,
    **kwargs,
) -> None:
    """Start a single worker with the given queues and name.
    This function is intended to be run in a separate process using multiprocessing.Process

    """
    from rq import Connection, Worker

    # Sort kwargs to those for Worker.__init__() vs. Worker.work()
    worker_init_kwarg_list = [x for x in Worker.__init__.__code__.co_varnames if x != "self"]
    worker_work_kwarg_list = [x for x in Worker.work.__code__.co_varnames if x != "self"]
    worker_init_kwargs = {k: v for k, v in kwargs.items() if k in worker_init_kwarg_list}
    worker_work_kwargs = {k: v for k, v in kwargs.items() if k in worker_work_kwarg_list}
    # Start worker with connection
    redis = get_redis(host=redis_host, port=redis_port, db=redis_db)
    with Connection(connection=redis):
        worker = Worker(queues=queues, name=name, **worker_init_kwargs)
        worker.work(burst=burst, **worker_work_kwargs)


def start_workers(
    queues: list[str],
    num_workers: int = 8,
    name: str = "worker",
    burst: bool = False,
    **kwargs,
) -> None:
    """Start multiple workers. Each worker started in separate process."""
    for i in range(num_workers):
        multiprocessing.Process(
            target=start_worker,
            kwargs={
                "queues": queues,
                "name": f"{name}-{i}",
                "burst": burst,
            },
        ).start()


def get_all_workers(
    queue: str | Queue = "default",
    connection: Redis | None = None,
) -> list[Worker]:
    """Get all workers for a given queue."""
    connection = connection or get_redis()
    q = get_queue(connection=connection, queue=queue)
    return Worker.all(connection=connection, queue=q)


def shutdown_all_workers(
    queue: str | Queue = "default",
    connection: Redis | None = None,
    shutdown_type: str = "warm",
) -> None:
    """Shutdown all workers for a given queue"""
    connection = connection or get_redis()
    q = get_queue(connection=connection, queue=queue)
    workers = get_all_workers(connection=connection, queue=q)
    for worker in workers:
        if shutdown_type == "warm":
            send_shutdown_command(connection=connection, worker_name=worker.name)
        elif shutdown_type == "kill":
            send_kill_horse_command(connection=connection, worker_name=worker.name)
        elif shutdown_type == "stop-job":
            job_id = worker.get_current_job_id()
            try:
                send_stop_job_command(connection=connection, job_id=job_id)
            except Exception as ex:
                logger.info(f"Job ID: {job_id} is not currently executing.  Exception: {ex}")
