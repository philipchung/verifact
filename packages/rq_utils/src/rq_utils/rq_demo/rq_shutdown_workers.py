# %%
from rq import get_queue, get_redis, shutdown_all_workers
from utils import load_environment

load_environment()

redis = get_redis()
queue = get_queue(queue="te", connection=redis)
# Kill job that workers are working on
shutdown_all_workers(connection=redis, queue=queue, shutdown_type="kill")
# %%
