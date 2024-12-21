import logging

import requests
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from rag.components import get_llm
from rq import Queue, Retry
from rq.job import Job
from rq_utils import (
    block_and_accumulate_results,
    enqueue_jobs,
    get_queue,
    get_redis,
    shutdown_all_workers,
    start_workers,
)
from utils import load_environment

load_environment()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def count_words_at_url(url, **kwargs) -> str | int:
    resp = requests.get(url)
    result = len(resp.text.split())
    if kwargs.get("description"):
        return f"Description: {kwargs.get('description')}.  Result: {result}"
    else:
        return result


def write_joke(topic: str = "apples", character: str = "pirate") -> str:
    # NOTE: need to load environment variables within function for rq to work
    from utils import load_environment

    load_environment()

    llm = get_llm(system_prompt=f"You are a comedic {character}.")
    response: ChatResponse = llm.chat(
        messages=[
            ChatMessage(role=MessageRole.USER, content=f"Tell me a joke about {topic}"),
        ]
    )
    return response.message.content


topics = [
    "The history of the Roman Empire",
    "The psychology of color",
    "The impact of social media on mental health",
    "Quantum computing",
    "Climate change and its effects on wildlife",
    "The art of negotiation",
    "The future of renewable energy",
    "The role of artificial intelligence in healthcare",
    "The history of jazz music",
    "Space exploration and colonization",
    "The significance of dreams in psychology",
    "The evolution of language",
    "The benefits of meditation",
    "The history of video games",
    "The impact of diet on brain function",
    "The ethics of genetic engineering",
    "The cultural significance of tattoos",
    "The history of the internet",
    "The science of happiness",
    "The influence of ancient Greek philosophy",
    "The future of work and automation",
    "The role of women in World War II",
    "The benefits of a plant-based diet",
    "The history of the Renaissance",
    "The impact of globalization on local cultures",
    "The psychology of addiction",
    "The development of virtual reality technology",
    "The history of the Olympic Games",
    "The science of climate change",
    "The role of art in society",
    "The benefits of outdoor activities on mental health",
    "The history of space exploration",
    "The impact of technology on education",
    "The significance of body language in communication",
    "The history of rock 'n' roll music",
    "The future of transportation",
    "The psychology of motivation",
    "The impact of tourism on the environment",
    "The history of the printing press",
    "The role of storytelling in human culture",
    "The benefits of learning a second language",
    "The history of fashion",
    "The impact of artificial intelligence on employment",
    "The science of nutrition",
    "The significance of rituals in different cultures",
    "The future of urban development",
    "The psychology of leadership",
    "The history of film and cinema",
    "The impact of pandemics on human history",
    "The benefits of physical exercise on mental health",
]


def create_joke_jobs(queue: str | Queue) -> list[Job]:
    q = get_queue(queue=queue)
    job_datas = [
        Queue.prepare_data(
            write_joke,
            kwargs={"topic": topics[i], "character": "pirate"},
            job_id=f"job-{i}",
            description=f"Joke {i}: {topics[i]}",
            retry=Retry(max=10, interval=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]),
        )
        for i in range(50)
    ]
    return enqueue_jobs(job_datas=job_datas, queue=q)


def main(queue: str = "default", num_workers: int = 10) -> list[Job]:
    # Redis Connection
    redis = get_redis()
    # Get Queue
    q = get_queue(connection=redis, queue=queue)
    # Put Jobs on Queue
    jobs = create_joke_jobs(queue=queue)
    # Create Workers to Process Jobs
    start_workers(queues=[q], num_workers=num_workers)

    # Poll Jobs & Block until Jobs are done.  Accumulate results as jobs finish.
    block_and_accumulate_results(jobs=jobs, polling_interval=3, show_progress=True)

    # Warm Shutdown Command to Workers (Shut down after they finish current Jobs)
    print("Queue is empty. Shutting down workers.")
    shutdown_all_workers(connection=redis, queue=q)

    return jobs


if __name__ == "__main__":
    # Run Jobs with Multiple Workers
    jobs = main(queue="default", num_workers=24)
    results = [job.result for job in jobs]
