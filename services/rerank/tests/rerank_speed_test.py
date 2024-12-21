# %% [markdown]
# Test Context Window, Memory Usage, Embedding Speed on Infinity using
# BGE-reranker-v2-m3 Reranking Model
# %%
# Imports & Prepare Dummy Text
import os
import time

import nest_asyncio
from httpx import AsyncClient, Client, Response
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer
from utils import load_environment

load_environment()
nest_asyncio.apply()

# Read 100 paragraphs of dummy text
with open("lorem_ipsum.txt") as f:
    text = f.read()

model_name = os.environ["RERANK_MODEL_NAME"]
base_url = os.environ["RERANK_URL_BASE"]
tokenizer = AutoTokenizer.from_pretrained(model_name)

token_ids = tokenizer.encode(text)
print("Full Text. Num Tokens: ", len(token_ids))

# NOTE: actual context lengths are usually 8192, 4096, 2048, 1024, but
# this includes both input and output. We leave some tokens for the LLM's output
# generation, otherwise it will fail.
token_ids_8k = token_ids[:7950]
token_ids_4k = token_ids[:3950]
token_ids_2k = token_ids[:1950]
token_ids_1k = token_ids[:950]
token_ids_500 = token_ids[:450]

text_8k = tokenizer.decode(token_ids_8k)
text_4k = tokenizer.decode(token_ids_4k)
text_2k = tokenizer.decode(token_ids_2k)
text_1k = tokenizer.decode(token_ids_1k)
text_500 = tokenizer.decode(token_ids_500)
print(
    "Created Text with Num Tokens: ",
    [
        len(token_ids_8k),
        len(token_ids_4k),
        len(token_ids_2k),
        len(token_ids_1k),
        len(token_ids_500),
    ],
)
# %%
client = Client()
RERANK_URL = base_url.lstrip("/") + "/rerank"

# Rerank (score) a batch of examples (each example is a pair of sentences)
query = "Tell me a story in latin."
documents = [text_8k] * 5
response = client.request(
    method="POST",
    url=RERANK_URL,
    json={
        "query": query,
        "documents": documents,
    },
)
response.json()

# %%
# Test Multiple Rerank requests (asynchronous requests)
aclient = AsyncClient(timeout=300)


async def score(query: str, documents: list[str]) -> Response:
    response = await aclient.request(
        method="POST",
        url=RERANK_URL,
        json={
            "query": query,
            "documents": documents,
        },
    )
    return response


texts = {
    "500 tokens": text_500,
    "1k tokens": text_1k,
    "2k tokens": text_2k,
    "4k tokens": text_4k,
    "8k tokens": text_8k,
}
num_items = 100

durations = {}
for key, sentence in texts.items():
    start = time.perf_counter()
    responses = await tqdm.gather(  # noqa: F704
        *[score(query=query, documents=[sentence]) for _ in range(num_items)]
    )
    end = time.perf_counter()
    durations[key] = end - start


print(f"Total Durations for reranking {num_items} times:")
for key, duration in durations.items():
    print(f"{key}: {duration:.2f} seconds")
# %%
# NOTE: bge-reranker-v2-m3 will occupy upward of ~5.6GB of VRAM for heavy reranking jobs with 8k tokens
