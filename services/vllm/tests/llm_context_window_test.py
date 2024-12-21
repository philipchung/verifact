# %% [markdown]
# Test Context Window & Memory Usage on vLLM
# %%
# Imports & Prepare Dummy Text
import os
import time

import nest_asyncio
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer
from utils import load_environment

load_environment()
nest_asyncio.apply()

# Read 100 paragraphs of dummy text
with open("lorem_ipsum.txt") as f:
    text = f.read()

model_name = os.environ["LLM_MODEL_NAME"]
base_url = os.environ["LLM_URL_BASE"]
tokenizer_model_name = os.environ["TOKENIZER_MODEL_NAME"]
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
# %%
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
# List Models
client = OpenAI(api_key="EMPTY", base_url=base_url)
client.models.list()
# %%
prompt = f"""\
Instruction: Extract the three most common words from text. Do not elaborate.
Text: {text_8k}
"""

prompt2 = (
    "You are a multilingual asssitant. "
    "Say 'hello world' in 20 different languages. "
    "Give me the response as a key-value dictionary."
)
# %%
# Test Single ChatCompletion (synchronous request)
messages = [
    {
        "role": "user",
        "content": prompt,
    },
]
response = client.chat.completions.create(
    model=os.environ["LLM_MODEL_NAME"],
    messages=messages,
    extra_body={
        "stop_token_ids": [128009]
    },  # Prompt fix for Llama3 vllm (https://github.com/vllm-project/vllm/issues/4180)
)
print("Used tokens: \n", {k: v for k, v in response.usage.dict().items()})
print("Response message: \n", response.choices[0].message.content)
# %%
# Test Multiple ChatCompletions (asynchronous requests)
aclient = AsyncOpenAI(api_key="EMPTY", base_url=base_url)

num_items = 100

texts = {
    "500 tokens": text_500,
    "1k tokens": text_1k,
    "2k tokens": text_2k,
    "4k tokens": text_4k,
    "8k tokens": text_8k,
}

durations = {}
for key, sentence in texts.items():
    prompt = f"""\
Instruction: Extract the three most common words from text. Do not elaborate.
Text: {sentence}
"""
    messages = [
        {
            "role": "user",
            "content": prompt,
        },
    ]
    start = time.perf_counter()
    responses = await tqdm.gather(  # noqa: F704
        *[
            aclient.chat.completions.create(
                model=os.environ["LLM_MODEL_NAME"],
                messages=messages,
                extra_body={"stop_token_ids": [128009]},
                # Prompt fix for Llama3 vllm (https://github.com/vllm-project/vllm/issues/4180))
            )
            for _ in range(num_items)
        ]
    )
    end = time.perf_counter()
    durations[key] = end - start

print(
    f"Total Durations for LLM generations for {num_items} times with prompts of different lengths:"
)
for key, duration in durations.items():
    print(f"{key}: {duration:.2f} seconds")

# %%
