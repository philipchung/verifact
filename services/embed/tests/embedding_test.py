# %% [markdown]
# Generate Embeddings by making a HTTP POST request to the embedding service
# %%
import httpx

EMBED_URL = "http://embed.localhost/v1/embeddings"

# Embed a batch of examples
sentences = ["pediatric", "adult", "ETT", "surgery", "intubation"]

response = httpx.request(
    method="POST",
    url=EMBED_URL,
    json={
        "input": sentences,
    },
)
response.json()
# %%
dense0 = response.json()["data"][0]["embedding"]["dense"]
sparse0 = response.json()["data"][0]["embedding"]["sparse"]
colbert0 = response.json()["data"][0]["embedding"]["colbert"]

dense1 = response.json()["data"][1]["embedding"]["dense"]
sparse1 = response.json()["data"][1]["embedding"]["sparse"]
colbert1 = response.json()["data"][1]["embedding"]["colbert"]

dense2 = response.json()["data"][2]["embedding"]["dense"]
sparse2 = response.json()["data"][2]["embedding"]["sparse"]
colbert2 = response.json()["data"][2]["embedding"]["colbert"]

dense3 = response.json()["data"][3]["embedding"]["dense"]
sparse3 = response.json()["data"][3]["embedding"]["sparse"]
colbert3 = response.json()["data"][3]["embedding"]["colbert"]

dense4 = response.json()["data"][4]["embedding"]["dense"]
sparse4 = response.json()["data"][4]["embedding"]["sparse"]
colbert4 = response.json()["data"][4]["embedding"]["colbert"]

# %%
# Embed multiple samples in a loop
sentences = ["pediatric", "adult", "ETT", "surgery", "intubation"]
response_jsons = []
for sentence in sentences:
    response = httpx.request(
        method="POST",
        url=EMBED_URL,
        json={
            "input": [sentence],
        },
    )
    response_jsons.append(response.json())
print(response_jsons)

# %% [markdown]
# Use OpenAI Python SDK to interact with the embedding service
# %%
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://embed.localhost/v1/")
client.models.list()

# %%
sentences = ["pediatric", "adult", "ETT", "surgery", "intubation"]
response = client.embeddings.create(model="BAAI/bge-m3", input=sentences)

# %%
dense0 = response.data[0].embedding["dense"]
sparse0 = response.data[0].embedding["sparse"]
colbert0 = response.data[0].embedding["colbert"]

dense1 = response.data[1].embedding["dense"]
sparse1 = response.data[1].embedding["sparse"]
colbert1 = response.data[1].embedding["colbert"]

dense2 = response.data[2].embedding["dense"]
sparse2 = response.data[2].embedding["sparse"]
colbert2 = response.data[2].embedding["colbert"]

dense3 = response.data[3].embedding["dense"]
sparse3 = response.data[3].embedding["sparse"]
colbert3 = response.data[3].embedding["colbert"]

dense4 = response.data[4].embedding["dense"]
sparse4 = response.data[4].embedding["sparse"]
colbert4 = response.data[4].embedding["colbert"]

# %%
