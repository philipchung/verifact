# %% [markdown]
# Generate Embeddings by making a HTTP POST request to the embedding service
# %%
import httpx

RERANK_URL = "http://rerank.localhost/v1/rerank"

# Rerank (score) a batch of examples (each example is a pair of sentences)
query = "Invasive procedure"
documents = ["IV placement", "Endoscopy", "Appendectomy", "Heart Transplant"]
response = httpx.request(
    method="POST",
    url=RERANK_URL,
    json={
        "query": query,
        "documents": documents,
    },
)
response.json()
# %%
# Rerank (score) multiple samples in a loop
query = "Invasive procedure"
documents = ["IV placement", "Endoscopy", "Appendectomy", "Heart Transplant"]
response_jsons = []
for document in documents:
    response = httpx.request(
        method="POST",
        url=RERANK_URL,
        json={
            "query": query,
            "documents": [document],
        },
    )
    response_jsons.append(response.json())
print(response_jsons)
# Each response json is a dict with key "results" containing scores
# %%
# NOTE: OpenAI does not have a reranking service, thus there is no Python SDK for reranking.
