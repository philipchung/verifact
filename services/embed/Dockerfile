# Dockerfile that setups up a container that runs a modified Infinity Embeddings server.
# Infinity Embeddings: https://github.com/michaelfeil/infinity
#
# This server:
# - customized the sentence_transformer GPU backend for BGE-M3 to output 
#   dense, sparse, and colbert vectors (though we disable colbert for 20x speedup)
# - Provides Embedding API endpoints with batching and acceleration

FROM michaelf34/infinity:0.0.53

## Patch Infinity Embedding files in App running in container for BGE-M3 Multivector Compatibility
# Patch Sentence Transformer Embedder (Main Embedding Logic)
COPY ./infinity_patch/transformer/embedder/sentence_transformer.py /app/infinity_emb/transformer/embedder/
# Patch FastAPI Pydantic Schemas
COPY ./infinity_patch/fastapi_schemas/pymodels.py /app/infinity_emb/fastapi_schemas/

# Serve Embedding Model using Torch Backend on GPU
ENTRYPOINT ["/bin/sh", "-c", "\
    . /app/.venv/bin/activate && \
    infinity_emb v2 --model-id=BAAI/bge-m3 --served-model-name=bge-m3 \
    --port=7997 --url-prefix=/v1 --engine=torch --device=cuda"]