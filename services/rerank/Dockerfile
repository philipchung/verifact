# Dockerfile that setups up a container that runs a modified Infinity Embeddings server.
# Infinity Embeddings: https://github.com/michaelfeil/infinity
#
# This Dockerfile is used to build the server targeted toward reranking using the torch engine.

FROM michaelf34/infinity:0.0.53

## Patch Infinity Embedding files in App running in container
# Patch torch crossencoder engine for reranking to free GPU VRAM as soon as tensors moved to CPU.
COPY ./infinity_patch/transformer/crossencoder/torch.py /app/infinity_emb/transformer/crossencoder/

# Reranker (Torch, GPU)
ENTRYPOINT ["/bin/sh", "-c", "\
    . /app/.venv/bin/activate && \
    infinity_emb v2 --model-id=BAAI/bge-reranker-v2-m3 --served-model-name=bge-reranker-v2-m3 \
    --port=7997 --url-prefix=/v1 --engine=torch --device=cuda"]