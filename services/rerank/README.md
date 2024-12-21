# Infinity serving BGE-Reranker-V2-M3

BGE-Reranker-V2-M3 reranking model is configured to run with torch backend on GPU.

## Build and Run Infinity Reranker Docker Container

The key modification made is the deletion of tensors on GPU immediately after they are copied to CPU and converted to numpy arrays. This prevents tensors from occupying VRAM, which can be quite significant for long sequences at 8k context window limit. This modification makes it more feasible to allow reranker model to share GPU with another model.

```sh
# Build container. Custom Dockerfile patches the torch & optimum embedding engine and FastAPI schemas/pydantic models for embedding models to support BGE-M3
# To switch between CPU/GPU backend for BGE-M3 embedding model, adjust entrypoint in Dockerfile.
docker build . -t services/infinity

# Run the container with 2 servers--an embedding & reranker model.
# NOTE: it can take 1-2 minutes for models to be downloaded, ONNX model to be optimized, and start servers
docker run -d --gpus all -p 8001:8001 -p 8002:8002 --name=infinity services/infinity
```
