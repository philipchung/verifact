# Infinity serving BGE-M3 embedding model

## Build and Run Infinity Embeddings Docker Container

To run BGE-M3 and generate sparse, dense, and colbert vectors as outputs, we customize the Infinity Embeddings server to enable generation of all 3 types of embeddings. This requires overriding several files in the docker container, so we define our own Dockerfile to do this.

Another optimization made is the deletion of tensors on GPU immediately after they are copied to CPU and converted to numpy arrays. This prevents tensors from occupying VRAM, which can be quite significant for long sequences at 8k context window limit. This modification makes it more feasible to allow embedding model to share GPU with another model.

```sh
# Build container. Custom Dockerfile patches the torch & optimum embedding engine and FastAPI schemas/pydantic models for embedding models to support BGE-M3
# To switch between CPU/GPU backend for BGE-M3 embedding model, adjust entrypoint in Dockerfile.
docker build . -t services/infinity

# Run the container with 2 servers--an embedding & reranker model.
# NOTE: it can take 1-2 minutes for models to be downloaded, ONNX model to be optimized, and start servers
docker run -d --gpus all -p 8001:8001 -p 8002:8002 --name=infinity services/infinity
```

## Disabled ColBERT Embeddings on Customized Torch GPU Backend

Computing ColBERT embeddings is very compute intensive for long sequences and results in roughly a 20x slow down for the GPU torch-accelerated backend. Because we do not use ColBERT embeddings in our downstream applications, we disable it.

To enable it in the torch backend, modify the code in `infinity_patch/transformer/embedder/sentence_transformer.py`.
