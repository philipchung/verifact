# Verifying Facts in LLM-Generated Clinical Text with Electronic Health Records

**Preprint Manuscript:** [VeriFact: Verifying Facts in LLM-Generated Clinical Text with Electronic Health Records](https://arxiv.org/abs/2501.16672)

**Dataset:** [MIMIC-III-Ext-VeriFact-BHC: Labeled Propositions From Brief Hospital Course Summaries for Long-form Clinical Text Evaluation](https://physionet.org/content/mimic-iii-ext-verifact-bhc/1.0.0/)

`VeriFact`is a long-form text fact-checker that verifies any text written about a patient against their own electronic health record (EHR). VeriFact decomposes the text into a set of propositions which are individually verified against the patient's EHR. VeriFact combines RAG with LLM-as-a-Judge to perform fact verification.

`VeriFact-BHC` is a dataset to benchmark `VeriFact` performance against human clinicians. This dataset is derived from MIMIC-III Clinical Database v1.4. It contains human-written Brief Hospital Course (BHC) narratives typically found in discharge summaries and also a LLM-written BHC for 100 patients. It also contains the reference EHR for each patient. All BHC narratives are decomposed into propositions which are annotated by clinicians to develop a human clinician ground truth.

## Scripts

Scripts to generate the unannotated `VeriFact-BHC` dataset, run the `VeriFact` system to generate AI rater labels, and compute interrater agreement and classificaiton metrics are contained in `scripts`. These scripts rely on the locally-deployed services which are described below.

## Environment Variables

Add your environment variables in `.env`.

```sh
# Hugging Face Token: https://huggingface.co/docs/hub/en/security-tokens
HF_TOKEN=${HUGGINGFACE_READ_TOKEN}
# Hugging Face Cache
HF_HOME=${HOME}/.cache/huggingface

# Local Machine URL
SERVER_BASE_URL=localhost

# Traefik Configuration
ADMIN_EMAIL=email@domain.edu
```

If you plan to commit this code to a public repo, git ignore the `.env` file so you do not commit your secrets. The `.env` is made available in this repo for visibility to default environment variables which are used by docker containers and scripts.

## Python Environment

```sh
# Create python virtual environment
uv venv
# Create/update lock file (only if needed, otherwise skip this step)
uv lock
# Sync virtual environment with lockfile specification
uv sync --all-packages
```

## Services

All models used in `VeriFact` are local open-source models which can be launched using the provided `docker-compose.yml` configuration.

Local services include:

1. Local Embedding Model (requires GPU): customized `infinity` inference engine to serve the [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) model with both dense and sparse embedding generation.
2. Local Rerank Model (requires GPU): customized `infinity` inference engine to serve the [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) reranking model.
3. Local LLM Inference Service (requires GPU): `vLLM` serving for [hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4).
4. Vector Database: locally hosted `qdrant` vector database
5. Traefik: router, reverse proxy, load balancer
6. Redis: key-value store for redis-queue
7. Redis-Queue (RQ) Dashboard: monitoring `rq` jobs
8. Prometheus + Grafana: monitoring dashboard for `vLLM`.

These services are all containerized using docker. Docker Compose is used to coordinate launching and stopping these microservices.

```sh
# Start All Services (in detached mode)
docker compose up -d
# Check All Services Running
docker ps
# Check Logs
docker logs <container name>
# Inspect Each Container
docker exec -it <container name> /bin/sh
# Stop All Services
docker compose down
```

### Example Service Deployment

LLM Inference is significantly more compute intensive than Embedding or Reranking.  Thus it is recommended to setup LLMs in data parallel configuration. Embedding and Reranking models can share a GPU.

On a server with 4-GPUs (using the `docker-compose.yml` in this project):

```sh
# Launch Traefik for reverse proxy & load balancing
# Traefik Dashboard: ${SERVER_URL}:8090/dashboard
docker compose up traefik -d

# Launch Qdrant for vector database, Redis & RQ-Dashboard for tracking tasks in queue
# Qdrant Dashboard: ${SERVER_URL}:6333/dashboard
# Redis Stack Dashboard: ${SERVER_URL}:6380/redis-stack
# RQ-Dashboard: ${SERVER_URL}:9181
docker compose up qdrant redis rq-dashboard -d

# Launch Local LLM Inference API in Tensor Parallel Configuration (uses vLLM)
# The default LLM is a quantized Llama 3.1 70B model, which requires 37GB VRAM for the model itself. This container configures the LLM inference service in tensor parallelism which splits model weights across 2 GPUs.
docker compose up llm-tp2 -d

# Alternatively, launch local LLM Inference on a single GPU. Multiple docker containers can be launched and traefik will distribute API requests across the LLM containers in round-robin fashion
# docker compose up llm0 llm1 llm2 -d

# Launch Prometheus, Grafana dashboards for monitoring vLLM inference throughput
# Prometheus Dashboard: ${SERVER_URL}:9090
# Grafana Dashboard: ${SERVER_URL}:3000
docker compose up prometheus grafana -d

# Launch Embedding & Rerank Inference API on GPU3 (uses Infinity Embeddings)
# These containers are customized for compatibility with BGE-M3 model and to reduce VRAM use
docker compose up embed1 rerank1 -d
```

Specific configurations for ports and URLs are found in the `.env` file that `docker-compose.yml` references.

Docker services are reached via Traefik reverse proxy and load balancer. Using Traefik, multiple docker containers providing LLM inference can service the same API endpoint. Same is true for embedding and rerank inference services. Traefik will load balance the API requests equally across docker containers hosting the same service.

Parallel tasks are  managed using `rq` which is a queue backed by `redis`.

The vLLM inference service metrics are monitored via Prometheus and a Grafana dashboard. Prometheus and Grafana setup is described in `verifact/services/vllm/monitoring/README.md`.

## Performance

Performance of locally-hosted models is dependent on your GPU accelerator and local hardware. Lower latency and higher throughput may be achieved by replacing locally-hosted models with dedicated API inference services.

## Citation

```
@article{Chung2025,
      title={VeriFact: Verifying Facts in LLM-Generated Clinical Text with Electronic Health Records}, 
      author={Philip Chung and Akshay Swaminathan and Alex J. Goodell and Yeasul Kim and S. Momsen Reincke and Lichy Han and Ben Deverett and Mohammad Amin Sadeghi and Abdel-Badih Ariss and Marc Ghanem and David Seong and Andrew A. Lee and Caitlin E. Coombes and Brad Bradshaw and Mahir A. Sufian and Hyo Jung Hong and Teresa P. Nguyen and Mohammad R. Rasouli and Komal Kamra and Mark A. Burbridge and James C. McAvoy and Roya Saffary and Stephen P. Ma and Dev Dash and James Xie and Ellen Y. Wang and Clifford A. Schmiesing and Nigam Shah and Nima Aghaeepour},
      year={2025},
      eprint={2501.16672},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2501.16672}, 
}
```
