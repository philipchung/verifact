# Run VeriFact

This directory supports two different evaluation paths. Choose the path that matches the
reproduction question.

1. `replay_physionet_v1_1_0.py` replays LLM-as-a-judge inference using the exact propositions,
   retrieved reference contexts, and model/configuration coverage published in PhysioNet v1.1.0.
   It does not require Qdrant, embedding, reranking, Redis, or rebuilding the EHR vector database.
2. `run_verifact.py` reruns the complete retrieval-and-judging pipeline. It requires the EHR vector
   database and all retrieval services. Retrieval results may differ from the released manuscript
   inputs if the database, model, or service versions differ.

## Setup

Install the Python environment, download
[VeriFact-BHC v1.1.0](https://physionet.org/content/mimic-iii-ext-verifact-bhc/1.1.0/),
and create a local environment file:

```sh
uv sync --all-packages --frozen
cp .env.example .env
```

Set `PROJECT_DIR`, `VERIFACTBHC_DATASET_DIR`, `HF_HOME`, GPU assignments, and `HF_TOKEN` in
`.env` for the local machine. The `.env` file is ignored by Git. Do not place credentials in a
model profile.

## Exact manuscript-input replay

The replay manifest starts from the rows in `verifact/verdicts.parquet`. It then validates every
row against `rater_configurations.csv`, `model_configuration_matrix.csv`, and
`propositions.csv.gz`. Consequently, it replays only observations that exist in the release; it
does not synthesize missing model/patient/configuration combinations.

First validate and summarize a selection without loading reference text or calling a model:

```sh
uv run --frozen python scripts/evaluate/replay_physionet_v1_1_0.py \
  --model-profile configs/inference/qwen-3-30b-a3b-thinking.env \
  --dry-run
```

Start Traefik and the selected main model. Reasoning models also require the auxiliary structured
output model:

```sh
# Non-reasoning example
docker compose --env-file .env \
  --env-file configs/inference/gemma-3-12b.env \
  up -d traefik llm-main

# Reasoning example
docker compose --env-file .env \
  --env-file configs/inference/qwen-3-30b-a3b-thinking.env \
  up -d traefik llm-main llm-aux
```

Run a small inference smoke test before a larger replay:

```sh
uv run --frozen python scripts/evaluate/replay_physionet_v1_1_0.py \
  --model-profile configs/inference/qwen-3-30b-a3b-thinking.env \
  --subject-id 1084 \
  --limit 10
```

Selections over 1,000 rows require `--allow-large-run`. Repeat options such as `--subject-id`,
`--rater-alias`, `--author-type`, `--proposition-type`, `--fact-type`, and `--top-n` to select
multiple values. Output is partitioned by rater and BHC, and `--resume` validates and skips complete
partitions.

Each output row includes the release key, expected released verdict, replayed verdict and reason,
and `verdict_matches_release`. This comparison is diagnostic: exact input replay does not guarantee
byte-identical generated text or labels because inference can be stochastic and hardware/runtime
details can affect generation.

## Published model profiles

The files in `configs/inference/` pin the release model and tokenizer identities and the runtime
mode needed by each manuscript judge:

| Release model | Profile | Reasoning |
| --- | --- | --- |
| Llama-8B | `llama-8b.env` | No |
| Llama-70B | `llama-70b.env` | No |
| R1-8B | `r1-8b.env` | Yes |
| R1-70B | `r1-70b.env` | Yes |
| Gemma-3-12B | `gemma-3-12b.env` | No |
| Gemma-3-27B | `gemma-3-27b.env` | No |
| Qwen-3-30B-A3B-Instruct | `qwen-3-30b-a3b-instruct.env` | No |
| Qwen-3-30B-A3B-Thinking | `qwen-3-30b-a3b-thinking.env` | Yes |
| Qwen-3-32B | `qwen-3-32b.env` | No; thinking is explicitly disabled |

The replay command fails before inference if the selected profile disagrees with
`model_metadata.csv`. See `configs/inference/README.md` for the profile contract.

## Full retrieval-and-judging pipeline

Use this path to test a changed retrieval system or to fact-check arbitrary text. It requires:

- Qdrant populated with sentence and atomic-claim EHR facts;
- embedding and reranking APIs;
- Redis and RQ workers;
- the selected main LLM and, for reasoning models, the auxiliary LLM.

After creating the vector database as described in `scripts/dataset/README.md`, run a bounded
configuration explicitly:

```sh
uv run --frozen python scripts/evaluate/run_verifact.py \
  --model-profile configs/inference/qwen-3-30b-a3b-thinking.env \
  --run-name LLMClaimClaim-NumFacts50 \
  --subject-id 1084 \
  --author-type llm \
  --proposition-type claim \
  --fact-type claim \
  --top-n 50 \
  --retrieval-method rerank \
  --reference-format absolute_time \
  --reference-only-admission
```

Options with list types may be repeated to run a matrix. Avoid invoking the broad defaults until
the bounded command and all services have been verified. `--model-profile` configures the host
Python client as well as documenting the matching Docker overlay; it also supplies the reasoning
mode and default output subdirectory.
