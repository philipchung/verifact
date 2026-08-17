# Dataset and EHR vector-database creation

This directory contains the historical construction pipeline for the unannotated VeriFact-BHC
dataset and the full-retrieval inference path. It is not required for saved-verdict analysis or
for exact manuscript-input replay from PhysioNet v1.1.0.

There are three reproducibility levels:

1. **Saved-verdict analysis:** use `scripts/analysis_verifact/` with PhysioNet v1.1.0.
2. **Exact judge-input replay:** use `scripts/evaluate/replay_physionet_v1_1_0.py`; released
   propositions and retrieved contexts are reused, so no vector database is required.
3. **Full pipeline:** use the steps below to regenerate propositions or retrieval. LLM sampling,
   model/runtime changes, and vector-index construction mean outputs are not expected to be
   byte-identical to the frozen release.

## Prerequisites

Create `.env` from `.env.example`, configure local paths and services, and obtain authorized access
to [MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/).

```sh
cp .env.example .env
uv sync --all-packages --frozen

wget -r -N -c -np --directory-prefix=data \
  --user "${PHYSIONET_USERNAME}" --ask-password \
  https://physionet.org/files/mimiciii/1.4/
```

The default `.env.example` expects MIMIC-III at
`data/physionet.org/files/mimiciii/1.4/`.

## Recreate the unannotated study dataset

Sample the patient cohort and prepare the source tables:

```sh
uv run --frozen python scripts/dataset/sample_mimic_dataset.py
```

Generate LLM-written Brief Hospital Courses (BHCs):

```sh
uv run --frozen python scripts/dataset/write_llm_hospital_course.py
```

These scripts use local model services and Redis Queue. The generated BHCs and subsequent text
decomposition can differ from the released texts because generation is stochastic.

Decompose human- and LLM-written BHCs into sentence and atomic-claim propositions:

```sh
uv run --frozen python scripts/dataset/decompose_text.py \
  --input-file bhc_noteevents.feather \
  --dataset-dir data/dataset \
  --upsert-db --collection-name bhc_noteevents \
  --save-pickle --output-dir-name bhc_nodes \
  --queue-name human_bhc_decompose \
  --num-parallel-pipelines 50 --llm-n-jobs 32

uv run --frozen python scripts/dataset/decompose_text.py \
  --input-file llm_bhc_noteevents.feather \
  --dataset-dir data/dataset \
  --upsert-db --collection-name llm_bhc_noteevents \
  --save-pickle --output-dir-name llm_bhc_nodes \
  --queue-name llm_bhc_decompose \
  --num-parallel-pipelines 50 --llm-n-jobs 32
```

## Build an EHR fact vector database

For a newly constructed local dataset, decompose all reference EHR notes:

```sh
uv run --frozen python scripts/dataset/decompose_text.py \
  --input-file ehr_noteevents.feather \
  --dataset-dir data/dataset \
  --upsert-db --collection-name ehr_noteevents \
  --no-save-pickle \
  --no-load-nodes-from-vectorstore-if-exists \
  --queue-name ehr_note_decompose \
  --num-parallel-pipelines 32 --llm-n-jobs 16
```

To build the database from the released patient EHRs, download
[VeriFact-BHC v1.1.0](https://physionet.org/content/mimic-iii-ext-verifact-bhc/1.1.0/) and run:

```sh
wget -r -N -c -np --directory-prefix=data \
  --user "${PHYSIONET_USERNAME}" --ask-password \
  https://physionet.org/files/mimic-iii-ext-verifact-bhc/1.1.0/

uv run --frozen python scripts/dataset/decompose_text.py \
  --input-file ehr_noteevents.csv.gz \
  --dataset-dir data/physionet.org/files/mimic-iii-ext-verifact-bhc/1.1.0/reference_ehr \
  --upsert-db --collection-name ehr_noteevents \
  --no-save-pickle \
  --no-load-nodes-from-vectorstore-if-exists \
  --queue-name ehr_note_decompose \
  --num-parallel-pipelines 32 --llm-n-jobs 16
```

This operation requires Qdrant, embedding, atomic-claim generation, Redis/RQ, and their model
services. Once the vector database is populated, follow `scripts/evaluate/README.md` for the full
retrieval-and-judging pipeline.
