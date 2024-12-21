# Unannotated Dataset Construction

The following steps reproduce the creation of the unannotated dataset for the `VeriFact` study and `VeriFact-BHC`, meaning it does not contain any human or AI annotations.

The following steps also run the VeriFact AI system to generate AI ratings for each proposition extracted from Brief Hospital Course (BHC) text narratives in the dataset. However, the text decomposition may be slightly different than in `VeriFact-BHC` and our reported experiments due to the non-deterministic nature of LLMs, so refer to `analysis` scripts on downloading the exact text decomposition and annotations used in `VeriFact-BHC`.

## Create Dataset, Decomposed Text, and Vector Database of Facts

### Data Download

Follow instructions to access [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) on the Physionet platform.

```sh
# Download Dataset. 
# Replace verifact with this repository's root path
# Replace ${PHYSIONET_USERNAME} with your physionet username
wget -r -N -c -np --directory-prefix=~/verifact/data --user ${PHYSIONET_USERNAME} --ask-password https://physionet.org/files/mimiciii/1.4/
# Downloaded dataset is at verifact/data/physionet.org/files/mimiciii/1.4/
```

### Sample Patient Cohort & Dataset

`scripts/dataset` directory contains all scripts used for construction of the unannotated dataset and running the VeriFact AI system to generate AI ratings for each proposition extracted from Brief Hospital Course text narratives in the dataset.

```sh
# Create subset of patients from MIMIC-III and dataset
uv run python scripts/dataset/sample_mimic_dataset.py
```

The script produces the dataset in `verifact/data/dataset`. At this point, the patient study cohort has been selected, the original discharge summary has been separated from the rest of the clinical notes for these patients, and the original human-written Brief Hospital Course (BHC) has been extracted from the discharge summary. The remaining notes from each patient forms a patient-specific reference EHR.

### Generate LLM-written Brief Hospital Course (BHC)

```sh
# Produce LLM-written Brief Hospital Courses
uv run python scripts/dataset/write_llm_hospital_course.py
```

The LLM-written brief hospital courses are exported to the following paths:

* `verifact/data/dataset/bhc_note_text`
* `verifact/data/dataset/llm_bhc_noteevents.feather`

This script uses redis queue (RQ) to manage jobs and workers. Each worker is processing a single patient's set of clinical notes from the hospital admission. Each worker is able to make multiple asynchronous LLM API calls. This allows multiple LLM-written BHCs to be generated in parallel. `RQ-dashboard` (http://${SERVER_URL}:9181) shows the job progress.

To suspend RQ workers while job is running:

```sh
# suspend for 5 minutes
rq suspend --duration 300
```

### Decompose LLM-written and Human-written Brief Hospital Course (BHC) into Propositions

The human-written BHC and LLM-written BHC undergo text decomposition to yield sentence propositions and atomic claim propositions. This process is as follows:

1. BHC text narrative is converted into text chunks using semantic splitting.
2. Each text chunk is converted into sentences using NLTK sentence splitter.
3. Each text chunk is also converted into atomic claims using LLM-based atomic claim extraction.

Each text unit is saved as a LlamaIndex `node`.

This script uses redis queue (RQ) to manage jobs and workers. Each worker is able to process a single note which undergoes the transformation described above and may involve multiple embedding and LLM calls.

```sh
# Convert original Brief Hospital Course --> Nodes 
# Save Nodes to VectorDB
# Save Nodes to pickle files on disk
uv run python scripts/dataset/decompose_text.py --input-file="bhc_noteevents.feather" \
    --dataset-dir="/remote/home/chungp/Developer/verifact/data/dataset" \
    --upsert-db --collection-name="bhc_noteevents" \
    --save-pickle --output-dir-name="bhc_nodes" \
    --queue-name="human_bhc_decompose" --num-parallel-pipelines=100 --llm-n-jobs=100

# Convert LLM-generated Brief Hospital Course --> Nodes 
# Save Nodes to VectorDB
# Save Nodes to pickle files on disk
uv run python scripts/dataset/decompose_text.py --input-file="llm_bhc_noteevents.feather" \
    --dataset-dir="/remote/home/chungp/Developer/verifact/data/dataset" \
    --upsert-db --collection-name="llm_bhc_noteevents" \
    --save-pickle --output-dir-name="llm_bhc_nodes" \
    --queue-name="llm_bhc_decompose" --num-parallel-pipelines=100 --llm-n-jobs=100
```

### Decompose Patient's Reference EHR into Facts

All EHR notes undergo text decomposition to yield sentence facts and atomic claim facts. These facts are stored in a vector database for subsequent retrieval. The text decomposition process is exactly the same as for BHC.

```sh
# Convert EHR notes --> Nodes
# Save Nodes to VectorDB
# Skip saving to pickle files on disk
uv run python scripts/dataset/decompose_text.py --input-file="ehr_noteevents.feather" \
    --dataset-dir="/remote/home/chungp/Developer/verifact/data/dataset" \
    --upsert-db --collection-name="ehr_noteevents" \
    --no-save-pickle \
    --no-load-nodes-from-vectorstore-if-exists \
    --queue-name="ehr_note_decompose" --num-parallel-pipelines=100 --llm-n-jobs=100
```

Each text unit is saved as a LlamaIndex `node`.

This script uses redis queue (RQ) to manage jobs and workers. Each worker is able to process a single note which undergoes the transformation described above and may involve multiple embedding and LLM calls.

## Evaluation

### Run VeriFact to Produce AI-generated Labels for each Proposition

Evaluate each proposition against retrieved facts in the EHR. The same propositions from a BHC are evaluated together to generate a score report.

Arguments specifying a specific BHC:

* `subject_id`
* `input_text_kind`
* `node_kind`

VeriFact Hyperparameters:

* `retrieval_method`
* `reference_format`
* `reference_only_admission`
* `top_n`

Each combination of BHC arguments and VeriFact Hyperparameters yields a possible `ScoreReport`, which summarizes the degree to which a BHC is `Supported`, `Not Supported` or `Not Addressed`, provides explanations for each category, and also includes proposition-level verdicts and expalantions.

```sh
# Run Evaluation for all subject_ids, author_types, proposition_types and all hyperparameters
uv run python scripts/dataset/run_verifact.py

# Only run evaluation for specific subject_id, author_type, proposition_type
uv run python scripts/dataset/run_verifact.py --subject-ids=1084 --input-text-kinds=llm --node-kinds="claim" --retrieval-methods="hybrid" --reference-formats="absolute_time" --reference-only-admission=True top-ns=50
```

The evaluation outputs are saved as a pandas DataFrames:

* ScoreReport DataFrame: `data/dataset/judges/score_reports/score_report.feather`: Each row is a `ScoreReport` that is generated during evaluation. A ScoreReport is a single rater's evaluation of all propositions of a text input (e.g. one BHC for one Subject ID). Additional columns specify Subject ID, BHC type, and VeriFact Hyperparameters.
* Verdicts DataFrame: `data/dataset/judges/score_reports/verdicts.feather`: Each row is a `proposition` that has been assigned a `verdict` label (`Supported`, `Not Supported`, `Not Addressed`). Additional columns specify Subject ID, BHC type and VeriFact Hyperparameters. `proposition_id` uniquely identifies a proposition, which may be rated by multiple raters.
