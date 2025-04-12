# Creation of Unannotated Dataset and EHR Vector Database

The first portion of these instructions under `Uannotated Dataset Creation` contains steps to create the unannotated dataset for the `VeriFact` study and `VeriFact-BHC`, meaning it does not contain any human or AI annotations. Due to the use of LLMs and some inherent stochasticity, the results will not be identical to the `VeriFact-BHC` dataset. This is because the LLM-written Brief Hospital Course (BHC) narrative will likely contain different text and the decomposition of LLM-written and Human-written BHC may be slightly different.

The second portion of these instructions under `Create Reference Vector Database of EHR Facts for Evaluation` and `Evaluation` contains steps needed to run `VeriFact` on the annotated dataset we have created called `VeriFact-BHC`. The propositions from text decomposition are fixed in this dataset but derived from the same steps as `Unannotated Dataset Creation`. After the propositions were fixed, human clinicans annotated the dataset for ground truth labels to compare `VeriFact` against. You only need to run the code in the second portion to benchmark any modifications to `VeriFact` against the `VeriFact-BHC` dataset.

## Unannotated Dataset Creation

The following steps reproduce the creation of the unannotated dataset for the `VeriFact` study and `VeriFact-BHC`, meaning it does not contain any human or AI annotations.

The following steps also run the VeriFact AI system to generate AI ratings for each proposition extracted from Brief Hospital Course (BHC) text narratives in the dataset. However, the text decomposition may be slightly different than in `VeriFact-BHC` and our reported experiments due to the non-deterministic nature of LLMs, so refer to `analysis` scripts on downloading the exact text decomposition and annotations used in `VeriFact-BHC`

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

## Create Propositions for Dataset

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
    --dataset-dir="/remote/home/chungp/Developer/verifact-private/data/dataset" \
    --upsert-db --collection-name="bhc_noteevents" \
    --save-pickle --output-dir-name="bhc_nodes" \
    --queue-name="human_bhc_decompose" --num-parallel-pipelines=50 --llm-n-jobs=32

# Convert LLM-generated Brief Hospital Course --> Nodes 
# Save Nodes to VectorDB
# Save Nodes to pickle files on disk
uv run python scripts/dataset/decompose_text.py --input-file="llm_bhc_noteevents.feather" \
    --dataset-dir="/remote/home/chungp/Developer/verifact-private/data/dataset" \
    --upsert-db --collection-name="llm_bhc_noteevents" \
    --save-pickle --output-dir-name="llm_bhc_nodes" \
    --queue-name="llm_bhc_decompose" --num-parallel-pipelines=50 --llm-n-jobs=32
```

NOTE: that the scripts up to this point produces the unannotated dataset and is presented here to illustrate how the VeriFact propositions are created.

## Create Reference Vector Database of EHR Facts for Evaluation

### From Unannotated Dataset

For each patient, all EHR notes undergo text decomposition to yield sentence facts and atomic claim facts. These facts are stored in a vector database for subsequent retrieval. The text decomposition process is exactly the same as for decomposing the LLM-written and Human-written BHC.

```sh
# Convert EHR notes --> Nodes
# Save Nodes to VectorDB
# Skip saving to pickle files on disk
uv run python scripts/dataset/decompose_text.py --input-file="ehr_noteevents.feather" \
    --dataset-dir="/remote/home/chungp/Developer/verifact-private/data/dataset" \
    --upsert-db --collection-name="ehr_noteevents" \
    --no-save-pickle \
    --no-load-nodes-from-vectorstore-if-exists \
    --queue-name="ehr_note_decompose" --num-parallel-pipelines=32 --llm-n-jobs=16
```

Each text unit is saved as a LlamaIndex `node`.

This script uses redis queue (RQ) to manage jobs and workers. Each worker is able to process a single note which undergoes the transformation described above and may involve multiple embedding and LLM calls.

### From Annotated VeriFact-BHC Dataset

All of a patient's EHR notes for the `VeriFact-BHC` dataset are also available as part of the Annotated `VeriFact-BHC` dataset that can be downloaded from physionet.

In order to download the dataset, please complete the required CITI training and Data Use Agreement: <https://physionet.org/content/mimic-iii-ext-verifact-bhc/1.0.0/>

```sh
# Download Dataset with Human Annotations from Physionet
# Replace verifact with this repository's root path
# Replace ${PHYSIONET_USERNAME} with your physionet username
wget -r -N -c -np --directory-prefix=data --user ${PHYSIONET_USERNAME} --ask-password https://physionet.org/files/mimic-iii-ext-verifact-bhc/1.0.0/

# Downloaded dataset is at path: verifact/data/physionet.org/files/mimic-iii-ext-verifact-bhc/1.0.0/

# Convert EHR notes --> Nodes
# Save Nodes to VectorDB
# Skip saving to pickle files on disk
uv run python scripts/dataset/decompose_text.py --input-file="ehr_noteevents.csv.gz" \
    --dataset-dir="data/physionet.org/files/mimic-iii-ext-verifact-bhc/1.0.0/reference_ehr" \
    --upsert-db --collection-name="ehr_noteevents" \
    --no-save-pickle \
    --no-load-nodes-from-vectorstore-if-exists \
    --queue-name="ehr_note_decompose" --num-parallel-pipelines=32 --llm-n-jobs=16
```
