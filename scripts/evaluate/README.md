# Run VeriFact

## Download Annotated Dataset

Evaluation involves using `VeriFact` to generate AI annotations for each proposition and compare them against human annotations. This requires human clinician annotations, which were generated separately and is not included in the scripts in `scripts/dataset`.

The Annotated `VeriFact-BHC` Dataset can be downloaded from PhysioNet. In order to download the dataset, please complete the required CITI training and Data Use Agreement: <https://physionet.org/content/mimic-iii-ext-verifact-bhc/1.0.0/>

```sh
# Download Dataset with Human Annotations from Physionet
# Replace verifact with this repository's root path
# Replace ${PHYSIONET_USERNAME} with your physionet username
wget -r -N -c -np --directory-prefix=data --user ${PHYSIONET_USERNAME} --ask-password https://physionet.org/files/mimic-iii-ext-verifact-bhc/1.0.0/

# Downloaded dataset is at path: verifact/data/physionet.org/files/mimic-iii-ext-verifact-bhc/1.0.0/
```

In the dataset, we fix the propositions extracted from the long-form input text to make benchmarking consistent. If you want to run VeriFact to fact-check any arbitrary text written about one of the patients, please use `scripts/evaluate/run_verifact.py`.

## Run VeriFact to Produce AI-generated Labels for each Proposition

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

Each combination of BHC arguments and VeriFact Hyperparameters yields a possible `ScoreReport`, which summarizes the degree to which a BHC is `Supported`, `Not Supported` or `Not Addressed`, provides explanations for each category, and also includes proposition-level verdicts and explantions.

```sh
# Run Evaluation for all subject_ids, author_types, proposition_types and all hyperparameters
uv run python scripts/evaluate/run_verifact.py

# Only run evaluation for specific subject_id, author_type, proposition_type
uv run python scripts/evaluate/run_verifact.py --subject-ids=1084 --author_type=llm --proposition_type=claim --retrieval-method=hybrid --reference-format=absolute_time --reference-only-admission --top-n=50
```

## VeriFact Experiments

Below contain sets of CLI commands used to run the `VeriFact` experiments.

The evaluation outputs are saved as a pandas DataFrames:

* **ScoreReport DataFrame:** `data/dataset/judges/score_reports/score_report.feather`: Each row is a `ScoreReport` that is generated during evaluation. A ScoreReport is a single rater's evaluation of all propositions of a text input (e.g. one BHC for one Subject ID). Additional columns specify Subject ID, BHC type, and VeriFact Hyperparameters.
* **Verdicts DataFrame:** `data/dataset/judges/score_reports/verdicts.feather`: Each row is a `proposition` that has been assigned a `verdict` label (`Supported`, `Not Supported`, `Not Addressed`). Additional columns specify Subject ID, BHC type and VeriFact Hyperparameters. `proposition_id` uniquely identifies a proposition, which may be rated by multiple raters.

### Non-Reasoning Models

CLI Command to Run All Experiments for Non-Reasoning Model for a Single Model
For each of these, adjust in `.env` LLM_MODEL_NAME and VERIFACT_RESULTS_DIR. The addition of `--output-subdir` argument will output the evaluation results in its own subdirectory.

Models:

1. `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`  (`--output-subdir="verifact_llama3_1_8B"`)
2. `hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4` (`--output-subdir="verifact_llama3_1_70B"`)

#### Author Type: LLM

All experimental runs with:

* Proposition Type: `Claim`, Fact Type: `Claim`
* Proposition Type: `Claim`, Fact Type: `Sentence`

```sh
# Top N Experiments
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts5 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts10 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts25 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts50 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts75 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts100 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts125 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=125 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts150 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=150 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts5 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts10 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts25 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts50 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts75 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts100 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission
```

```sh
# Retrieval Method, Reference Format, Reference Only Admission Experiments
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-RetrievalMethod --author-type=llm --proposition-type=claim --fact-type=claim --top-n=50 --retrieval-method=dense --retrieval-method=sparse --retrieval-method=hybrid --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\ 
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-ReferenceFormat --author-type=llm --proposition-type=claim --fact-type=claim --top-n=50 --retrieval-method=rerank --reference-format=score --reference-format=absolute_time --reference-format=relative_time --reference-only-admission;\ 
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-ReferenceOnlyAdmission --author-type=llm --proposition-type=claim --fact-type=claim --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --no-reference-only-admission
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-RetrievalMethod --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=50 --retrieval-method=dense --retrieval-method=sparse --retrieval-method=hybrid --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\ 
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-ReferenceFormat --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=50 --retrieval-method=rerank --reference-format=score --reference-format=absolute_time --reference-format=relative_time --reference-only-admission;\ 
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-ReferenceOnlyAdmission --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --no-reference-only-admission
```

#### Author Type: LLM

All experimental runs with:

* Proposition Type: `Sentence`, Fact Type: `Claim`
* Proposition Type: `Sentence`, Fact Type: `Sentence`

```sh
# Top N Experiments
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts5 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts10 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts25 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts50 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts75 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts100 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts125 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=125 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts150 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=150 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts5 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts10 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts25 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts50 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts75 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts100 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission
```

#### Author Type: Human

All experimental runs with:

* Proposition Type: `Claim`, Fact Type: `Claim`
* Proposition Type: `Claim`, Fact Type: `Sentence`

```sh
# Top N Experiments
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts5 --author-type=human --proposition-type=claim --fact-type=claim --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts10 --author-type=human --proposition-type=claim --fact-type=claim --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts25 --author-type=human --proposition-type=claim --fact-type=claim --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts50 --author-type=human --proposition-type=claim --fact-type=claim --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts75 --author-type=human --proposition-type=claim --fact-type=claim --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts100 --author-type=human --proposition-type=claim --fact-type=claim --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts125 --author-type=human --proposition-type=claim --fact-type=claim --top-n=125 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts150 --author-type=human --proposition-type=claim --fact-type=claim --top-n=150 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts5 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts10 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts25 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts50 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts75 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts100 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission
```

### R1 Distilled Reasoning Models

CLI Command to Run All Experiments for Reasoning Model for a Single Model As of vLLM v0.7.2, reasoning models are not compatible with Structured Outputs, so we chain the output of the reasoning model with an auxillary LLM (Llama 3.1 8B) to obtain structured outputs. This logic is activated with the flag `--is-reasoning-model`

For each of these, adjust in `.env` LLM_MODEL_NAME and VERIFACT_RESULTS_DIR
AUX_LLM_MODEL_NAME is fixed to `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`. The addition of `--output-subdir` argument will output the evaluation results in its own subdirectory.

Models:

1. `casperhansen/deepseek-r1-distill-llama-8b-awq` (`--output-subdir="verifact_deepseek_r1_distill_llama_8B"`)
2. `casperhansen/deepseek-r1-distill-llama-70b-awq` (`--output-subdir="verifact_deepseek_r1_distill_llama_70B"`)

#### Author Type: LLM

All experimental runs with:

* Proposition Type: `Claim`, Fact Type: `Claim`
* Proposition Type: `Claim`, Fact Type: `Sentence`

```sh
# Top N Experiments
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts5 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts10 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts25 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts50 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts75 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts100 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts125 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=125 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-NumFacts150 --author-type=llm --proposition-type=claim --fact-type=claim --top-n=150 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts5 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts10 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts25 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts50 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts75 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-NumFacts100 --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model
```

```sh
# Retrieval Method, Reference Format, Reference Only Admission Experiments
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-RetrievalMethod --author-type=llm --proposition-type=claim --fact-type=claim --top-n=50 --retrieval-method=dense --retrieval-method=sparse --retrieval-method=hybrid --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\ 
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-ReferenceFormat --author-type=llm --proposition-type=claim --fact-type=claim --top-n=50 --retrieval-method=rerank --reference-format=score --reference-format=absolute_time --reference-format=relative_time --reference-only-admission --is-reasoning-model;\ 
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimClaim-ReferenceOnlyAdmission --author-type=llm --proposition-type=claim --fact-type=claim --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model --no-reference-only-admission
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-RetrievalMethod --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=50 --retrieval-method=dense --retrieval-method=sparse --retrieval-method=hybrid --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\ 
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-ReferenceFormat --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=50 --retrieval-method=rerank --reference-format=score --reference-format=absolute_time --reference-format=relative_time --reference-only-admission --is-reasoning-model;\ 
uv run python scripts/evaluate/run_verifact.py --run-name=LLMClaimSentence-ReferenceOnlyAdmission --author-type=llm --proposition-type=claim --fact-type=sentence --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model --no-reference-only-admission
```

#### Author Type: LLM

All experimental runs with:

* Proposition Type: `Sentence`, Fact Type: `Claim`
* Proposition Type: `Sentence`, Fact Type: `Sentence`

```sh
# Top N Experiments
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts5 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts10 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts25 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts50 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts75 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts100 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts125 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=125 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceClaim-NumFacts150 --author-type=llm --proposition-type=sentence --fact-type=claim --top-n=150 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts5 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts10 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts25 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts50 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts75 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=LLMSentenceSentence-NumFacts100 --author-type=llm --proposition-type=sentence --fact-type=sentence --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model
```

#### Author Type: Human

All experimental runs with:

* Proposition Type: `Claim`, Fact Type: `Claim`
* Proposition Type: `Claim`, Fact Type: `Sentence`

```sh
# Top N Experiments
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts5 --author-type=human --proposition-type=claim --fact-type=claim --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts10 --author-type=human --proposition-type=claim --fact-type=claim --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts25 --author-type=human --proposition-type=claim --fact-type=claim --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts50 --author-type=human --proposition-type=claim --fact-type=claim --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts75 --author-type=human --proposition-type=claim --fact-type=claim --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts100 --author-type=human --proposition-type=claim --fact-type=claim --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts125 --author-type=human --proposition-type=claim --fact-type=claim --top-n=125 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimClaim-NumFacts150 --author-type=human --proposition-type=claim --fact-type=claim --top-n=150 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts5 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=5 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts10 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=10 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts25 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=25 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts50 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=50 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts75 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=75 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model;\
uv run python scripts/evaluate/run_verifact.py --run-name=HumanClaimSentence-NumFacts100 --author-type=human --proposition-type=claim --fact-type=sentence --top-n=100 --retrieval-method=rerank --reference-format=absolute_time --reference-only-admission --is-reasoning-model
```

## Run VeriFact on Arbitrary Text for a Patient in the Dataset

For any Subject ID stored in the EHR vector database, run `VeriFact` to evaluate any arbitrary text written about that patient. This assumes the patient's EHR facts have been ingested into the vector database and is available for retrieval which is established under section `Decompose Patient's Reference EHR into Facts` in the `README.md` file in `scripts/dataset`.

```sh
# Evaluate Text and Save Score Report to disk. Must explicitly specify patient's Subject ID
uv run scripts/evaluate/run_verifact2.py --text="Patient had sepsis, encephalopathy, and agitation, requiring the patient to be admitted to the ICU. When his confusion cleared, he was found to have taken Oxycodone instead of methadone." --subject-id=1084 --output-file=score_report.pkl
```

```python
# Load Score Report
from utils import load_pickle

sr = load_pickle("score_report.pkl")

print(sr.report())
# Supported Score: 0.80
# Not Supported Score: 0.20
# Not Addressed Score: 0.00
# Supported Count: 4
# Not Supported Count: 1
# Not Addressed Count: 0
# Total Count: 5
# Supported Explanation: The text is supported by the reference context due to the following reasons: the patient's sepsis is confirmed by infectious complications and UTI, encephalopathy is supported by multiple references to altered mental status, agitation is verified by multiple entries describing the patient's agitated state, and ICU admission is confirmed by references to ICU care and intubation.
# Not Supported Explanation: The text is not supported by the reference context because it incorrectly states the patient took Oxycodone instead of methadone, when in fact the reference context mentions the patient took Ambien instead of methadone.
# Not Addressed Explanation: No reasons provided for the text not being addressed by the reference context.
# Proposition Type: claim
# Fact Type: claim
# Judge Configuration: {'subject_id': 1084, 'author_type': 'llm', 'proposition_type': 'claim', 'fact_type': 'claim', 'retrieval_method': 'rerank', 'top_n': 50, 'reference_format': 'absolute_time', 'reference_only_admission': True, 'deduplicate_text': True}
```

Alternative ways to run verifact on arbitrary text.

```sh
# Load Text from File, Evaluate and Save Score Report to disk. Must explicitly specify patient's Subject ID
uv run scripts/evaluate/run_verifact2.py --text-file={path_to_textfile}.txt --subject-id=1084 --output-file=score_report.pkl

# Specify atomic claim proposition
uv run scripts/evaluate/run_verifact2.py --text-file={path_to_textfile}.txt --subject-id=1084 --output-file=score_report.pkl --proposition-type=claim

# Specify sentence proposition
uv run scripts/evaluate/run_verifact2.py --text-file={path_to_textfile}.txt --subject-id=1084 --output-file=score_report.pkl --proposition-type=sentence

# Save Score Report as a dataframe instead of a pickled ScoreReport object.
uv run scripts/evaluate/run_verifact2.py --text-file={path_to_textfile}.txt --subject-id=1084 --output-file=score_report.csv
```
