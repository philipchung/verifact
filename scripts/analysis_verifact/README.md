# Comparing Agreement & Metrics: VeriFact AI-Generated Labels vs. Human Clinican Ground Truth

## PhysioNet v1.1.0 data

Set `VERIFACTBHC_DATASET_DIR` to the root of the PhysioNet v1.1.0 export before running the analysis scripts. The compatibility helpers in `release_data.py` load the normalized proposition, rater-configuration, verdict, and reference-payload tables from that root while preserving the historical analysis model names.

Default annotation loads do not include reference text. Reference lengths are calculated from the payload table in bounded-memory batches, and full reference payloads are attached only after an analysis has selected the examples it will display. This keeps the plotting and aggregate-analysis paths from materializing all reference text in memory.

The v1.1.0 release includes preliminary sensitivity configurations only for `Llama-8B`, `Llama-70B`, `R1-8B`, and `R1-70B`. Gemma and Qwen models contain curated main configurations, so missing sensitivity rows for those models reflect release coverage rather than adapter data loss.

Scripts 2 through 12 analyze saved verdicts and do not require an LLM inference server or a vector database. Script 0 audits the historical raw run directories. Parts of script 1 use the original MIMIC-III source data and project vector database and are therefore separate from the release-only reproduction path.

## 0. Completeness Check

The `0_check_completeness.py` contains notebook cells for checking completeness of raw `VeriFact` experiment runs. It intentionally continues to inspect the pre-release run directories rather than the PhysioNet export.

## 1. VeriFact-BHC Dataset Statistics

The `1_verifact_BHC_dataset_statistics.py` contains notebook cells for computing dataset statistics.

## 2. Compute VeriFact Agreement & Metrics

The `2_compute_verifact_metrics.py` is a script that computes agreement & metrics for different `VeriFact` raters against the human clinican ground truth labels.

1. Percent Agreement
2. Gwet's AC1
3. Matthew's Correlation Coefficient (MCC)
4. Supported - True Positive Rate (TPR)
5. Supported - True Negative Rate (TNR)
6. Supported - Positive Predictive Value (PPV)
7. Supported - Negative Predictive Value (NPV)
8. Not Supported - True Positive Rate (TPR)
9. Not Supported - True Negative Rate (TNR)
10. Not Supported - Positive Predictive Value (PPV)
11. Not Supported - Negative Predictive Value (NPV)
12. Not Addressed - True Positive Rate (TPR)
13. Not Addressed - True Negative Rate (TNR)
14. Not Addressed - Positive Predictive Value (PPV)
15. Not Addressed - Negative Predictive Value (NPV)

```sh
## Computed with Original Label Space
# Compute Percent Agreement
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="PercentAgreement" --metric="percent_agreement" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000

# Compute Gwet's AC1
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="GwetAC1" --metric="gwet" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000

# Compute MCC
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="MCC" --metric="mcc" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000

# Compute Supported TPR, TNR, PPV, NPV
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="SupportedTPR" --metric="s-tpr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="SupportedTNR" --metric="s-tnr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="SupportedPPV" --metric="s-ppv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="SupportedNPV" --metric="s-npv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000

# Compute NotSupported TPR, TNR, PPV, NPV
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotSupportedTPR" --metric="ns-tpr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotSupportedTNR" --metric="ns-tnr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotSupportedPPV" --metric="ns-ppv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotSupportedNPV" --metric="ns-npv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000

# Compute NotAddressed TPR, TNR, PPV, NPV
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotAddressedTPR" --metric="na-tpr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotAddressedTNR" --metric="na-tnr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotAddressedPPV" --metric="na-ppv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotAddressedNPV" --metric="na-npv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000
```

```sh
## Computed with Binarized Label Space
# Compute Percent Agreement w/ Binarized Label Space
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="PercentAgreement-Binarized" --metric=percent_agreement --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels

# Compute Gwet's AC1 w/ Binarized Label Space
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="GwetAC1-Binarized" --metric=gwet --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels

# Compute MCC w/ Binarized Label Space
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="MCC" --metric=mcc --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels

# Compute Supported TPR, TNR, PPV, NPV
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="SupportedTPR" --metric="s-tpr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="SupportedTNR" --metric="s-tnr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="SupportedPPV" --metric="s-ppv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="SupportedNPV" --metric="s-npv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels

# Compute NotSupported TPR, TNR, PPV, NPV
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotSupportedTPR" --metric="ns-tpr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotSupportedTNR" --metric="ns-tnr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotSupportedPPV" --metric="ns-ppv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotSupportedNPV" --metric="ns-npv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels

# Compute NotAddressed TPR, TNR, PPV, NPV
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotAddressedTPR" --metric="na-tpr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotAddressedTNR" --metric="na-tnr" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotAddressedPPV" --metric="na-ppv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels;\
uv run python scripts/analysis_verifact/2_compute_verifact_metrics.py --run-name="NotAddressedNPV" --metric="na-npv" --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels
```

## 3. Sensitivity Analysis Plots for Top N, Retrieval Method, Reference Context Format, Reference Only Admission

The `3_sensitivity_analysis_plots.py` contain notebook cells for plotting sensitivity analysis of different `VeriFact` AI raters and hyperparameter combinations against the Percent Agreement with human clinician ground truth labels.

These are preliminary sensitivity analyses to identify the optimal value for the following hyperparameters in the `VeriFact` system:

* `Top N`: Number of facts to retrieve from EHR Vector Database when evaluating propositions. (`N` = `5`, `10`, `25`, `50`, `75`, `100`, `125`, `150`).
* `Retrieval Method`: Retrieval method to use (`dense`, `sparse`, `hybrid`, `rerank`)
* `Reference Context Format`: Text format for reference context containing retrieved EHR facts when passed to LLM-as-a-Judge (`score`, `absolute time`, `relative time`).
* `Reference Only Admission`: Whether to limit retrieval of EHR facts only to current hospital admission (`True`, `False`).

## 4. Agreement vs. Reference Context Length (and Number of EHR Facts)

The `4_agreement_mcc_plots.py` contains notebook cells for plotting `VeriFact` AI rater agreement and Matthews Correlation Coefficient against reference-context length. Both `sentence` and `claim` EHR facts are examined. Since the average length of each `sentence` and `claim` EHR fact is different, the X-axis is selected as the average word length for the reference context to ensure both scenarios are directly comparable.

## 5. Best Model Sensitivity, Specificity, PPV, NPV

The `5_best_model_classification_metrics.py` contains notebook cells that compute the sensitivity, specificity, positive predictive value, and negative predictive value for each label (`Supported`, `Not Supported`, `Not Addressed`) when compared against the human clinician ground truth labels.

## 6. Metric Result Tables

The `6_metric_result_tables.py` contains notebook cells for exporting agreement, MCC, and per-class classification metrics with computed 95% confidence intervals for all studied `VeriFact` system configurations. These were computed using the script in `2_compute_verifact_metrics.py`.

Here is a list of hyperparameters:

* `Author Type`: Whether Brief Hospital Course (BHC) text is written by `llm` or `human`
* `Proposition Type`: Whether BHC is decomposed into `sentence` or atomic `claim` propositions for evaluation with the `VeriFact` system.
* `Fact Type`: Whether EHR fact retrieved from the vector database is a `sentence` or atomic `claim` fact.
* `Model`: Model used for LLM-as-a-Judge (`Gemma3-12B`, `Gemma3-27B`, `Qwen3-30B-A3B-Instruct`, `Qwen3-30B-A3B-Thinking`, `Qwen3-32B`, `R1-8B`, `R1-70B`, `Llama-8B`, `Llama-70B`)
* `Top N`: Number of facts to retrieve from EHR Vector Database when evaluating propositions. (`N` = `5`, `10`, `25`, `50`, `75`, `100`, `125`, `150`).
* `Retrieval Method`: Retrieval method to use (`dense`, `sparse`, `hybrid`, `rerank`)
* `Reference Context Format`: Text format for reference context containing retrieved EHR facts when passed to LLM-as-a-Judge (`score`, `absolute time`, `relative time`).
* `Reference Only Admission`: Whether to limit retrieval of EHR facts only to current hospital admission (`True`, `False`).

## 7. Error Analysis

The `7_error_analysis.py` contains notebook cells for selecting specific examples from the dataset for analysis on errors made by `VeriFact` when using different LLM-as-a-Judge (e.g. small vs. large models, reasoning vs. non-reasoning models).

## 8. Score Sheet Figure

The `8_score_sheet_figure.py` contains notebook cells for selecting specific examples from the dataset to illustrate the `VeriFact` score sheet which is shown as a figure in the manuscript.

## 9. Confusion Matrices

The `9_confusion_matrices.py` notebook script generates multiclass confusion matrices for all nine LLM judges. The `9a_confusion_matrices_llama_vs_R1.py` notebook script generates paired correctness matrices used to compare model size and reasoning capability for the Llama and DeepSeek-R1-Distill-Llama families.

## 10. Proposition Analysis

The `10_examine_propositions.py` notebook script summarizes proposition validity and human annotation characteristics used in the supplemental analyses.

## 11. Metrics Stratified by Clinician Agreement

The `11_compute_verifact_metrics_stratified.py` notebook script computes percent agreement, Gwet's AC1, and MCC for the selected Qwen Thinking configuration after stratifying propositions by round-one clinician agreement.

## 12. Stratified Metric Tables

The `12_analyze_verifact_agreement_stratified.py` notebook script converts the stratified metric results from script 11 into the contingency tables used for reporting.
