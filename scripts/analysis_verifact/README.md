# Comparing Agreement: VeriFact AI-Generated Labels vs. Human Clinican Ground Truth

## 0. Completeness Check

The `0_check_completeness.py` contains notebook cells for checking completeness of `VeriFact` experiment runs.

## 1. VeriFact-BHC Dataset Statistics

The `1_verifact_BHC_dataset_statistics.py` contains notebook cells for computing dataset statistics.

## 2. Compute VeriFact Agreement

The `2_compute_verifact_agreement.py` is a script that computes agreement of different `VeriFact` raters against the human clinican ground truth labels.

Two types of inter-rater agreement metrics are computed:

1. Percent Agreement
2. Gwet's AC1

```sh
# Compute Percent Agreement
uv run python scripts/analysis_verifact/2_compute_verifact_agreement.py --run-name="VeriFact_PercentAgreement" --metric=percent_agreement --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000

# Compute Gwet's AC1
uv run python scripts/analysis_verifact/2_compute_verifact_agreement.py --run-name="VeriFact_GwetAC1" --metric=gwet --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000

# Compute Percent Agreement w/ Binarized Label Space
uv run python scripts/analysis_verifact/2_compute_verifact_agreement.py --run-name="VeriFact_PercentAgreement" --metric=percent_agreement --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels

# Compute Gwet's AC1 w/ Binarized Label Space
uv run python scripts/analysis_verifact/2_compute_verifact_agreement.py --run-name="VeriFact_GwetAC1" --metric=gwet --workers=20 --num-parallel-raters=4 --bootstrap-iterations=1000 --binarize-labels
```

## 3. Sensitivity Analysis Plots for Top N, Retrieval Method, Reference Context Format, Reference Only Admission

The `3_sensitivity_analysis_plots.py` contain notebook cells for plotting sensitivity analysis of different `VeriFact` AI raters and hyperparameter combinations against the Percent Agreement with human clinician ground truth labels.

These are preliminary sensitivity analyses to identify the optimal value for the following hyperparameters in the `VeriFact` system:

* `Top N`: Number of facts to retrieve from EHR Vector Database when evaluating propositions. (`N` = `5`, `10`, `25`, `50`, `75`, `100`, `125`, `150`).
* `Retrieval Method`: Retrieval method to use (`dense`, `sparse`, `hybrid`, `rerank`)
* `Reference Context Format`: Text format for reference context containing retrieved EHR facts when passed to LLM-as-a-Judge (`score`, `absolute time`, `relative time`).
* `Reference Only Admission`: Whether to limit retrieval of EHR facts only to current hospital admission (`True`, `False`).

## 4. Agreement vs. Reference Context Length (and Number of EHR Facts)

The `4_top_n_vs_reference_context_length.py` contain notebook cells for plotting `VeriFact` AI rater agreement with human clinician ground truth labels. Both `sentence` and `claim` EHR facts are examined. Since the average length of each `sentence` and `claim` EHR fact is different, the X-axis is selected as the average word length for the reference context to ensure both scenarios are directly comparable.

## 5. Best Model Sensitivity, Specificity, PPV, NPV

The `5_best_model_sens_spec_ppv_npv.py` contains notebook cells that computes the sensitivity, specificity, positive predictive value, and negative predictive value for each label (`Supported`, `Not Supported`, `Not Addressed`) when compared against the human clinician ground truth labels.

## 6. Metric Result Tables

The `6_metric_result_tables.py` contains notebook cells for exporting the computed Percent Agreement and Gwet's AC1 with computed 95% confidence intervals for all hyperparameter combinations of `VeriFact` system studied in the experiments. These were computed using the script in `2_compute_verifact_agreement.py`.

Here is a list of hyperparameters:

* `Author Type`: Whether Brief Hospital Course (BHC) text is written by `llm` or `human`
* `Proposition Type`: Whether BHC is decomposed into `sentence` or atomic `claim` propositions for evaluation with the `VeriFact` system.
* `Fact Type`: Whether EHR fact retrieved from the vector database is a `sentence` or atomic `claim` fact.
* `Model`: Model used for LLM-as-a-Judge (`Llama-8B`, `Llama-70B`, `R1-8B`, `R1-70B`)
* `Top N`: Number of facts to retrieve from EHR Vector Database when evaluating propositions. (`N` = `5`, `10`, `25`, `50`, `75`, `100`, `125`, `150`).
* `Retrieval Method`: Retrieval method to use (`dense`, `sparse`, `hybrid`, `rerank`)
* `Reference Context Format`: Text format for reference context containing retrieved EHR facts when passed to LLM-as-a-Judge (`score`, `absolute time`, `relative time`).
* `Reference Only Admission`: Whether to limit retrieval of EHR facts only to current hospital admission (`True`, `False`).

## 7. Error Analysis

The `7_error_analysis.py` contains notebook cells for selecting specific examples from the dataset for analysis on errors made by `VeriFact` when using different LLM-as-a-Judge (e.g. small vs. large models, reasoning vs. non-reasoning models).

## 8. Score Sheet Figure

The `8_score_sheet_figure.py` contains notebook cells for selecting specific examples from the dataset to illustrate the `VeriFact` score sheet which is shown as a figure in the manuscript.
