# Scripts for Dataset Creation, VeriFact Evaluation, Experimental Analysis

* `/scripts/dataset`: Scripts required to generate the unannotated `VeriFact-BHC` dataset, including LLM-written BHC, perform proposition extraction, perform Electronic Health Record (EHR) fact extraction and store in vector database, run evaluation on LLM-written BHC and human-written BHC.
* `/scripts/evaluate`: Run experiments to evaluate `VeriFact` with various hyperparameter combinations and models used as LLM-as-a-Judge. This requires downloading the annotated `VeriFact-BHC` dataset for human ground truth labels. It also requires ingestion of EHR facts for each SubjectID for which a script is provided in `/scripts/dataset`. Also contains script to run `VeriFact` to evaluate any arbitrary text written about a patient against facts for that patient stored in the EHR vector database.
* `/scripts/analysis_dataset`: Examine the validity of propositions in the Annotated `VeriFact-BHC` dataset.
* `/scripts/analysis_interrater`: Run interrater agreement analysis on human clinician annotations in the Annotated `VeriFact-BHC` dataset.
* `/scripts/analysis_verifact`: Run `VeriFact` system to produce AI labels and compare against human clinician ground truth annotations in the Annotated `VeriFact-BHC` dataset.

[!NOTE]
Weaker LLMs like Llama 3.1 8B may not have sufficient modeling capacity and instruction following capability to adhere to the complex prompts and perform the required tasks (especially atomic claim extraction). Recommend using stronger LLMs like Llama 3.1 70B.
