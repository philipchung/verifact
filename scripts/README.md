# Scripts for Dataset Creation, VeriFact Evaluation, Experimental Analysis

Weaker LLMs like Llama 3.1 8B may not have sufficient modeling capacity and instruction following capability to adhere to the complex prompts and perform the required tasks (especially atomic claim extraction). Recommend using stronger LLMs like Llama 3.1 70B.

* `/scripts/dataset`: Scripts required to generate the unannotated dataset, including LLM-written BHC, perform proposition extraction, perform EHR fact extraction and store in vector database, run evaluation on LLM-written BHC and human-written BHC.
* `/scripts/evaluate`: For any Subject ID stored in the EHR vector database, run VeriFact to evaluate any arbitrary text written about that patient.
* `/scripts/analysis`: Download annotated dataset and run statistical analysis.
