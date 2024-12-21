# Evaluate Any Text for a Patient

For any Subject ID stored in the EHR vector database, run VeriFact to evaluate any arbitrary text written about that patient. This assumes the patient's EHR facts have been ingested into the vector database and is available for retrieval which is established under section `Decompose Patient's Reference EHR into Facts` in the `README.md` file in `scripts/dataset`.

## Run Evaluation

```sh
# Evaluate Text and Save Score Report to disk. Must explicitly specify patient's Subject ID
uv run run_verifact.py --text="Patient had sepsis, encephalopathy, and agitation, requiring the patient to be admitted to the ICU. When his confusion cleared, he was found to have taken Oxycodone instead of methadone." --subject-id=1084 --output-file=score_report.pkl

# Load Text from File, Evaluate and Save Score Report to disk. Must explicitly specify patient's Subject ID
uv run run_verifact.py --text-file={path_to_textfile}.txt --subject-id=1084 --output-file=score_report.pkl

# Specify atomic claim proposition
uv run run_verifact.py --text-file={path_to_textfile}.txt --subject-id=1084 --output-file=score_report.pkl --proposition-type=claim

# Specify sentence proposition
uv run run_verifact.py --text-file={path_to_textfile}.txt --subject-id=1084 --output-file=score_report.pkl --proposition-type=sentence

# Save Score Report as a dataframe instead of a pickled ScoreReport object.
uv run run_verifact.py --text-file={path_to_textfile}.txt --subject-id=1084 --output-file=score_report.csv
```
