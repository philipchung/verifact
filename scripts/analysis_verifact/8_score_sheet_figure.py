# %% [markdown]
# ## Get Text Examples for Score Sheet Figure in Manuscript
# %%
import os
from pathlib import Path

import pandas as pd
from irr_metrics import coerce_types
from utils import load_environment, load_pandas

load_environment()
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

# Load Human Clinician Verdict Labels (One Row Per Proposition)
human_verdicts = load_pandas(
    Path(os.environ["VERIFACTBHC_PROPOSITIONS_DIR"]) / "human_verdicts.csv.gz"
)
human_verdicts = coerce_types(human_verdicts)
# Isolate Human Ground Truth Labels
human_gt = (
    human_verdicts.assign(rater_name="human_gt")
    .astype({"rater_name": "string"})
    .rename(columns={"human_gt": "verdict"})
    .loc[
        :,
        ["proposition_id", "text", "author_type", "proposition_type", "rater_name", "verdict"],
    ]
)

# Map of VeriFact Result Directories for Each Model
model_dir_map = {
    "Llama-8B": "verifact_llama3_1_8B",
    "Llama-70B": "verifact_llama3_1_70B",
    "R1-8B": "verifact_deepseek_r1_distill_llama_8B",
    "R1-70B": "verifact_deepseek_r1_distill_llama_70B",
}
# Enumerate Unique Models
models = list(model_dir_map.keys())
# Load VeriFact Labels for Different Models
model_data_dict: dict[str, pd.DataFrame] = {}
for model, model_dir in model_dir_map.items():
    model_data_dict[model] = coerce_types(
        load_pandas(
            Path(os.environ["VERIFACT_RESULTS_DIR"])
            / model_dir
            / "score_reports"
            / "verdicts.feather"
        )
    )

# Combine all VeriFact Labels into a Single DataFrame
ai_verdicts = coerce_types(
    pd.concat(
        model_data_dict,
        axis="index",
        names=("model", "proposition_id"),
    )
    .reset_index()
    .astype({"model": "string"})
)
ai_verdicts = ai_verdicts.assign(
    rater_name=ai_verdicts.apply(lambda row: f"model={row.model},{row.rater_name}", axis="columns")
)

# %%
subject_ids = ai_verdicts.query(
    "author_type == 'llm' & text.str.contains('77')"
).subject_id.unique()
subject_ids
# %%
subject_id = 31135
num_facts = 25
df = ai_verdicts.query(
    f"subject_id == {subject_id} & author_type == 'llm' & "
    f"proposition_type == 'claim' & fact_type == 'sentence' & "
    f"top_n == {num_facts}"
)
df.shape

# %%
supported_proposition = df.loc[
    df.text == "The patient had a history of chronic obstructive pulmonary disease."
].squeeze()
print("Proposition Text:")
print(supported_proposition.text)
print("")
print("Reference Context:")
print(supported_proposition.reference)
# %%
not_supported_proposition = df.loc[
    df.text == "The patient was intubated due to worsening septic shock."
].squeeze()
print("Proposition Text:")
print(not_supported_proposition.text)
print("")
print("Reference Context:")
print(not_supported_proposition.reference)
# %%
