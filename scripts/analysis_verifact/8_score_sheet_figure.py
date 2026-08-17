# %% [markdown]
# ## Get Text Examples for Score Sheet Figure in Manuscript
# %%
import os
from pathlib import Path

import pandas as pd
from release_data import attach_reference_payloads, load_release_annotations
from utils import load_environment

load_environment()
release_dir = Path(os.environ["VERIFACTBHC_DATASET_DIR"])
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 60)
pd.set_option("display.min_rows", 60)

models = ["Llama-8B", "Llama-70B", "R1-8B", "R1-70B"]
ai_verdicts = load_release_annotations(release_dir, models=models)

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
supported_proposition = attach_reference_payloads(
    df.loc[df.text == "The patient had a history of chronic obstructive pulmonary disease."],
    release_dir,
).squeeze()
print("Proposition Text:")
print(supported_proposition.text)
print("")
print("Reference Context:")
print(supported_proposition.reference)
# %%
not_supported_proposition = attach_reference_payloads(
    df.loc[df.text == "The patient was intubated due to worsening septic shock."],
    release_dir,
).squeeze()
print("Proposition Text:")
print(not_supported_proposition.text)
print("")
print("Reference Context:")
print(not_supported_proposition.reference)
# %%
