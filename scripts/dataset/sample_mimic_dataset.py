# %% [markdown]
## Create Unannotated Dataset by Sampling Patients and their EHR Data from MIMIC-III
# %%
import logging
import os
from functools import partial
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm
from utils import load_environment, parallel_process, save_pandas, save_pickle, save_text

load_environment()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Input Data Paths
mimic3_dir = Path(os.environ["MIMIC3_DIR"])
# Define Output Paths
output_dir = Path(os.environ["DATA_DIR"]) / "dataset"
output_dir.mkdir(exist_ok=True, parents=True)

# %%
## Load MIMIC III Tables
# Patients: https://mimic.mit.edu/docs/iii/tables/patients/
patients = pd.read_csv(mimic3_dir / "PATIENTS.csv.gz")
# Admissions: https://mimic.mit.edu/docs/iii/tables/admissions/
admissions = pd.read_csv(mimic3_dir / "ADMISSIONS.csv.gz")
# Notes: https://mimic.mit.edu/docs/iii/tables/noteevents/
notes = pd.read_csv(mimic3_dir / "NOTEEVENTS.csv.gz")
# NOTE:
# - All notes have a CHARTDATE
# - Some Notes do not have a CHARTTIME (Datetime of Note)
# - Many Notes do not have a store time (Datetime of Note Entry)
# - Combination of Note CATEGORY + DESCRIPTION tells you what kind of note it is
# - Includes floor notes from physicians, nurses, therapists, and also radiology reports
#   and discharge summaries

# Remove notes identified as Errors by physicians]
notes = notes[notes.ISERROR != 1.0]
# Remove leading/trailing whitespace from CATEGORY column
notes.loc[:, "CATEGORY"] = notes.CATEGORY.str.strip()
# Remove leading/trailing whitespace from DESCRIPTION column
notes.loc[:, "DESCRIPTION"] = notes.DESCRIPTION.str.strip()
# %%
## Note Version Deduplication
# We perform a best-attempt deduplication of notes that have been addended/updated over time.
# There is no explicit indicator of which copy of the note is an addendum or which copy is
# the final one. So we make the following assumptions to deduplicate and select the final note:
# 1. A unique note is identified by the combination of SUBJECT_ID, HADM_ID, CHARTDATE, CHARTTIME,
#    CATEGORY, DESCRIPTION, and CGID.
# 2. After grouping all notes by the above columns, we sort by ROW_ID in ascending order.
#    We assume ROW_IDs are monotonic and from our inspection of the data, it appears that
#    that the higher ROW_IDs are the more recent notes.
# 3. We select the note with the most highest ROW_ID as the final note.
# 4. We avoid applying this logic to CATEGORY == 'Discharge summary' notes as these lack
#    a CHARTTIME and STORETIME. Also, only 'Discharge summary' notes have their addendums
#    marked as adddendum, which we will handle separately

# Skip notes with nan in any of the columns we will group by
avoid_version_deduplicate_flag = (
    notes.loc[
        :,
        [
            "SUBJECT_ID",
            "HADM_ID",
            "CHARTDATE",
            "CHARTTIME",
            "STORETIME",
            "CATEGORY",
            "DESCRIPTION",
            "CGID",
        ],
    ]
    .isna()
    .any(axis=1)
)
notes_to_avoid_version_deduplicate = notes.loc[avoid_version_deduplicate_flag]
notes_to_version_deduplicate = notes.loc[~avoid_version_deduplicate_flag]

# Identify Final Note Version & Deduplicate Note Versions
tqdm.pandas(desc="Identifying Final Note Version")
row_ids_to_keep = (
    notes_to_version_deduplicate.groupby(
        ["SUBJECT_ID", "HADM_ID", "CHARTDATE", "CHARTTIME", "CATEGORY", "DESCRIPTION", "CGID"]
    )[["ROW_ID", "TEXT", "ISERROR"]]
    .progress_apply(lambda group: group.sort_values(by="ROW_ID", ascending=True).iloc[-1].ROW_ID)
    .reset_index(drop=True)
)
version_deduplicated_notes = notes_to_version_deduplicate.query("ROW_ID in @row_ids_to_keep")

# Combine Notes with Deduplicated Versions with Notes We Left Out of Deduplication Process
notes = (
    pd.concat([version_deduplicated_notes, notes_to_avoid_version_deduplicate])
    .sort_values(by=["SUBJECT_ID", "CHARTDATE", "CHARTTIME", "STORETIME"], ascending=True)
    .reset_index(drop=True)
)

# %%
## Inclusion Criteria for Notes:
# 1. Hospital Admission Has a Discharge Summary
# 2. Discharge Summary has a "Brief Hospital Course" section noted in the text
# 3. Hospital Admission has at least 2 physician notes (excluding discharge summary)
# 4. Patient has at least 10 notes occuring prior to the discharge summary
#   (these may include other note categories like nursing, radiology, etc. and may also
#   be from previous hospital admissions.)

## Apply Inclusion Criteria to Discharge Summaries
# Get notes that are discharge summaries & have "Brief Hospital Course" in the text
dc_sum_notes1 = notes.query("CATEGORY == 'Discharge summary'")
dc_sum_with_brief_hospital_course_flag = dc_sum_notes1["TEXT"].str.contains("Brief Hospital Course")
dc_sum_notes2 = dc_sum_notes1.loc[dc_sum_with_brief_hospital_course_flag]


# If patients have multiple hospital admissions, use the discharge summary from last
# hospital admission to maximize the number of historical notes available for the patient.
# This also ensures that when we select discharge summaries that there is only one
# discharge summary per patient in the final dataset.
def select_last_admission(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by="CHARTDATE", ascending=True).iloc[-1].HADM_ID


last_hadm_per_patient = dc_sum_notes2.groupby("SUBJECT_ID")[["HADM_ID", "CHARTDATE"]].apply(
    select_last_admission
)
dc_sum_notes3 = dc_sum_notes2.query("HADM_ID in @last_hadm_per_patient")

# Some hospital admissions have multiple discharge summaries filed (usu. duplicate notes).
# We remove these admissions from consideration and only consider admissions where there is
# only a single discharge summary (though it is ok to have addendum).
hadm_ids_with_one_dc_sum_flag = dc_sum_notes3.HADM_ID.value_counts() == 1
hadm_ids_with_one_dc_sum = hadm_ids_with_one_dc_sum_flag.loc[
    hadm_ids_with_one_dc_sum_flag
].index.to_list()
dc_sum_notes4 = dc_sum_notes3.query("HADM_ID in @hadm_ids_with_one_dc_sum")

# %%
## Apply Inclusion Criteria to Notes Other than Discharge Summaries
# Narrow notes to those from admissions with at least 2 physician notes before discharge summary
# and patient has had at least 10 historical notes before discharge summary
num_physician_notes_before_dc_sum_in_admission = 2


def has_min_physician_notes_before_dc_summary(
    hadm_id: int, df: pd.DataFrame, num_notes: int = 10
) -> dict[float, bool]:
    discharge_summaries = df.query("CATEGORY == 'Discharge summary'")
    last_discharge_summary = discharge_summaries.iloc[-1]
    last_discharge_summary_date = last_discharge_summary.CHARTDATE  # noqa: F841
    physician_notes_df = df.query("CATEGORY == 'Physician'")
    physician_notes_before_dc_sum = physician_notes_df.query(
        "CHARTDATE <= @last_discharge_summary_date"
    )
    return {hadm_id: len(physician_notes_before_dc_sum) >= num_notes}


has_2_physician_notes_before_dc_summary = partial(
    has_min_physician_notes_before_dc_summary,
    num_notes=num_physician_notes_before_dc_sum_in_admission,
)

# Apply criteria for 2 physician notes in hospital admission before discharge summary
# NOTE: we use HADM_ID from `dc_sum_notes2` as this dataframe contains all hospital admission
# for patients that we are considering before we select only the final hospital admission
# to get the last discharge summary.
other_notes1 = notes.query("HADM_ID in @dc_sum_notes2.HADM_ID")
results = parallel_process(
    iterable=other_notes1.groupby("HADM_ID"),
    function=has_2_physician_notes_before_dc_summary,
    n_jobs=32,
    use_args=True,
    desc="Finding Hospital Admissions with 2 Physician Notes Before DC Summary",
)
hadm_id_with_2_physician_notes_flag = pd.Series({k: v for d in results for k, v in d.items()})
hadm_ids_with_2_physician_notes = hadm_id_with_2_physician_notes_flag.loc[
    hadm_id_with_2_physician_notes_flag
].index.tolist()
other_notes2 = other_notes1.query("HADM_ID in @hadm_ids_with_2_physician_notes")

# Narrow notes to those from patients with at least 10 notes before discharge summary
# these include all kinds of notes and may be from prior hospital admissions
num_total_notes_before_dc_sum = 10


def has_min_notes_before_dc_summary(subject_id: int, df: pd.DataFrame, num_notes: int = 10) -> dict:
    discharge_summaries = df.query("CATEGORY == 'Discharge summary'")
    last_discharge_summary = discharge_summaries.iloc[-1]
    last_discharge_summary_date = last_discharge_summary.CHARTDATE  # noqa: F841
    notes_before_dc_sum = df.query(
        "CHARTDATE <= @last_discharge_summary_date & CATEGORY != 'Discharge summary'"
    )
    return {subject_id: len(notes_before_dc_sum) >= num_notes}


has_10_notes_before_dc_summary = partial(
    has_min_notes_before_dc_summary, num_notes=num_total_notes_before_dc_sum
)

# Apply criteria for 10 notes for patient before discharge summary
results = parallel_process(
    iterable=other_notes2.groupby("SUBJECT_ID"),
    function=has_10_notes_before_dc_summary,
    n_jobs=32,
    use_args=True,
    desc="Finding Patients with 10 Notes Before DC Summary",
)
subject_id_with_10_notes_flag = pd.Series({k: v for d in results for k, v in d.items()})
subject_ids_with_10_notes = subject_id_with_10_notes_flag.loc[
    subject_id_with_10_notes_flag
].index.tolist()
other_notes3 = other_notes2.query("SUBJECT_ID in @subject_ids_with_10_notes")

# %%
# Retain only Hospital Admission & Patients that meet Criteria from both Discharge Summary
# table and Other Notes Table
dc_sum_hadm_ids = dc_sum_notes4.HADM_ID.unique()
dc_sum_subject_ids = dc_sum_notes4.SUBJECT_ID.unique()
other_notes_hadm_ids = other_notes3.HADM_ID.unique()
other_notes_subject_ids = other_notes3.SUBJECT_ID.unique()

hadm_ids_to_keep = list(set(dc_sum_hadm_ids) & set(other_notes_hadm_ids))
subject_ids_to_keep = list(set(dc_sum_subject_ids) & set(other_notes_subject_ids))

dc_sum_notes_intersect = dc_sum_notes4.query(
    "HADM_ID in @hadm_ids_to_keep & SUBJECT_ID in @subject_ids_to_keep"
)
other_notes_intersect = other_notes3.query("SUBJECT_ID in @subject_ids_to_keep")


# %%
## In Discharge Summary Notes, Extract Hospital Course Text
def extract_brief_hospital_course_text(text: str) -> str:
    start_str = "\nBrief Hospital Course:"
    end_str = "\nMedications on Admission:"
    start_idx = text.find(start_str)
    end_idx = text.find(end_str)
    # If fail to extract hospital course text, return empty string
    if start_idx == -1 or end_idx == -1:
        return ""
    return text[start_idx + len(start_str) : end_idx].strip()


bhc_text = dc_sum_notes_intersect.TEXT.apply(extract_brief_hospital_course_text)
cols = dc_sum_notes_intersect.columns.tolist()
cols.remove("TEXT")
bhc_intersect = pd.concat([dc_sum_notes_intersect[cols], bhc_text], axis=1)
# Remove hospital admissions were we failed to extract hospital course text
bhc_intersect = bhc_intersect.query("TEXT != ''")
dc_sum_notes_intersect = dc_sum_notes_intersect.query("HADM_ID in @bhc_intersect.HADM_ID")
other_notes_intersect = other_notes_intersect.query("SUBJECT_ID in @bhc_intersect.SUBJECT_ID")
# Exclude Discharge Summary Notes from "Other Notes" Table (both original note & addendums)
hadm_to_exclude_dc_sum = dc_sum_notes_intersect.HADM_ID
notes_to_exclude = other_notes_intersect.query(
    "HADM_ID in @hadm_to_exclude_dc_sum & CATEGORY == 'Discharge summary'"
)
other_notes_intersect = other_notes_intersect.query("ROW_ID not in @notes_to_exclude.ROW_ID")

# %%
## Sample from Valid Patients and Create Dataset

# Selected MIMIC-III Subject IDs (randomly selected, fixed here for reproducibility)
dataset_subject_ids = (
    1084,
    4954,
    5954,
    6214,
    8501,
    9030,
    15057,
    16076,
    20804,
    21706,
    22289,
    23979,
    25027,
    26858,
    27036,
    27617,
    27823,
    28016,
    28399,
    29426,
    29894,
    30820,
    31135,
    40233,
    41042,
    43599,
    43923,
    45115,
    45979,
    46564,
    47874,
    47907,
    47940,
    48177,
    50524,
    54797,
    55106,
    55204,
    55750,
    55881,
    58861,
    59250,
    59752,
    59840,
    60130,
    60747,
    61157,
    61377,
    63103,
    64997,
    65399,
    66218,
    67117,
    67771,
    68941,
    69023,
    70462,
    71676,
    71743,
    72844,
    72940,
    74237,
    76990,
    77434,
    77869,
    78015,
    79952,
    80084,
    80454,
    80606,
    80889,
    81230,
    82030,
    82177,
    82301,
    82564,
    83566,
    84454,
    84606,
    85689,
    86429,
    87344,
    87897,
    87982,
    88254,
    89909,
    89957,
    90720,
    91315,
    93147,
    93571,
    93640,
    93788,
    94075,
    94404,
    97212,
    97683,
    97885,
    98089,
    98790,
)

# Select Patients from Discharge Summaries Table
dc_sum_notes_sample = dc_sum_notes_intersect.query("SUBJECT_ID in @dataset_subject_ids")

# Sort Discharge Summaries by Subject_ID and time.
dc_sum_notes_sample.loc[:, "CHARTDATE"] = dc_sum_notes_sample.CHARTDATE.apply(pd.to_datetime)
dc_sum_notes_sample.loc[:, "CHARTTIME"] = dc_sum_notes_sample.CHARTTIME.apply(pd.to_datetime)
dc_sum_notes_sample.loc[:, "STORETIME"] = dc_sum_notes_sample.STORETIME.apply(pd.to_datetime)
dc_sum_notes_sample = dc_sum_notes_sample.sort_values(
    by=["SUBJECT_ID", "CHARTDATE", "CHARTTIME", "STORETIME"], ascending=True
).reset_index(drop=True)
# Subject ID and HADM ID lists
sample_subject_ids = dc_sum_notes_sample.SUBJECT_ID.tolist()
sample_hadm_ids = dc_sum_notes_sample.HADM_ID.tolist()
# Sample Patients from Hospital Course Table
bhc_sample = bhc_intersect.query("HADM_ID in @sample_hadm_ids")
bhc_sample.loc[:, "CHARTDATE"] = bhc_sample.CHARTDATE.apply(pd.to_datetime)
bhc_sample.loc[:, "CHARTTIME"] = bhc_sample.CHARTTIME.apply(pd.to_datetime)
bhc_sample.loc[:, "STORETIME"] = bhc_sample.STORETIME.apply(pd.to_datetime)
bhc_sample = bhc_sample.sort_values(
    by=["SUBJECT_ID", "CHARTDATE", "CHARTTIME", "STORETIME"], ascending=True
).reset_index(drop=True)
# Sample Patients from Other Notes Table
other_notes_sample = other_notes_intersect.query("SUBJECT_ID in @sample_subject_ids")
other_notes_sample.loc[:, "CHARTDATE"] = other_notes_sample.CHARTDATE.apply(pd.to_datetime)
other_notes_sample.loc[:, "CHARTTIME"] = other_notes_sample.CHARTTIME.apply(pd.to_datetime)
other_notes_sample.loc[:, "STORETIME"] = other_notes_sample.STORETIME.apply(pd.to_datetime)
other_notes_sample = other_notes_sample.sort_values(
    by=["SUBJECT_ID", "CHARTDATE", "CHARTTIME", "STORETIME"], ascending=True
).reset_index(drop=True)
# Sample Admissions from Hospital Admissions Table
admissions_sample = admissions.query("HADM_ID in @sample_hadm_ids")
admissions_sample.loc[:, "ADMITTIME"] = admissions_sample.ADMITTIME.apply(pd.to_datetime)
admissions_sample.loc[:, "DISCHTIME"] = admissions_sample.DISCHTIME.apply(pd.to_datetime)
admissions_sample = admissions_sample.sort_values(
    by=["SUBJECT_ID", "ADMITTIME"], ascending=True
).reset_index(drop=True)
# Sample Patients from Patients Table
patients_sample = (
    patients.query("SUBJECT_ID in @sample_subject_ids")
    .sort_values(by="SUBJECT_ID")
    .reset_index(drop=True)
)

# %%
# Check that Discharge Summaries & Hospital course do not have overlapping ROW_ID (Note ID)
# with `Other Notes` which represent all other notes for those patients in the EHR
assert not set(bhc_sample.ROW_ID) & set(other_notes_sample.ROW_ID)
assert not set(dc_sum_notes_sample.ROW_ID) & set(other_notes_sample.ROW_ID)

# %%
# Save Brief Hospital Course, Discharge Summary, EHR Note Reference as .feather and .csv files
save_pandas(df=bhc_sample, filepath=output_dir / "bhc_noteevents.feather")
save_pandas(df=dc_sum_notes_sample, filepath=output_dir / "dc_noteevents.feather")
save_pandas(df=other_notes_sample, filepath=output_dir / "ehr_noteevents.feather")

# Save Sampled Admissions Table
save_pandas(df=admissions_sample, filepath=output_dir / "admissions.feather")

# Save Sampled Patients Table
save_pandas(df=patients_sample, filepath=output_dir / "patients.feather")

# Export Discharge Summary Notes to Text Files (for easy readability)
text_file_dir = output_dir / "dc_note_text"
for row in dc_sum_notes_sample.itertuples():
    subject_id = row.SUBJECT_ID
    hadm_id = int(row.HADM_ID) if not pd.isnull(row.HADM_ID) else 0
    row_id = row.ROW_ID
    text = row.TEXT
    save_text(text=text, filepath=text_file_dir / f"{subject_id}_{hadm_id}_{row_id}.txt")

# Export Hospital Course to Text Files
text_file_dir = output_dir / "bhc_note_text"
for row in bhc_sample.itertuples():
    subject_id = row.SUBJECT_ID
    hadm_id = int(row.HADM_ID) if not pd.isnull(row.HADM_ID) else 0
    row_id = row.ROW_ID
    text = row.TEXT
    save_text(text=text, filepath=text_file_dir / f"{subject_id}_{hadm_id}_{row_id}.txt")

# Save Subject IDs List
save_pickle(obj=sample_subject_ids, filepath=output_dir / "subject_ids.pkl")
subject_ids_csv_str = ", ".join(str(x) for x in sorted(sample_subject_ids))
save_text(text=subject_ids_csv_str, filepath=output_dir / "subject_ids.txt")

# %%
