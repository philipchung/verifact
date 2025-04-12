# %%
import os
from pathlib import Path

import pandas as pd
from qdrant_client import models
from rag.components import get_vectorstore
from tqdm.auto import tqdm
from utils import load_environment, load_pandas

load_environment()

mimic3_dir = Path(os.environ["MIMIC3_DIR"])
verifact_bhc_dir = Path(os.environ["VERIFACTBHC_DATASET_DIR"])
patients_df = load_pandas(mimic3_dir / "PATIENTS.csv.gz")
propositions_df = load_pandas(verifact_bhc_dir / "propositions" / "propositions.csv.gz")
admissions_df = load_pandas(verifact_bhc_dir / "reference_ehr" / "admissions.csv.gz")
ehr_noteevents_df = load_pandas(verifact_bhc_dir / "reference_ehr" / "ehr_noteevents.csv.gz")


def count_words(text: str) -> int:
    return len(text.split())


# Word Count per Proposition & EHR Note Events
propositions_df = propositions_df.assign(num_words=propositions_df["text"].apply(count_words))
ehr_noteevents_df = ehr_noteevents_df.assign(num_words=ehr_noteevents_df["TEXT"].apply(count_words))

# %%
# Merge Patients and Admissions
df = pd.merge(
    left=patients_df.loc[:, ["SUBJECT_ID", "GENDER", "DOB"]],
    right=admissions_df.loc[
        :,
        [
            "SUBJECT_ID",
            "HADM_ID",
            "ADMITTIME",
            "DISCHTIME",
            "ETHNICITY",
            "DIAGNOSIS",
            "HOSPITAL_EXPIRE_FLAG",
        ],
    ],
    on="SUBJECT_ID",
)
df = df.assign(DOB=pd.to_datetime(df["DOB"])).astype(
    {
        "DOB": "datetime64[ns]",
        "ADMITTIME": "datetime64[ns]",
        "DISCHTIME": "datetime64[ns]",
        "HOSPITAL_EXPIRE_FLAG": "bool",
    }
)

# Derive Age, Length of Stay, and Harmonize Ethnicity


def compute_age(row: pd.Series) -> int:
    """Determine Patient Age at time of admission"""
    admit_date = row["ADMITTIME"].date()
    dob = row["DOB"].date()
    age = (admit_date - dob).days / 365
    # NOTE: To comply with HIPAA, patients with age > 89 appear
    # in dataset as being over 300 years old; so we just cap these
    # patients age to 90 to enable proper descriptive statistics of
    # the dataset.
    if age > 89:
        return 90
    else:
        return int(age)


def length_of_stay(row: pd.Series) -> int:
    """Compute Length of Stay in Days"""
    return (row["DISCHTIME"] - row["ADMITTIME"]).days


def harmonize_ethnicity(row: pd.Series) -> str:
    """Consolidate Patient Ethnicity into Categories."""
    ethnicity = row["ETHNICITY"]
    if ethnicity.startswith("WHITE"):
        return "WHITE"
    elif ethnicity.startswith("BLACK"):
        return "BLACK/AFRICAN AMERICAN"
    elif ethnicity.startswith("ASIAN"):
        return "ASIAN"
    elif ethnicity.startswith("HISPANIC"):
        return "HISPANIC/LATINO"
    elif ethnicity in ("UNKNOWN/NOT SPECIFIED", "UNABLE TO OBTAIN", "PATIENT DECLINED TO ANSWER"):
        return "UNKNOWN/DECLINED TO ANSWER"
    else:
        return ethnicity


df = df.assign(
    AGE=df.apply(compute_age, axis=1),
    LENGTH_OF_STAY=df.apply(length_of_stay, axis=1),
    ETHNICITY=df.apply(harmonize_ethnicity, axis=1),
)
# %%
# Patient Demographics
# - Number of Patients
# - Number of Ethnicity
# - Age Mean (Std)
# - Admission Length of Stay
# - Number of Hospital Admissions per Subject ID
print("=== Patient Demographics ===")
num_pts = df["SUBJECT_ID"].nunique()
print(f"Number of Patients, N: {num_pts}")

ethnicity_ct = df["ETHNICITY"].value_counts()
print("Ethnicity Counts, N: ")
for k, v in ethnicity_ct.items():
    print(f"- {k}: {v}")

age_stats = df["AGE"].describe()
age_mean = age_stats["mean"]
age_std = age_stats["std"]
print(f"Age, Mean (Std): {age_mean:.2f} ({age_std:.2f})")

los_stats = df["LENGTH_OF_STAY"].describe()
los_mean = los_stats["mean"]
los_std = los_stats["std"]
print(f"Length of Stay, Mean (Std): {los_mean:.2f} ({los_std:.2f})")

admission_ct_per_subject_id_stats = (
    ehr_noteevents_df.groupby("SUBJECT_ID")[["HADM_ID"]].nunique().describe()
)
admission_ct_per_subject_id_mean = admission_ct_per_subject_id_stats.loc["mean", "HADM_ID"]
admission_ct_per_subject_id_std = admission_ct_per_subject_id_stats.loc["std", "HADM_ID"]
print(
    f"Number of Hospital Admissions per Subject ID, Mean (Std): "
    f"{admission_ct_per_subject_id_mean:.2f} ({admission_ct_per_subject_id_std:.2f})"
)

# %%
# LLM-Written Brief Hospital Course
# - Number of Words
# - Number of Sentences
# - Number of Atomic Claims
print("=== LLM-Written Brief Hospital Course Statistics ===")
llm_author_propositions = propositions_df.query("author_type == 'llm'")

llm_bhc_word_stats = llm_author_propositions.brief_hospital_course.apply(
    lambda x: count_words(x)
).describe()
llm_bhc_word_mean = llm_bhc_word_stats["mean"]
llm_bhc_word_std = llm_bhc_word_stats["std"]
print(
    f"Number of Words in Brief Hospital Course, Mean (Std): "
    f"{llm_bhc_word_mean:.2f} ({llm_bhc_word_std:.2f})"
)
# %%

print("= Atomic Claims =")
llm_claim = llm_author_propositions.query("proposition_type == 'claim'")
llm_claim_ct = llm_claim.shape[0]
print(f"Total Number of Atomic Claims: {llm_claim_ct}")

llm_claim_per_subject = llm_claim.subject_id.value_counts()
llm_claim_per_subject_stats = llm_claim_per_subject.describe()
llm_claim_per_subject_mean = llm_claim_per_subject_stats["mean"]
llm_claim_per_subject_std = llm_claim_per_subject_stats["std"]
print(
    f"Number of Atomic Claims per Subject, Mean (Std): "
    f"{llm_claim_per_subject_mean:.2f} ({llm_claim_per_subject_std:.2f})"
)

llm_claim_stats = llm_claim["num_words"].describe()
llm_claim_mean = llm_claim_stats["mean"]
llm_claim_std = llm_claim_stats["std"]
print(f"Number of Words in Atomic Claims, Mean (Std): "
      f"{llm_claim_mean:.2f} ({llm_claim_std:.2f})")

print("= Sentences =")
llm_sentence = llm_author_propositions.query("proposition_type == 'sentence'")
llm_sentence_ct = llm_sentence.shape[0]
print(f"Total Number of Sentences: {llm_sentence_ct}")

llm_sentence_per_subject = llm_sentence.subject_id.value_counts()
llm_sentence_per_subject_stats = llm_sentence_per_subject.describe()
llm_sentence_per_subject_mean = llm_sentence_per_subject_stats["mean"]
llm_sentence_per_subject_std = llm_sentence_per_subject_stats["std"]
print(
    f"Number of Sentences per Subject, Mean (Std): "
    f"{llm_sentence_per_subject_mean:.2f} ({llm_sentence_per_subject_std:.2f})"
)

llm_sentence_stats = llm_sentence["num_words"].describe()
llm_sentence_mean = llm_sentence_stats["mean"]
llm_sentence_std = llm_sentence_stats["std"]
print(f"Number of Words in Sentences, Mean (Std): "
      f"{llm_sentence_mean:.2f} ({llm_sentence_std:.2f})")

# %%
# Human-Written Brief Hospital Course
# - Number of Words
# - Number of Sentences
# - Number of Atomic Claims
print("=== Human-Written Brief Hospital Course Statistics ===")
human_author_propositions = propositions_df.query("author_type == 'human'")

human_bhc_word_stats = human_author_propositions.brief_hospital_course.apply(
    lambda x: count_words(x)
).describe()
human_bhc_word_mean = human_bhc_word_stats["mean"]
human_bhc_word_std = human_bhc_word_stats["std"]
print(
    f"Number of Words in Brief Hospital Course, Mean (Std): "
    f"{human_bhc_word_mean:.2f} ({human_bhc_word_std:.2f})"
)

print("= Atomic Claims =")
human_claim = human_author_propositions.query("proposition_type == 'claim'")
human_claim_ct = human_claim.shape[0]
print(f"Total Number of Atomic Claims: {human_claim_ct}")

human_claim_per_subject = human_claim.subject_id.value_counts()
human_claim_per_subject_stats = human_claim_per_subject.describe()
human_claim_per_subject_mean = human_claim_per_subject_stats["mean"]
human_claim_per_subject_std = human_claim_per_subject_stats["std"]
print(
    f"Number of Atomic Claims per Subject, Mean (Std): "
    f"{human_claim_per_subject_mean:.2f} ({human_claim_per_subject_std:.2f})"
)

human_claim_stats = human_claim["num_words"].describe()
human_claim_mean = human_claim_stats["mean"]
human_claim_std = human_claim_stats["std"]
print(
    f"Number of Words in Atomic Claims, Mean (Std): "
    f"{human_claim_mean:.2f} ({human_claim_std:.2f})"
)

print("= Sentences =")
human_sentence = human_author_propositions.query("proposition_type == 'sentence'")
human_sentence_ct = human_sentence.shape[0]
print(f"Total Number of Sentences: {human_sentence_ct}")

human_sentence_per_subject = human_sentence.subject_id.value_counts()
human_sentence_per_subject_stats = human_sentence_per_subject.describe()
human_sentence_per_subject_mean = human_sentence_per_subject_stats["mean"]
human_sentence_per_subject_std = human_sentence_per_subject_stats["std"]
print(
    f"Number of Sentences per Subject, Mean (Std): "
    f"{human_sentence_per_subject_mean:.2f} ({human_sentence_per_subject_std:.2f})"
)

human_sentence_stats = human_sentence["num_words"].describe()
human_sentence_mean = human_sentence_stats["mean"]
human_sentence_std = human_sentence_stats["std"]
print(
    f"Number of Words in Sentences, Mean (Std): "
    f"{human_sentence_mean:.2f} ({human_sentence_std:.2f})"
)

# %%
# EHR Notes
# - Number of Notes per Subject ID
# - Number of Notes per Admission
# - Average number of words per EHR note
# - Number of Atomic Claims
# - Number of Sentences
# - Number of Atomic Claims per Subject ID
# - Number of Sentences per Subject ID
# - Number of Words per Atomic Claim
# - Number of Words per Sentence

print("=== EHR Notes Statistics ===")
total_ehr_notes = len(ehr_noteevents_df)
print(f"Total Number of EHR Notes: {total_ehr_notes}")
# %%
ehr_notes_per_subject = ehr_noteevents_df.groupby("SUBJECT_ID").size()
ehr_notes_per_subject_stats = ehr_notes_per_subject.describe()
ehr_notes_per_subject_mean = ehr_notes_per_subject_stats["mean"]
ehr_notes_per_subject_std = ehr_notes_per_subject_stats["std"]
print(
    f"Number of EHR Notes per Subject, Mean (Std): "
    f"{ehr_notes_per_subject_mean:.2f} ({ehr_notes_per_subject_std:.2f})"
)

ehr_notes_per_admission = ehr_noteevents_df.groupby("HADM_ID").size()
ehr_notes_per_admission_stats = ehr_notes_per_admission.describe()
ehr_notes_per_admission_mean = ehr_notes_per_admission_stats["mean"]
ehr_notes_per_admission_std = ehr_notes_per_admission_stats["std"]
print(
    f"Number of EHR Notes per Admission, Mean (Std): "
    f"{ehr_notes_per_admission_mean:.2f} ({ehr_notes_per_admission_std:.2f})"
)

ehr_notes_word_count_stats = ehr_noteevents_df["num_words"].describe()
ehr_notes_word_count_mean = ehr_notes_word_count_stats["mean"]
ehr_notes_word_count_std = ehr_notes_word_count_stats["std"]
print(
    f"Number of Words in EHR Notes, Mean (Std): "
    f"{ehr_notes_word_count_mean:.2f} ({ehr_notes_word_count_std:.2f})"
)
# %%
# Query Qdrant Database for Number of Atomic Claims

load_environment()
vectorstore = get_vectorstore()
client = vectorstore.client
collection_name = os.environ["MIMIC3_EHR_COLLECTION_NAME"]
# %%
print(" === Number of Atomic Claims and Sentences for all Subject IDs ===")
result = client.count(
    collection_name=f"{collection_name}",
    count_filter=models.Filter(
        must=[
            models.FieldCondition(key="node_kind", match=models.MatchValue(value="claim")),
        ]
    ),
    exact=True,
)
num_ehr_claims = result.count
print(f"Total Number of EHR Claims: {result.count}")
result = client.count(
    collection_name=f"{collection_name}",
    count_filter=models.Filter(
        must=[
            models.FieldCondition(key="node_kind", match=models.MatchValue(value="sentence")),
        ]
    ),
    exact=True,
)
num_ehr_sentences = result.count
print(f"Total Number of EHR Sentences: {result.count}")

# %%
print(" === Number of Atomic Claims and Sentences per Subject ID ===")
subject_ids = ehr_noteevents_df.SUBJECT_ID.unique()
claims_per_subject_id: dict[int, int] = {}
sentences_per_subject_id: dict[int, int] = {}
for subject_id in tqdm(
    subject_ids, total=len(subject_ids), desc="Counting Claims and Sentences per Subject ID"
):
    result = client.count(
        collection_name=f"{collection_name}",
        count_filter=models.Filter(
            must=[
                models.FieldCondition(key="node_kind", match=models.MatchValue(value="claim")),
                models.FieldCondition(
                    key="SUBJECT_ID", match=models.MatchValue(value=int(subject_id))
                ),
            ]
        ),
        exact=True,
    )
    num_ehr_claims_per_subject_id = result.count
    claims_per_subject_id[subject_id] = num_ehr_claims_per_subject_id
    result = client.count(
        collection_name=f"{collection_name}",
        count_filter=models.Filter(
            must=[
                models.FieldCondition(key="node_kind", match=models.MatchValue(value="sentence")),
                models.FieldCondition(
                    key="SUBJECT_ID", match=models.MatchValue(value=int(subject_id))
                ),
            ]
        ),
        exact=True,
    )
    num_ehr_sentences_per_subject_id = result.count
    sentences_per_subject_id[subject_id] = num_ehr_sentences_per_subject_id

claims_per_subject_id = pd.Series(claims_per_subject_id)
claims_per_subject_id_stats = claims_per_subject_id.describe()
claims_per_subject_id_mean = claims_per_subject_id_stats["mean"]
claims_per_subject_id_std = claims_per_subject_id_stats["std"]
print(
    f"Number of EHR Claims per Subject ID, Mean (Std): "
    f"{claims_per_subject_id_mean:.2f} ({claims_per_subject_id_std:.2f})"
)

sentences_per_subject_id = pd.Series(sentences_per_subject_id)
sentences_per_subject_id_stats = sentences_per_subject_id.describe()
sentences_per_subject_id_mean = sentences_per_subject_id_stats["mean"]
sentences_per_subject_id_std = sentences_per_subject_id_stats["std"]
print(
    f"Number of EHR Sentences per Subject ID, Mean (Std): "
    f"{sentences_per_subject_id_mean:.2f} ({sentences_per_subject_id_std:.2f})"
)

# %%
print(" === Number of Words per Atomic Claim and Sentence ===")
print(" = Number of Words per Atomic Claim =")
claims_num_words: dict[str, int] = {}
# Scroll Through All Claim Points in Vector Database
node_kind = "claim"
batch_size = 1000
expected_num_batches = num_ehr_claims // batch_size + 1
pbar = tqdm(total=num_ehr_claims, desc="Counting Words in EHR Claims")
next_page_offset = "zzz"
while next_page_offset:
    pbar.update(n=batch_size)
    # Scroll Points
    points, next_page_offset = client.scroll(
        collection_name=f"{collection_name}",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="node_kind", match=models.MatchValue(value=node_kind)),
            ]
        ),
        limit=batch_size,
        with_payload=True,
        with_vectors=False,
        offset=None if next_page_offset == "zzz" else next_page_offset,
    )
    # Unpack the payload in points and extract the number of words in each claim
    for point in points:
        _id = point.id
        text = point.payload["text"]
        num_words = len(text.split())
        claims_num_words[_id] = num_words

claims_num_words = pd.Series(claims_num_words)
claims_num_words_stats = claims_num_words.describe()
claims_num_words_mean = claims_num_words_stats["mean"]
claims_num_words_std = claims_num_words_stats["std"]
print(
    f"Number of Words in EHR Claims, Mean (Std): "
    f"{claims_num_words_mean:.2f} "({claims_num_words_std:.2f})"
)

# %%
print(" = Number of Words per Sentence =")
sentences_num_words: dict[str, int] = {}
# Scroll Through All Sentence Points in Vector Database
node_kind = "sentence"
batch_size = 1000
expected_num_batches = num_ehr_sentences // batch_size + 1
pbar = tqdm(total=num_ehr_sentences, desc="Counting Words in EHR Sentences")
next_page_offset = "zzz"
while next_page_offset:
    pbar.update(n=batch_size)
    # Scroll Points
    points, next_page_offset = client.scroll(
        collection_name=f"{collection_name}",
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="node_kind", match=models.MatchValue(value=node_kind)),
            ]
        ),
        limit=batch_size,
        with_payload=True,
        with_vectors=False,
        offset=None if next_page_offset == "zzz" else next_page_offset,
    )
    # Unpack the payload in points and extract the number of words in each claim
    for point in points:
        _id = point.id
        text = point.payload["text"]
        num_words = len(text.split())
        sentences_num_words[_id] = num_words

sentences_num_words = pd.Series(sentences_num_words)
sentences_num_words_stats = sentences_num_words.describe()
sentences_num_words_mean = sentences_num_words_stats["mean"]
sentences_num_words_std = sentences_num_words_stats["std"]
print(
    f"Number of Words in EHR Sentences, Mean (Std): "
    f"{sentences_num_words_mean:.2f} ({sentences_num_words_std:.2f})"
)

# %%
