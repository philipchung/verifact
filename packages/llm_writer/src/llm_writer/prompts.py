import inspect


def system_prompt_hospital_physician() -> str:
    prompt = """
You are a physician who takes care of patients in a hospital.
"""
    return inspect.cleandoc(prompt)


def prompt_summarize_note(text: str) -> str:
    prompt = f"""
## Task:
Summarize the text in 1-4 sentences making sure to include the key points. \
Return only the summary text. Do not state it is a summary.

## Text:
{text}

## Summary:
"""
    return inspect.cleandoc(prompt)


def prompt_combine_summaries(text: str) -> str:
    prompt = f"""
## Task:
Combine the summaries into a single cohesive summary that is at most \
4 sentences long, making sure to include the key points. \
Return only the combined summary text. Do not state it is a combined summary.

## Summaries:
{text}

## Combined Summary:
"""
    return inspect.cleandoc(prompt)


def prompt_brief_hospital_course(text: str) -> str:
    prompt = f"""
## Task:
Write a "Brief Hospital Course" section for a patient's discharge summary using \
the provided information from the patient's electronic health record from this \
hospital admission. \
Respond only with content for the "Brief Hospital course" section \
without any elaboration or explanation.

## Information from Patient's Electronic Health Record:
{text}

## Brief Hospital Course:
"""
    return inspect.cleandoc(prompt)


def prompt_refine_brief_hospital_course(brief_hospital_course: str, new_text: str) -> str:
    prompt = f""" 
## Task:
You are preparing a patient for discharge from the hospital and have written part \
of the "Brief Hospital Course" section for the patient's discharge summary. \
Refine the "Brief Hospital Course" section by incorporating the new information \
from the patient's electronic health record. \
Respond only with content for the "Brief Hospital course" section \
without any elaboration or explanation.

## Existing Draft of "Brief Hospital Course":
{brief_hospital_course}

## New Information from Patient's Electronic Health Record:
{new_text}

## Refined "Brief Hospital Course":
"""
    return inspect.cleandoc(prompt)


def prompt_compact_brief_hospital_course(brief_hospital_course: str) -> str:
    prompt = f"""
## Task:
You are preparing a patient for discharge from the hospital and have written a \
"Brief Hospital Course" section for the patient's discharge summary. \
However, the text is too long. Please make it more concise while retaining \
the key points.

## Existing Draft of "Brief Hospital Course":
{brief_hospital_course}

## Refined "Brief Hospital Course":
"""
    return inspect.cleandoc(prompt)
