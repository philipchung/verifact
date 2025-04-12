# ruff: noqa: E501
import inspect


def system_prompt_judge() -> str:
    prompt = """
You are judging whether text written about a patient is supported by information from the patient's electronic health record.
"""
    return inspect.cleandoc(prompt)


def prompt_determine_verdict(json: str) -> str:
    prompt = f"""
## Task:
Using the provided reference from the patient's electronic health record, determine whether the text is supported, not supported, or not addressed by the reference context.

## Detailed Instructions:
1. Compare the text and reference, which is provided in the "text" and "reference" keys of the input JSON object.
2. Provide a verdict in the output JSON object under the key "verdict".
    a. Determine whether the text is addressed or not addressed by the reference context.
    b. If the text is not addressed by the reference context, output the verdict "Not Addressed".
    c. If the text is addressed and fully supported by the reference context, output the verdict "Supported".
    d. If the text is addressed and not fully supported by the reference context, output the verdict "Not Supported". This includes cases where the text is only partially supported by the reference context.
    e. The only valid outputs are "Supported", "Not Supported", or "Not Addressed".
    d. When determining a verdict, consider the information in the reference context to be correct and the source of truth.
3. Provide a reason for your verdict in the output JSON object under the key "reason". Try to make your reason concise and to the point. You must provide a reason. Do not leave it blank.
4. Format your response as a JSON object with the keys "verdict" and "reason". Return only JSON. No explanation is needed.

## Actual Task
Input JSON:
{json}
Output JSON:
"""
    return inspect.cleandoc(prompt)


def prompt_determine_verdict_fewshot(json: str) -> str:
    prompt = f"""
## Task:
Using the provided reference from the patient's electronic health record, determine whether the text is supported, not supported, or not addressed by the reference context.

## Detailed Instructions:
1. Compare the text and reference, which is provided in the "text" and "reference" keys of the input JSON object.
2. Provide a verdict in the output JSON object under the key "verdict".
    a. Determine whether the text is addressed or not addressed by the reference context.
    b. If the text is not addressed by the reference context, output the verdict "Not Addressed".
    c. If the text is addressed and fully supported by the reference context, output the verdict "Supported".
    d. If the text is addressed and not fully supported by the reference context, output the verdict "Not Supported". This includes cases where the text is only partially supported by the reference context.
    e. The only valid outputs are "Supported", "Not Supported", or "Not Addressed".
    d. When determining a verdict, consider the information in the reference context to be correct and the source of truth.
3. Provide a reason for your verdict in the output JSON object under the key "reason". Try to make your reason concise and to the point. You must provide a reason. Do not leave it blank.
4. Format your response as a JSON object with the keys "verdict" and "reason". Return only JSON. No explanation is needed.

## Examples
Input JSON:
{{
    "text": "Jenny's Hemoglobin A1c indicated she has poorly controlled diabetes."
    "reference": "Jenny's Hemoglobin A1c was 6.8% on admission. Danny has a Hemoglobin A1c of 9.2%."
}}
Output JSON:
{{
    "verdict": "Not Supported",
    "reason": "The text states that Jenny has poorly controlled diabetes, but her Hemoglobin A1c was 6.8% on admission which is within normal range."
}}

Input JSON:
{{
    "text": "Mr. Sanchez has severe cardiac disease."
    "reference": "Carlos Sanchez had a LVEF 23% on transthoracic echocardiogram."
}}
Output JSON:
{{
    "verdict": "Supported",
    "reason": "The text states that Mr. Sanchez has severe cardiac disease which is supported by his transthoracic echocardiogram showing a low LVEF of 23%."
}}

Input JSON:
{{
    "text": "The patient is allergic to penicillin."
    "reference": "The patient is allergic to latex and cefazolin."
}}
Output JSON:
{{
    "verdict": "Not Addressed",
    "reason": "The reference context does not mention the patient's allergy to penicillin. The patient is allergic to cefazolin, but this is a cephalosporin antibiotic and not a penicillin and chances of cross-reactivity are extremely low."
}}

## Actual Task
Input JSON:
{json}
Output JSON:
"""
    return inspect.cleandoc(prompt)


def prompt_determine_verdict_fewshot_reasoning(json: str) -> str:
    prompt = f"""
## Task:
Using the provided reference from the patient's electronic health record, determine whether the text is supported, not supported, or not addressed by the reference context.

## Detailed Instructions:
1. Compare the text and reference, which is provided in the "text" and "reference" keys of the input JSON object.
2. You may think about the task before answering.
3. Provide a verdict in the output JSON object under the key "verdict".
    a. Determine whether the text is addressed or not addressed by the reference context.
    b. If the text is not addressed by the reference context, output the verdict "Not Addressed".
    c. If the text is addressed and fully supported by the reference context, output the verdict "Supported".
    d. If the text is addressed and not fully supported by the reference context, output the verdict "Not Supported". This includes cases where the text is only partially supported by the reference context.
    e. The only valid outputs are "Supported", "Not Supported", or "Not Addressed".
    d. When determining a verdict, consider the information in the reference context to be correct and the source of truth.
4. Provide a reason for your verdict in the output JSON object under the key "reason". Try to make your reason concise and to the point. You must provide a reason. Do not leave it blank.
5. Format your response as a JSON object with the keys "verdict" and "reason". Return only JSON.

## Examples
Input JSON:
{{
    "text": "Jenny's Hemoglobin A1c indicated she has poorly controlled diabetes."
    "reference": "Jenny's Hemoglobin A1c was 6.8% on admission. Danny has a Hemoglobin A1c of 9.2%."
}}
Output JSON:
{{
    "verdict": "Not Supported",
    "reason": "The text states that Jenny has poorly controlled diabetes, but her Hemoglobin A1c was 6.8% on admission which is within normal range."
}}

Input JSON:
{{
    "text": "Mr. Sanchez has severe cardiac disease."
    "reference": "Carlos Sanchez had a LVEF 23% on transthoracic echocardiogram."
}}
Output JSON:
{{
    "verdict": "Supported",
    "reason": "The text states that Mr. Sanchez has severe cardiac disease which is supported by his transthoracic echocardiogram showing a low LVEF of 23%."
}}

Input JSON:
{{
    "text": "The patient is allergic to penicillin."
    "reference": "The patient is allergic to latex and cefazolin."
}}
Output JSON:
{{
    "verdict": "Not Addressed",
    "reason": "The reference context does not mention the patient's allergy to penicillin. The patient is allergic to cefazolin, but this is a cephalosporin antibiotic and not a penicillin and chances of cross-reactivity are extremely low."
}}

## Actual Task
Input JSON:
{json}
"""
    return inspect.cleandoc(prompt)


def prompt_determine_verdict_reasoning(json: str) -> str:
    prompt = f"""
## Task:
Using the provided reference from the patient's electronic health record, determine whether the text is supported, not supported, or not addressed by the reference context.

## Detailed Instructions:
1. Compare the text and reference, which is provided in the "text" and "reference" keys of the input JSON object.
2. You may think about the task before answering.
3. Provide a verdict in the output JSON object under the key "verdict".
    a. Determine whether the text is addressed or not addressed by the reference context.
    b. If the text is not addressed by the reference context, output the verdict "Not Addressed".
    c. If the text is addressed and fully supported by the reference context, output the verdict "Supported".
    d. If the text is addressed and not fully supported by the reference context, output the verdict "Not Supported". This includes cases where the text is only partially supported by the reference context.
    e. The only valid outputs are "Supported", "Not Supported", or "Not Addressed".
    d. When determining a verdict, consider the information in the reference context to be correct and the source of truth.
4. Provide a reason for your verdict in the output JSON object under the key "reason". Try to make your reason concise and to the point. You must provide a reason. Do not leave it blank.
5. Format your response as a JSON object with the keys "verdict" and "reason". Return only JSON.

## Input JSON:
{json}
"""
    return inspect.cleandoc(prompt)


def prompt_json_structured_output(text: str) -> str:
    prompt = f"""
## Task Definition:
You are given text with a "verdict" and "reason" which may not be formatted as a JSON object. Please fix the formatting and return a JSON object with the keys "verdict" and "reason". 

## Actual Task
Text:
{text}
Output JSON:
"""
    return inspect.cleandoc(prompt)


def prompt_summarize_reasons(json: str, label: str) -> str:
    prompt = f"""
## Task Definition
Summarize the reasons for which the text was {label} by the reference context.
Output the summary as a JSON object with the key "summary".

## Reasons
{json}

## Summary
"""
    return inspect.cleandoc(prompt)
