import inspect


def system_prompt_extract_atomic_claims() -> None:
    prompt = """
You are a physician tasked with extracting atomic claims from different text \
reference sources to represent the information in the text as a list of atomic claims.
"""
    return inspect.cleandoc(prompt)


def prompt_contains_atomic_claims(text: str) -> None:
    prompt = f"""
## Task Definition
Determine whether the text contains any atomic claims.

## Atomic Claim Definition
An atomic claim is a phrase or sentence that makes a single assertion. \
The assertion may be factual or may be a hypothesis posed by the text. \
Atomic claims are indivisible and cannot be decomposed into more fundamental \
claims. More complex facts and statements can be composed from atomic facts.\
Atomic claims should have a subject, object, and predicate. \
The predicate relates the subject to the object.

## Detailed Instructions
1. Read the text from the "text" key in the input JSON object. Determine if the text \
    contains any atomic claims as defined above. If the text contains at least one \
    atomic claim, output "contains_atomic_claim" as True. If the text does not contain \
    any atomic claims, output "contains_atomic_claim" as False.
2. Format your response as a JSON object with "claims" key with the boolean value. \
    Return only JSON. No explanation is needed.

## Examples
Input JSON: 
{{
    "text": "Einstein won the noble prize in 1968 for his discovery \
        of the photoelectric effect."
}}
Output JSON: 
{{
    "contains_atomic_claim": true
}}

Input JSON:
{{
    "text":  "I ate a"
}}
Output JSON:
{{
    "contains_atomic_claim": false
}}

## Actual Task
Input JSON:
{{
    "text": "{text}"
}}
Output JSON:
"""
    return inspect.cleandoc(prompt)


def prompt_extract_atomic_claims(text: str) -> None:
    prompt = f"""
## Task Definition
Generate a list of unique atomic claims that can be inferred from the provided text.

## Atomic Claim Definition
An atomic claim is a phrase or sentence that makes a single assertion. \
The assertion may be factual or may be a hypothesis posed by the text. \
Atomic claims are indivisible and cannot be decomposed into more fundamental \
claims. More complex facts and statements can be composed from atomic facts.\
Atomic claims should have a subject, object, and predicate. The predicate relates \
the subject to the object.

## Detailed Instructions
1. Extract a list of claims from the text in "text" key from the JSON object. \
    Each claim should have a subject, object, and predicate and conform to the \
    Atomic Claim Definition provided above. Claims should be unambiguous \
    if they are read in isolation, which means you should avoid pronouns and \
    ambiguous references.
2. The list of claims should be comprehensive and cover all information in the text. 
    Claims you extract should include the full context it was presented in, \
    NOT cherry picked facts. You should NOT include any prior knowledge, \
    and take the text at face value when extracting claims.
3. Format all the claims as a JSON object with "claims" key with values as a \
    list of string claims. Return only JSON. No explanation is needed.

## Examples
Input JSON: 
{{
    "text": "Einstein won the noble prize in 1968 for his discovery \
        of the photoelectric effect."
}}
Output JSON: 
{{
    "claims": [
        "Einstein won the noble prize for his discovery of the photoelectric effect.",
        "Einstein won the noble prize in 1968."
    ]  
}}

## Actual Task
Input JSON:
{{
    "text": "{text}"
}}
Output JSON:
"""
    return inspect.cleandoc(prompt)
