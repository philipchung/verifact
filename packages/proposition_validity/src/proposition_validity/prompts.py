# ruff: noqa: E501
import inspect


def system_prompt_judge() -> None:
    prompt = """
    You are a logician tasked with judging whether a given text contains a valid claim.
    """
    return inspect.cleandoc(prompt)


# Same prompt used to check for claims prior to atomic claim extraction in ingestion pipeline
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
2. Format your response as a JSON object with "contains_atomic_claim" key with the \
    boolean value. Return only JSON. No explanation is needed.

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


# Check if the text contains a declarative proposition
def prompt_contains_proposition(text: str) -> None:
    prompt = f""" 
## Task Definition
Determine whether the text contains a valid proposition based on the rules of \
first-order formal logic.

## Proposition Definition
A proposition is a phrase or sentence that asserts something that can be either \
true or false. Key chracteristics of a proposition include:
1. Truth Value: Every proposition has a truth value, meaning it is either true or false, \
    though we may not know which.
2. Declarative Nature: Propositions are expressed through declarative sentences, \
    not questions, commands, or exclamations.
3. Independence from Expression: The same proposition can be expressed in different \
    languages or different ways without changing its truth value. For example, \
    “The sky is blue” and “El cielo es azul” express the same proposition.
4. Distinction from Sentences: Different sentences can express the same proposition, \
    and the same sentence can express different propositions depending on the context.

## Detailed Instructions
1. Read the text from the "text" key in the input JSON object. Determine if the text \
    contains any propositions as defined above. If the text contains at least one \
    proposition, output "contains_proposition" as True. If the text does not contain \
    any atomic claims, output "contains_proposition" as False.
2. Format your response as a JSON object with "contains_proposition" key with the \
    boolean value. Return only JSON. No explanation is needed.

## Examples
Input JSON: 
{{
    "text": "Einstein won the noble prize in 1968 for his discovery \
        of the photoelectric effect."
}}
Output JSON: 
{{
    "contains_proposition": true
}}

Input JSON:
{{
    "text":  "I ate a"
}}
Output JSON:
{{
    "contains_proposition": false
}}

Input JSON:
{{
    "text":  "Give me the camera."
}}
Output JSON:
{{
    "contains_proposition": false
}}

Input JSON:
{{
    "text":  "Did you want steak or fish?"
}}
Output JSON:
{{
    "contains_proposition": false
}}

Input JSON:
{{
    "text":  "Einstein is my favorite hero!"
}}
Output JSON:
{{
    "contains_proposition": false
}}

Input JSON:
{{
    "text":  "We ordered fish for dinner."
}}
Output JSON:
{{
    "contains_proposition": true
}}

## Actual Task
Input JSON:
{{
    "text": "{text}"
}}
Output JSON:
"""
    return inspect.cleandoc(prompt)


# Check for non-valid proposition types in First-Order Formal Logic
def prompt_contains_imperative_statement(text: str) -> None:
    prompt = f"""
## Task Definition
Determine whether the text contains an imperative statement (instruction or command).

## Imperative Statement Definition
Imperative statements give orders or instructions, asking someone to perform an action. \
They do not assert a fact about the world. Imperative statements do not have a truth value; \
they are neither true nor false. They are simply directives for action.

## Detailed Instructions
1. Read the text from the "text" key in the input JSON object. Determine if the text \
    contains any imperative statements as defined above. \
    If the text contains at least one imperative statement, \
    output "contains_imperative_statement" as True. \
    If the text does not contain any imperative statements, \
    output "contains_imperative_statement" as False.
2. Format your response as a JSON object with "contains_imperative_statement" key \
    with the boolean value. Return only JSON. No explanation is needed.

## Examples
Input JSON: 
{{
    "text": "Einstein won the noble prize in 1968 for his discovery \
        of the photoelectric effect."
}}
Output JSON: 
{{
    "contains_imperative_statement": false
}}

Input JSON:
{{
    "text":  "Close the door!"
}}
Output JSON:
{{
    "contains_imperative_statement": true
}}

Input JSON:
{{
    "text":  "Read this book."
}}
Output JSON:
{{
    "contains_imperative_statement": true
}}

## Actual Task
Input JSON:
{{
    "text": "{text}"
}}
Output JSON:
"""
    return inspect.cleandoc(prompt)


def prompt_contains_interrogative_statement(text: str) -> None:
    prompt = f"""
## Task Definition
Determine whether the text contains an interrogative statement (question).

## Interrogative Statement Definition
Questions ask for information or clarification, and are not making any assertions \
about the world. Questions cannot be evaluated as true or false; they are requests \
for information rather than claims about the state of affairs.

## Detailed Instructions
1. Read the text from the "text" key in the input JSON object. Determine if the text \
    contains any interrogative statements as defined above. \
    If the text contains at least one interrogative statement, \
    output "contains_interrogative_statement" as True. \
    If the text does not contain any interrogative statements, \
    output "contains_interrogative_statement" as False.
2. Format your response as a JSON object with "contains_interrogative_statement" key \
    with the boolean value. Return only JSON. No explanation is needed.

## Examples
Input JSON: 
{{
    "text": "Einstein won the noble prize in 1968 for his discovery \
        of the photoelectric effect."
}}
Output JSON: 
{{
    "contains_interrogative_statement": false
}}

Input JSON:
{{
    "text":  "Is the door open?"
}}
Output JSON:
{{
    "contains_interrogative_statement": true
}}

Input JSON:
{{
    "text":  "Who is the president?"
}}
Output JSON:
{{
    "contains_interrogative_statement": true
}}

## Actual Task
Input JSON:
{{
    "text": "{text}"
}}
Output JSON:
"""
    return inspect.cleandoc(prompt)


def prompt_contains_incomplete_statement(text: str) -> None:
    prompt = f"""
## Task Definition
Determine whether the text contains an incomplete or fragmentary statement.

## Incomplete Statement Definition
Incomplete or fragmentary sentences do not make a full assertion and \
therefore cannot be evaluated as true or false. These sentences lack \
a complete thought or claim, so they cannot be judged true or false without additional context.

## Detailed Instructions
1. Read the text from the "text" key in the input JSON object. Determine if the text \
    contains any incomplete statements as defined above. \
    If the text contains at least one incomplete statement, \
    output "contains_incomplete_statement" as True. \
    If the text does not contain any incomplete statements, \
    output "contains_incomplete_statement" as False.
2. Format your response as a JSON object with "contains_incomplete_statement" key \
    with the boolean value. Return only JSON. No explanation is needed.

## Examples
Input JSON: 
{{
    "text": "Einstein won the noble prize in 1968 for his discovery \
        of the photoelectric effect."
}}
Output JSON: 
{{
    "contains_incomplete_statement": false
}}

Input JSON:
{{
    "text":  "Because he was late."
}}
Output JSON:
{{
    "contains_incomplete_statement": true
}}

Input JSON:
{{
    "text":  "When the sun sets."
}}
Output JSON:
{{
    "contains_incomplete_statement": true
}}

Input JSON:
{{
    "text":  "I ate a"
}}
Output JSON:
{{
    "contains_incomplete_statement": true
}}

## Actual Task
Input JSON:
{{
    "text": "{text}"
}}
Output JSON:
"""
    return inspect.cleandoc(prompt)


def prompt_contains_vague_statement(text: str) -> None:
    prompt = f"""
## Task Definition
Determine whether the text contains an vague statement.

## Vague Statement Definition
Vague statements are ones that cannot be formalized as a claim or proposition \
in formal logic due to ambiguity, lack of specificity, or unclear subjects or \
entities. Vague statements fail to describe the full context of the situation \
and require additional information or clarification to be properly formalized \
as a claim in formal logic.

## Detailed Instructions
1. Read the text from the "text" key in the input JSON object. Determine if the text \
    contains any vague statements as defined above. \
    If the text contains at least one vague statement, \
    output "contains_vague_statement" as True. \
    If the text does not contain any vague statements, \
    output "contains_vague_statement" as False.
2. Format your response as a JSON object with "contains_vague_statement" key \
    with the boolean value. Return only JSON. No explanation is needed.

## Examples
Input JSON: 
{{
    "text": "Einstein won the noble prize in 1968 for his discovery \
        of the photoelectric effect."
}}
Output JSON: 
{{
    "contains_vague_statement": false
}}

Input JSON:
{{
    "text":  "He was late to this."
}}
Output JSON:
{{
    "contains_vague_statement": true
}}

Input JSON:
{{
    "text":  "They are scheduled to visit Dr."
}}
Output JSON:
{{
    "contains_vague_statement": true
}}

Input JSON:
{{
    "text":  "Found to be supratherapeutic."
}}
Output JSON:
{{
    "contains_vague_statement": true
}}

## Actual Task
Input JSON:
{{
    "text": "{text}"
}}
Output JSON:
"""
    return inspect.cleandoc(prompt)
