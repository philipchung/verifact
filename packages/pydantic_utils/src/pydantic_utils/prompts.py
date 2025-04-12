import inspect


def default_system_prompt() -> str:
    prompt = "You are a helpful assistant."
    return inspect.cleandoc(prompt)


def prompt_revise_json_format(json: str) -> str:
    prompt = f"""
## Task Definition
Correct the JSON format of the given JSON object.

## Detailed Instructions
1. The Input JSON contains errors in formatting that prevents it from being parsed correctly.
2. Correct the JSON string and return the corrected JSON string as the Output JSON.
Return only a JSON object. No other text or explanation is needed.
3. If the JSON string is already correctly formatted, return the original JSON string.

## Actual Task
Input JSON:
{json}
Output JSON:
"""
    return inspect.cleandoc(prompt)
