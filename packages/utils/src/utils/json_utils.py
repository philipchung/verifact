"""Adapted from DeepEval library."""

import json
from typing import Any


def trimAndLoadJson(input_string: str) -> Any:
    """Extract only JSON portion of string and load it as a python object."""
    start = input_string.find("{")
    end = input_string.rfind("}") + 1
    jsonStr = input_string[start:end] if start != -1 and end != 0 else ""

    try:
        return json.loads(jsonStr)
    except json.JSONDecodeError as err:
        error_str = (
            "Evaluation LLM outputted an invalid JSON. Please use a better evaluation model."
        )
        raise ValueError(error_str) from err
    except Exception as ex:
        raise Exception(f"An unexpected error occurred: {str(ex)}") from ex
