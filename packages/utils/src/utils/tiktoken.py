import tiktoken

# Derived from: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
# Encodings: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

# Sets of Base Models for Each Tokenizer Encoding Family
O200K_BASE_MODELS = {"gpt-4o-2024-05-13"}
CL100K_BASE_MODELS = {
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4-0314",
    "gpt-4-32k-0314",
    "gpt-4-0613",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
    "gpt-4-1106-vision-preview",
    "gpt-4-0125-preview",
    "gpt-4-turbo-2024-04-09",
}
# Pointers of family name to specific models
GPT_35_TURBO_MODEL = "gpt-3.5-turbo-0125"
GPT_4_MODEL = "gpt-4-0613"
GPT_4_VISION_MODEL = "gpt-4-1106-vision-preview"
GPT_4_TURBO_MODEL = "gpt-4-turbo-2024-04-09"
GPT_4O_MODEL = "gpt-4o-2024-05-13"
# Tokenizer Encoding Names
GPT_4O_ENCODING = "o200k_base"
GPT_4_TURBO_ENCODING = "cl100k_base"
GPT_4_ENCODING = "cl100k_base"
GPT_35_TURBO_ENCODING = "cl100k_base"


def num_tokens_from_string(text: str, encoding_name: str = GPT_4O_ENCODING) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text=text))
    return num_tokens


def num_tokens_from_messages(
    messages: list[dict[str, str]], model: str = "gpt-3.5-turbo-0613"
) -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in O200K_BASE_MODELS or model in CL100K_BASE_MODELS:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. "
            f"Returning num tokens assuming {GPT_35_TURBO_MODEL}."
        )
        return num_tokens_from_messages(messages, model=GPT_35_TURBO_MODEL)
    elif "gpt-4o" in model:
        print(
            "Warning: gpt-4o may update over time. "
            f"Returning num tokens assuming {GPT_4O_MODEL}."
        )
        return num_tokens_from_messages(messages, model=GPT_4O_MODEL)
    elif "gpt-4-turbo" in model:
        print(
            "Warning: gpt-4-turbo may update over time. "
            f"Returning num tokens assuming {GPT_4_TURBO_MODEL}."
        )
        return num_tokens_from_messages(messages, model=GPT_4_TURBO_MODEL)
    elif "gpt-4-vision-preview" in model:
        print(
            "Warning: gpt-4-vision-preview may update over time. "
            f"Returning num tokens assuming {GPT_4_VISION_MODEL}."
        )
        return num_tokens_from_messages(messages, model=GPT_4_VISION_MODEL)
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. " f"Returning num tokens assuming {GPT_4_MODEL}."
        )
        return num_tokens_from_messages(messages, model=GPT_4_MODEL)
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. "
            "See https://github.com/openai/openai-python/blob/main/chatml.md "
            "for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def compare_encodings(example_string: str) -> None:
    """Prints a comparison of three string encodings."""
    # print the example string
    print(f'\nExample string: "{example_string}"')
    # for each encoding, print the # of tokens, the token integers, and the token bytes
    for encoding_name in ["r50k_base", "p50k_base", "cl100k_base", "o200k_base"]:
        encoding = tiktoken.get_encoding(encoding_name)
        token_integers = encoding.encode(example_string)
        num_tokens = len(token_integers)
        token_bytes = [encoding.decode_single_token_bytes(token) for token in token_integers]
        print()
        print(f"{encoding_name}: {num_tokens} tokens")
        print(f"token integers: {token_integers}")
        print(f"token bytes: {token_bytes}")
