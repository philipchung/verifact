"""Model Components used in ingestion and query scripts and pipelines."""

import logging
import os

from llama_index.core.schema import MetadataMode

from rag.embedding import M3Embedding
from rag.llms.openai_like import OpenAILike
from rag.postprocessor import M3Reranker

logger = logging.getLogger()

DEFAULT_SYSTEM_PROMPT = (
    "You are a physician taking care of patients with expert knowledge of medical conditions "
    "and treatment options."
)


def get_llm(
    model_name: str | None = None,
    api_base: str | None = None,
    context_window: int | None = None,
    max_completion_tokens: int | None = None,
    is_chat_model: bool = True,
    collapse_system_prompt: bool = False,
    reuse_client: bool = False,
    max_retries: int = 10,
    timeout: float | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    top_p: float = 1.0,
    **kwargs,
) -> OpenAILike:
    """Get LLM object.
    The LLM service is hosted on vLLM which determines the specific model and api_base.
    The values specified here depend on the hosted model and vLLM configuration.

    Note that if vLLM is hosting on local machine, control for which GPUs/accelerators
    are used by vLLM is done through vLLM deployment configuration and not here. This
    method returns a OpenAI-like client that can perform API calls.

    Example Usage:
    ```
    from llama_index.llms import ChatMessage

    llm = get_llm()
    prompt = "You are a colorful pirate. Tell me a joke about apples."
    response = llm.chat(messages=ChatMessage(role="user", content=prompt))
    ```

    Args:
        model_name (str): Huggingface model name used in the vLLM service.
        api_base (str): URL for vLLM service hosting the LLM.
        context_window (int | None, optional): The maximum context window for the LLM
            deployed in the vLLM service. The context window includes both input
            and output tokens. Note that the actual context window should be chosen
            based on model documentation and is set in vLLM deployment configuration
            by the `--max-model-len` argument.  The value provided in this argument
            should be this value or less.
        max_completion_tokens (int | None, optional): The maximum number of tokens
            the LLM can generate before truncation. This value and the input prompt
            tokens should not exceed the `context_window`. Defaults to None.
        is_chat_model (bool, optional): Whether the model is a chat vs. completion model.
        collapse_system_prompt (bool, optional): Whether to concatenate the system
            message to the first user message. Some models like Mistral models do not
            support system messages, so this can reformat input messages to
            concatenate system message to the first user input message if a
            system message is provided. Defaults to False.
        reuse_client (bool, optional): Whether to reuse the client. When making large
            volumes of async calls, setting to False has better stability.
            Defaults to False.
        max_retries (int, optional): Maximum number of retries for API calls. Defaults to 10.
        timeout (float, optional): Timeout for API calls. Defaults to None.
        system_message (str, optional): System message to prepend to user message.
            NOTE: Mistral models do not support system messages.  Setting the argument
            `collapse_system_prompt` to True will result in the system message being prepended
            to the first user message. Otherwise, an error may occur.
        temperature (float, optional): The temperature for sampling from the LLM.
        top_p (float, optional): Nucleus sampling (top-p) value for sampling from the LLM.
        kwargs (dict): Additional keyword arguments for the LLM API call.

    Returns:
        llama_index.llms.OpenAILike: A llama-index wrapper around the LLM providing an
            OpenAI-like ChatCompletions API.
    """
    model_name = model_name or os.environ["LLM_MODEL_NAME"]
    api_base = api_base or os.environ["LLM_URL_BASE"]
    context_window = context_window or os.environ["LLM_MAX_MODEL_LEN"]
    additional_kwargs = {"top_p": top_p}
    # Model-specific Customization
    LLAMA_3_MODELS = (
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "meta-llama/Meta-Llama-3-70B",
        "meta-llama/Meta-Llama-Guard-2-8B",
        "casperhansen/llama-3-70b-instruct-awq",
        "casperhansen/llama-3-8b-instruct-awq",
    )
    MISTRAL_MODELS = (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.1",
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mixtral-8x22B-Instruct-v0.1",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mistral-7B-v0.2",
        "mistralai/Mistral-7B-v0.3",
        "mistralai/Mixtral-8x7B-v0.1",
        "mistralai/Mixtral-8x22B-v0.1",
    )
    DEEPSEEK_R1_MODELS = (
        "deepseek-ai/DeepSeek-R1",
        "deepseek-ai/DeepSeek-R1-Zero",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "casperhansen/deepseek-r1-distill-llama-70b-awq",
        "casperhansen/deepseek-r1-distill-qwen-32b-awq",
        "casperhansen/deepseek-r1-distill-qwen-14b-awq",
        "casperhansen/deepseek-r1-distill-llama-8b-awq",
        "casperhansen/deepseek-r1-distill-qwen-7b-awq",
        "casperhansen/deepseek-r1-distill-qwen-1.5b-awq",
    )
    match model_name:
        case x if x in LLAMA_3_MODELS:
            # Llama 3 Prompt Template Issue (https://github.com/vllm-project/vllm/issues/4180)
            if "extra_body" not in kwargs:
                kwargs["extra_body"] = {}
            kwargs["extra_body"] |= {"stop_token_ids": [128001, 128009]}
        case x if x in MISTRAL_MODELS:
            collapse_system_prompt = True
        case x if x in DEEPSEEK_R1_MODELS:
            collapse_system_prompt = True
        case _:
            pass
    additional_kwargs |= kwargs

    logger.debug(f"LLM Model: {model_name} | API Base: {api_base}")
    return OpenAILike(
        model=model_name,
        is_chat_model=is_chat_model,
        collapse_system_prompt=collapse_system_prompt,
        reuse_client=reuse_client,
        max_retries=max_retries,
        timeout=timeout,
        api_key="EMPTY",
        api_base=api_base,
        context_window=context_window,
        max_tokens=max_completion_tokens,
        system_prompt=system_prompt,
        temperature=temperature,
        additional_kwargs=additional_kwargs,
    )


def get_aux_llm(
    model_name: str | None = None,
    api_base: str | None = None,
    context_window: int | None = None,
    max_completion_tokens: int | None = None,
    is_chat_model: bool = True,
    collapse_system_prompt: bool = False,
    reuse_client: bool = False,
    max_retries: int = 10,
    timeout: float | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.7,
    top_p: float = 1.0,
    **kwargs,
) -> OpenAILike:
    """Get Auxillary LLM Model. This is mainly used when we also use a reasoning model.
    As of vLLM v0.7.2, reasoning models are not compatible with Structured Output generation.
    So after the main LLM (reasoning model) generates a reasoning chain and output,
    the output is passed to the auxillary LLM which is set to generate structured output
    from the original reasoning LLM's output.
    """
    return get_llm(
        model_name=model_name or os.environ["AUX_LLM_MODEL_NAME"],
        api_base=api_base or os.environ["AUX_LLM_URL_BASE"],
        context_window=context_window or os.environ["AUX_LLM_MAX_MODEL_LEN"],
        max_completion_tokens=max_completion_tokens,
        is_chat_model=is_chat_model,
        collapse_system_prompt=collapse_system_prompt,
        reuse_client=reuse_client,
        max_retries=max_retries,
        timeout=timeout,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        **kwargs,
    )


def get_embed_model(
    model_name: str | None = None,
    api_base: str | None = None,
    embed_batch_size: int = 50,
    dense_name: str = "dense",
    sparse_name: str = "sparse",
    colbert_name: str = "colbert",
    default_vector_name: str = "dense",
    metadata_mode: MetadataMode | str = MetadataMode.NONE,
    num_workers: int = 24,
    timeout: float | None = None,
) -> M3Embedding:
    """Get Embedding Model.

    This returns a class that makes API calls to the BGE-M3 embedding service,
    which needs to be running and accessible at the specified `api_base`.

    The backing model is BGE-M3, which can generate dense, sparse, and colbert embeddings
    with a single model call. This multiple embedding generation
    paradigm is not well supported by llama-index at this time and requires
    customization of llama-index nodes and llama-index components for compatibility.

    Example Usage:
    ```
    embed_model = get_embed_model()
    all_embeddings = embed_model.get_text_embedding()  # List of dict of embeddings
    m3_dense = [x["dense"] for x in all_embeddings]
    m3_sparse = [x["sparse"] for x in all_embeddings]
    m3_colbert = [x["colbert"] for x in all_embeddings]
    ```

    Args:
        model_name (str, optional): Huggingface embedding model.
        api_base (str, optional): Base URL for text embedding service
        embed_batch_size (int, optional): Batch size for embedding generation. This is the
            batch size constructed in a local queue prior to dispatching request to
            embedding service via API call. Defaults to 50.
        dense_name (str, optional): Name of dense vector. Defaults to "dense".
        sparse_name (str, optional): Name of sparse vector. Defaults to "sparse".
        colbert_name (str, optional): Name of colbert vector. Defaults to "colbert".
        default_vector_name (str, optional): Default vector name. Defaults to "default".
        metadata_mode (MetadataMode | str, optional): Metadata mode for node formatting
            prior to computing embedding. Defaults to MetadataMode.NONE.
        num_workers (int, optional): Number of parallel workers for embedding generation.
            This applies a limit to the amount of concurrent nodes that can be embedded.
            Defaults to 24.
        timeout (float, optional): Timeout for API calls. Defaults to None.

    Returns:
        M3Embedding: Llama-Index wrapper around embedding service hosting the
        BGE-M3 FlagEmbedding model.
    """
    model_name = model_name or os.environ["EMBED_MODEL_NAME"]
    api_base = api_base or os.environ["EMBED_URL_BASE"]
    logger.debug(
        f"Embed Model: {model_name} | API Base: {api_base} | Dense Name: {dense_name} | "
        f"Sparse Name: {sparse_name} | Colbert Name: {colbert_name} | "
        f"Default Vector Name: {default_vector_name}"
    )
    return M3Embedding(
        model_name=model_name,
        api_base=api_base,
        embed_batch_size=embed_batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert=False,
        dense_name=dense_name,
        sparse_name=sparse_name,
        colbert_name=colbert_name,
        default_vector_name=default_vector_name,
        metadata_mode=metadata_mode,
        num_workers=num_workers,
        timeout=timeout,
    )


def get_rerank_model(
    model_name: str | None = None,
    api_base: str | None = None,
    top_n: int | None = None,
) -> M3Reranker:
    """Get Rerank Model.

    Args:
        model_name (str, optional): Huggingface embedding model.
        api_base (str, optional): Base URL for text embedding service.
        top_n (int, optional): Number of nodes to return sorted by score. If `None`,
            all reranked nodes are returned. Defaults to None.

    Returns:
        M3Reranker: Llama-Index wrapper around reranker service hosting the
        BAAI/bge-reranker-v2-m3 cross-encoder model.
    """
    model_name = model_name or os.environ["RERANK_MODEL_NAME"]
    api_base = api_base or os.environ["RERANK_URL_BASE"]
    logger.debug(f"Rerank Model: {model_name} | API Base: {api_base}")
    return M3Reranker(
        model_name=model_name,
        api_base=api_base,
        top_n=top_n,
    )
