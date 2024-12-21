"""Modified implementation of LlamaIndex's OpenAI-Like Wrapper class.

Original: llama_index.llms.openai_like.base.OpenAILike
Also References: llama_index.llms.openai.base.OpenAI

Changes:
1. Add option `collapse_system_prompt`. Some models like Mistral models
do not take system messages. Setting this to true will concatenate
the system message to the first user message.
2. Customize `_chat`, `_stream_chat`, `_achat`, `_astream_chat` methods to
include a context manager for the client and async client prior to invoking
a chat completions creation.
"""

from collections.abc import Sequence
from typing import Any, cast

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    MessageRole,
)
from llama_index.core.bridge.pydantic import Field
from llama_index.llms.openai.utils import (
    create_retry_decorator,
    from_openai_message,
    from_openai_token_logprobs,
    to_openai_message_dicts,
)
from llama_index.llms.openai_like import OpenAILike as _OpenAILike
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
)

llm_retry_decorator = create_retry_decorator(
    max_retries=6,
    random_exponential=True,
    stop_after_delay_seconds=60,
    min_seconds=1,
    max_seconds=20,
)


def _reformat_messages(messages: Sequence[ChatMessage]) -> Sequence[ChatMessage]:
    """Reformat messages to concatenate system message to the first user input message.
    Assumes there is only a single system message."""
    user_contents = [m.content for m in messages if m.role == "user"]
    system_contents = [m.content for m in messages if m.role == "system"]

    # Raise error if multiple system messages are provided
    assert len(system_contents) <= 1, "Multiple system messages provided."

    # Append system message to the first user message
    if system_contents:
        user_contents[0] = f"{system_contents[0]}\n\n{user_contents[0]}"

    # Return user messages
    return [ChatMessage(role="user", content=user_content) for user_content in user_contents]


class OpenAILike(_OpenAILike):
    """Same as LlamaIndex OpenAILike, but with a "mistral mode" that reformats
    input messages by concatenate system message to the first user input message
    if a system message is provided. Unlike OpenAI models, Mistral models do not support
    system messages.

    Original: llama_index.llms.openai_like.base.OpenAILike

    OpenAI chat format:
    messages = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is the meaning of life?"),
    ]

    Mistral chat format:
    messages = [
        ChatMessage(role="user", content=
            "You are a helpful assistant.\n\nWhat is the meaning of life?"
        )
    ]
    """

    collapse_system_prompt: bool = Field(
        default=False,
        description=(
            "If True, reformats input message to concatenate system message "
            "to the first user input message."
        ),
    )
    max_retries: int = Field(
        default=5,
        description="The maximum number of API retries.",
        gte=0,
    )
    timeout: float | None = Field(
        default=600.0,
        description="The timeout, in seconds, for API requests.",
        gte=0,
    )
    reuse_client: bool = Field(
        default=False,
        description=(
            "Reuse the OpenAI client between requests. When doing anything with large "
            "volumes of async API calls, setting this to false can improve stability."
        ),
    )

    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat with the model."""
        if self.collapse_system_prompt:
            messages = _reformat_messages(messages)

        return super().chat(messages, **kwargs)

    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Stream chat with the model."""
        if self.collapse_system_prompt:
            messages = _reformat_messages(messages)

        return super().stream_chat(messages, **kwargs)

    # -- Async methods --
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Chat with the model."""
        if self.collapse_system_prompt:
            messages = _reformat_messages(messages)

        return await super().achat(messages, **kwargs)

    async def astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Stream chat with the model."""
        if self.collapse_system_prompt:
            messages = _reformat_messages(messages)

        return await super().astream_chat(messages, **kwargs)

    @llm_retry_decorator
    def _chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        client = self._get_client()
        message_dicts = to_openai_message_dicts(messages)
        with client:
            response = client.chat.completions.create(
                messages=message_dicts,
                stream=False,
                **self._get_model_kwargs(**kwargs),
            )
            openai_message = response.choices[0].message
            message = from_openai_message(openai_message)
            openai_token_logprobs = response.choices[0].logprobs
            logprobs = None
            if openai_token_logprobs and openai_token_logprobs.content:
                logprobs = from_openai_token_logprobs(openai_token_logprobs.content)

            return ChatResponse(
                message=message,
                raw=response,
                logprobs=logprobs,
                additional_kwargs=self._get_response_token_counts(response),
            )

    @llm_retry_decorator
    def _stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        client = self._get_client()
        message_dicts = to_openai_message_dicts(messages)

        def gen() -> ChatResponseGen:
            content = ""
            tool_calls: list[ChoiceDeltaToolCall] = []

            is_function = False
            with client:
                for response in client.chat.completions.create(
                    messages=message_dicts,
                    stream=True,
                    **self._get_model_kwargs(**kwargs),
                ):
                    response = cast(ChatCompletionChunk, response)
                    if len(response.choices) > 0:
                        delta = response.choices[0].delta
                    else:
                        if self._is_azure_client():
                            continue
                        else:
                            delta = ChoiceDelta()

                    # check if this chunk is the start of a function call
                    if delta.tool_calls:
                        is_function = True

                    # update using deltas
                    role = delta.role or MessageRole.ASSISTANT
                    content_delta = delta.content or ""
                    content += content_delta

                    additional_kwargs = {}
                    if is_function:
                        tool_calls = self._update_tool_calls(tool_calls, delta.tool_calls)
                        additional_kwargs["tool_calls"] = tool_calls

                    yield ChatResponse(
                        message=ChatMessage(
                            role=role,
                            content=content,
                            additional_kwargs=additional_kwargs,
                        ),
                        delta=content_delta,
                        raw=response,
                        additional_kwargs=self._get_response_token_counts(response),
                    )

        return gen()

    @llm_retry_decorator
    async def _achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        aclient = self._get_aclient()
        message_dicts = to_openai_message_dicts(messages)
        async with aclient:
            response = await aclient.chat.completions.create(
                messages=message_dicts,
                stream=False,
                **self._get_model_kwargs(**kwargs),
            )
            openai_message = response.choices[0].message
            message = from_openai_message(openai_message)
            openai_token_logprobs = response.choices[0].logprobs
            logprobs = None
            if openai_token_logprobs and openai_token_logprobs.content:
                logprobs = from_openai_token_logprobs(openai_token_logprobs.content)

            return ChatResponse(
                message=message,
                raw=response,
                logprobs=logprobs,
                additional_kwargs=self._get_response_token_counts(response),
            )

    @llm_retry_decorator
    async def _astream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        aclient = self._get_aclient()
        message_dicts = to_openai_message_dicts(messages)

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            tool_calls: list[ChoiceDeltaToolCall] = []

            is_function = False
            first_chat_chunk = True
            async with aclient:
                async for response in await aclient.chat.completions.create(
                    messages=message_dicts,
                    stream=True,
                    **self._get_model_kwargs(**kwargs),
                ):
                    response = cast(ChatCompletionChunk, response)
                    if len(response.choices) > 0:
                        # check if the first chunk has neither content nor tool_calls
                        # this happens when 1106 models end up calling multiple tools
                        if (
                            first_chat_chunk
                            and response.choices[0].delta.content is None
                            and response.choices[0].delta.tool_calls is None
                        ):
                            first_chat_chunk = False
                            continue
                        delta = response.choices[0].delta
                    else:
                        if self._is_azure_client():
                            continue
                        else:
                            delta = ChoiceDelta()
                    first_chat_chunk = False

                    # check if this chunk is the start of a function call
                    if delta.tool_calls:
                        is_function = True

                    # update using deltas
                    role = delta.role or MessageRole.ASSISTANT
                    content_delta = delta.content or ""
                    content += content_delta

                    additional_kwargs = {}
                    if is_function:
                        tool_calls = self._update_tool_calls(tool_calls, delta.tool_calls)
                        additional_kwargs["tool_calls"] = tool_calls

                    yield ChatResponse(
                        message=ChatMessage(
                            role=role,
                            content=content,
                            additional_kwargs=additional_kwargs,
                        ),
                        delta=content_delta,
                        raw=response,
                        additional_kwargs=self._get_response_token_counts(response),
                    )

        return gen()
