import logging
import warnings
from collections.abc import Awaitable, Callable
from typing import Self

from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from rag.llms.openai_like import OpenAILike
from rag.retry import RetryValueError, create_retry_decorator
from utils import trimAndLoadJson

from pydantic_utils.prompts import default_system_prompt, prompt_revise_json_format

retry_decorator = create_retry_decorator(
    max_retries=10,
    random_exponential=True,
    stop_after_delay_seconds=60,
    min_seconds=1,
    max_seconds=20,
)


def string_wrapper(string: str) -> Callable[[], str]:
    """Wraps string as a callable function. Used to convert strings to prompt functions."""

    def callable_string_fn() -> str:
        return string

    return callable_string_fn


class LLMBaseModel(BaseModel):
    """Pydantic BaseModel which contains generic methods for LLM generation calls.
    The LLM is a OpenAILike model which is deployed on vLLM and supports the
    extra_body parameter for guided JSON generation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: OpenAILike | OpenAI | None = Field(
        default=None, description="LLM model to use for generation.", exclude=True
    )
    num_workers: int = Field(
        default=16, description="Number of async LLM generations to run in parallel."
    )
    num_invalid_output_retries: int = Field(
        default=10,
        description="Number of attempts to use LLM to correct output.",
    )
    system_prompt_template: Callable | str = Field(
        default=default_system_prompt,
        description="System prompt function. Overrides default LLM system prompt.",
    )
    default_temperature: float = Field(
        default=0.1,
        description="Default temperature for LLM generation.",
    )
    retry_temperature_increment: float = Field(
        default=0.1,
        description="Temperature increment for retrying LLM generation.",
    )
    retry_temperature_max: float = Field(
        default=1.0,
        description="Maximum temperature for retrying LLM generation.",
    )
    logger: logging.Logger | None = Field(
        default=None, description="Logger object.", exclude=True, repr=False
    )

    @classmethod
    def class_name(cls) -> str:
        return "LLMBaseModel"

    @classmethod
    def from_defaults(
        cls,
        llm: OpenAILike | OpenAI | None = None,
        system_prompt_template: Callable[[], str] | str | None = default_system_prompt,
        retry_temperature_increment: float = 0.1,
        retry_temperature_max: float = 1.0,
        num_workers: int = 8,
        num_invalid_output_retries: int = 10,
    ) -> Self:
        return cls(
            llm=llm,
            system_prompt_template=system_prompt_template,
            default_temperature=llm.temperature,
            retry_temperature_increment=retry_temperature_increment,
            retry_temperature_max=retry_temperature_max,
            num_workers=num_workers,
            num_invalid_output_retries=num_invalid_output_retries,
        )

    @retry_decorator
    def _generate(
        self,
        *args,
        prompt: Callable[[str], str] | str,
        system_prompt: Callable[[], str] | str | None = None,
        response_format: BaseModel | None = None,
        validate_response_fn: Callable[[BaseModel], bool] | None = None,
        return_raw_response: bool = False,
        llm: OpenAILike | OpenAI | None = None,
        llm_kwargs: dict = {},
        **kwargs,
    ) -> ChatResponse | BaseModel | str:
        """Text generation using LLM and prompt function.  Any args and kwargs provided in
        the function call are passed as arguments to `prompt`.
        If `response_format` is None, this method acts as a normal LLM text generation call and
        the LLM response text is returned.
        If `response_format` is provided, the model's JSON schema is used for constrained generation
        and a parsed pydantic model object is returned."""
        llm = llm or self.llm
        llm.additional_kwargs |= llm_kwargs
        # If prompt or system_prompt is a string, wrap it in a function
        if isinstance(prompt, str):
            prompt = string_wrapper(prompt)
        if isinstance(system_prompt, str):
            system_prompt = string_wrapper(system_prompt)
        llm.system_prompt = system_prompt or self.system_prompt_template()
        # Guided JSON using Outlines if Pydantic Model provided
        if response_format:
            if "extra_body" not in llm.additional_kwargs:
                llm.additional_kwargs["extra_body"] = {}
            llm.additional_kwargs["extra_body"] |= {
                "guided_json": response_format.model_json_schema(),
                "guided_whitespace_pattern": " ",
            }
        messages = [ChatMessage(role=MessageRole.USER, content=prompt(*args, **kwargs))]
        response = llm.chat(messages=messages)
        # Check Response Finish Reason
        finish_reason = response.raw.choices[0].finish_reason
        if finish_reason == "stop":  # Regular completion
            # Return parsed Pydantic Object
            if response_format:
                obj = self._parse_response_or_revise(response.message.content, response_format)
                # Check if Pydantic Object is valid using custom validation function
                if validate_response_fn:
                    validate_result = validate_response_fn(obj)
                    if validate_result is True:
                        llm.temperature = self.default_temperature  # Reset LLM temperature
                        return obj
                    else:
                        err_string = validate_result
                        # Invalid Pydantic Object. Retry with LLM with temperature increment.
                        if llm.temperature <= self.retry_temperature_max:
                            llm.temperature += self.retry_temperature_increment
                        err_msg = (
                            "Pydantic Object Response failed custom validation. "
                            f"{err_string}"
                            f"Retrying with temperature: {llm.temperature}."
                        )
                        if self.logger:
                            self.logger.critical(err_msg)
                        raise RetryValueError(err_msg)
                else:
                    return obj
            # Return ChatResponse object, which wraps raw ChatCompletion
            elif return_raw_response:
                llm.temperature = self.default_temperature  # Reset LLM temperature
                return response
            # Return parsed response content
            else:
                llm.temperature = self.default_temperature  # Reset LLM temperature
                return response.message.content
        elif finish_reason == "length":  # Response was cut-off
            # NOTE: weaker/quantized models can get into infinite loop generation,
            # which is detected by this condition. We increment temperature to
            # increase probability of generating other tokens and avoid infinite loop.
            # Retry with higher temperature
            if llm.temperature <= self.retry_temperature_max:
                llm.temperature += self.retry_temperature_increment
            err_msg = f"Response was cut-off. Retrying with temperature: {llm.temperature}."
            if self.logger:
                self.logger.critical(err_msg)
            raise RetryValueError(err_msg)
        else:  # Other finish reasons (e.g. "content_filter", "tool_call")
            # Retry with same temperature
            err_msg = f"Retrying. LLM ChatCompletion finish reason: {finish_reason}. "
            if self.logger:
                self.logger.critical(err_msg)
            raise RetryValueError(err_msg)

    def _parse_response_or_revise(self, text: str, pydantic_model: BaseModel) -> BaseModel:
        """Parses response from LLM."""
        try:
            obj = pydantic_model.model_validate_json(text)
            return obj
        except Exception:
            try:
                # If fail to parse valid json, try to trim string and convert to python
                # dict first before validating object with pydantic model
                pydict = trimAndLoadJson(text)
                obj = pydantic_model.model_validate(pydict)
            except Exception:
                # If fail to parse valid json, try to use LLM to correct JSON format
                obj = None
                try:
                    obj = self._generate(
                        text=text,
                        prompt=prompt_revise_json_format,
                        pydantic_model=pydantic_model,
                    )
                    return obj
                except Exception:
                    warnings.warn(
                        f"Failed to parse text object to pydantic model. Text: {text}.",
                        stacklevel=2,
                    )

    @retry_decorator
    async def _a_generate(
        self,
        *args,
        prompt: Callable[[str], str] | str,
        system_prompt: Callable[[], str] | str | None = None,
        response_format: BaseModel | None = None,
        validate_response_fn: Callable[[BaseModel], bool] | None = None,
        return_raw_response: bool = False,
        llm: OpenAILike | OpenAI | None = None,
        llm_kwargs: dict = {},
        **kwargs,
    ) -> Awaitable[ChatResponse | BaseModel | str]:
        """Text generation using LLM and prompt function.  Any args and kwargs provided in
        the function call are passed as arguments to `prompt`.
        If `response_format` is None, this method acts as a normal LLM text generation call and
        the LLM response text is returned.
        If `response_format` is provided, the model's JSON schema is used for constrained generation
        and a parsed pydantic model object is returned."""
        llm = llm or self.llm
        llm.additional_kwargs |= llm_kwargs
        # If prompt or system_prompt is a string, wrap it in a function
        if isinstance(prompt, str):
            prompt = string_wrapper(prompt)
        if isinstance(system_prompt, str):
            system_prompt = string_wrapper(system_prompt)
        llm.system_prompt = system_prompt or self.system_prompt_template()
        # Guided JSON using Outlines if Pydantic Model provided
        if response_format:
            if "extra_body" not in llm.additional_kwargs:
                llm.additional_kwargs["extra_body"] = {}
            llm.additional_kwargs["extra_body"] |= {
                "guided_json": response_format.model_json_schema(),
                "guided_whitespace_pattern": " ",
            }
        messages = [ChatMessage(role=MessageRole.USER, content=prompt(*args, **kwargs))]
        response = await llm.achat(messages=messages)
        # Check Response Finish Reason
        finish_reason = response.raw.choices[0].finish_reason
        if finish_reason == "stop":  # Regular completion
            # Return parsed to Pydantic Object
            if response_format:
                obj = await self._a_parse_response_or_revise(
                    response.message.content, response_format
                )
                # Check if Pydantic Object is valid using custom validation function
                if validate_response_fn:
                    validate_result = validate_response_fn(obj)
                    if validate_result is True:
                        llm.temperature = self.default_temperature  # Reset LLM temperature
                        return obj
                    else:
                        err_string = validate_result
                        # Invalid Pydantic Object. Retry with LLM with temperature increment.
                        if llm.temperature <= self.retry_temperature_max:
                            llm.temperature += self.retry_temperature_increment
                        err_msg = (
                            "Pydantic Object Response failed custom validation. "
                            f"{err_string}"
                            f"Retrying with temperature: {llm.temperature}."
                        )
                        if self.logger:
                            self.logger.critical(err_msg)
                        raise RetryValueError(err_msg)
                else:
                    return obj
            # Return ChatResponse object, which wraps raw ChatCompletion
            elif return_raw_response:
                llm.temperature = self.default_temperature  # Reset LLM temperature
                return response
            # Return parsed response content
            else:
                llm.temperature = self.default_temperature  # Reset LLM temperature
                return response.message.content
        elif finish_reason == "length":  # Response was cut-off
            # NOTE: weaker/quantized models can get into infinite loop generation,
            # which is detected by this condition. We increment temperature to
            # increase probability of generating other tokens and avoid infinite loop.
            # Retry with higher temperature
            if llm.temperature <= self.retry_temperature_max:
                llm.temperature += self.retry_temperature_increment
            err_msg = f"Response was cut-off. Retrying with temperature: {llm.temperature}."
            if self.logger:
                self.logger.critical(err_msg)
            raise RetryValueError(err_msg)
        else:  # Other finish reasons (e.g. "content_filter", "tool_call")
            # Retry with same temperature
            err_msg = f"Retrying. LLM ChatCompletion finish reason: {finish_reason}. "
            if self.logger:
                self.logger.critical(err_msg)
            raise RetryValueError(err_msg)

    async def _a_parse_response_or_revise(
        self, text: str, pydantic_model: BaseModel
    ) -> Awaitable[BaseModel]:
        """Parses response from LLM."""
        try:
            obj = pydantic_model.model_validate_json(text)
            return obj
        except Exception:
            try:
                # If fail to parse valid json, try to trim string and convert to python
                # dict first before validating object with pydantic model
                pydict = trimAndLoadJson(text)
                obj = pydantic_model.model_validate(pydict)
            except Exception:
                # If fail to parse valid json, try to use LLM to correct JSON format
                obj = None
                try:
                    obj = await self._a_generate(
                        text=text,
                        prompt=prompt_revise_json_format,
                        pydantic_model=pydantic_model,
                    )
                    return obj
                except Exception:
                    warnings.warn(
                        f"Failed to parse text object to pydantic model. Text: {text}.",
                        stacklevel=2,
                    )


class LLM(LLMBaseModel):
    """Wrapper around LLMBaseModel which supports `call` and `a_call` methods."""

    @classmethod
    def class_name(cls) -> str:
        return "LLM"

    def __call__(
        self,
        *args,
        prompt: Callable[[str], str] | str,
        system_prompt: Callable[[], str] | str | None = default_system_prompt,
        response_format: BaseModel | None = None,
        validate_response_fn: Callable[[BaseModel], bool] | None = None,
        return_raw_response: bool = False,
        llm: OpenAILike | OpenAI | None = None,
        llm_kwargs: dict = {},
        **kwargs,
    ) -> str | BaseModel:
        return self.call(
            *args,
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format,
            validate_response_fn=validate_response_fn,
            return_raw_response=return_raw_response,
            llm=llm,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )

    def call(
        self,
        *args,
        prompt: Callable[[str], str] | str,
        system_prompt: Callable[[], str] | str | None = default_system_prompt,
        response_format: BaseModel | None = None,
        validate_response_fn: Callable[[BaseModel], bool] | None = None,
        return_raw_response: bool = False,
        llm: OpenAILike | OpenAI | None = None,
        llm_kwargs: dict = {},
        **kwargs,
    ) -> str | BaseModel:
        """Text generation using LLM and prompt, optionally with capability to enforce
        structured outputs response format.

        Args:
            prompt (Callable[[str], str] | str): Prompt or prompt function. Any args and kwargs
                in the function call are passed as arguments to `prompt` if it is a Callable.
                If `prompt` is a string, it is used as the prompt.
            system_prompt (Callable[[], str] | str | None, optional): The system prompt.
                Defaults to default_system_prompt.
            response_format (BaseModel | None, optional): Pydantic model used to define
                structured outputs response format. If None, then the LLM generation does
                not use structured outputes. Defaults to None.
            validate_response_fn (Callable[[BaseModel], bool] | None, optional): Custom
                validation function for Pydantic model output that checkins if
                pydantic object values conforms to desired value options and returns boolean.
                If None, no validation is performed. Defaults to None.
            return_raw_response (bool, optional): If True, returns the raw ChatResponse object
                from the LLM model. Defaults to False.
            llm (OpenAILike | OpenAI | None, optional): LLM model to use for generation.
                If None, the default LLM model is used. Defaults to None.
            llm_kwargs (dict, optional): Additional kwargs to pass to the LLM model. This
                gets merged with the default llm_kwargs and can override the defaults.
                Defaults to {}.

        Returns:
            str | BaseModel:
                If `response_format` is None and `llm_kwargs` does not request logprobs,
                this method acts as a normal LLM text generation call and a string is returned.
                If `response_format` is a Pydantic model, the model's JSON schema is
                used for structured output and a parsed pydantic model object is returned.
                If `llm_kwargs` requests logprobs, a LogProbsResponse object is returned.
                with the normal response text or Pydantic model structured output under
                attribute `content` and the logprobs under attribute `logprobs`.
                If `response_format` is a Pydantic model and `llm_kwargs` requests logprobs,`
                then the LogProbsResponse object is returned with the Pydantic model structured
                output under attribute `content` and the logprobs under attribute `logprobs`.
        """
        return self._generate(
            *args,
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format,
            validate_response_fn=validate_response_fn,
            return_raw_response=return_raw_response,
            llm=llm,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )

    async def a_call(
        self,
        *args,
        prompt: Callable[[str], str] | str,
        system_prompt: Callable[[], str] | str | None = default_system_prompt,
        response_format: BaseModel | None = None,
        validate_response_fn: Callable[[BaseModel], bool] | None = None,
        return_raw_response: bool = False,
        llm: OpenAILike | OpenAI | None = None,
        llm_kwargs: dict = {},
        **kwargs,
    ) -> Awaitable[str | BaseModel]:
        """Text generation using LLM and prompt, optionally with capability to enforce
        structured outputs response format. This is an async version of `call`.

        Args:
            prompt (Callable[[str], str] | str): Prompt or prompt function. Any args and kwargs
                in the function call are passed as arguments to `prompt` if it is a Callable.
                If `prompt` is a string, it is used as the prompt.
            system_prompt (Callable[[], str] | str | None, optional): The system prompt.
                Defaults to default_system_prompt.
            response_format (BaseModel | None, optional): Pydantic model used to define
                structured outputs response format. If None, then the LLM generation does
                not use structured outputes. Defaults to None.
            validate_response_fn (Callable[[BaseModel], bool] | None, optional): Custom
                validation function for Pydantic model output that checkins if
                pydantic object values conforms to desired value options and returns boolean.
                If None, no validation is performed. Defaults to None.
            return_raw_response (bool, optional): If True, returns the raw ChatResponse object
                from the LLM model. Defaults to False.
            llm (OpenAILike | OpenAI | None, optional): LLM model to use for generation.
                If None, the default LLM model is used. Defaults to None.
            llm_kwargs (dict, optional): Additional kwargs to pass to the LLM model. This
                gets merged with the default llm_kwargs and can override the defaults.
                Defaults to {}.

        Returns:
            str | BaseModel:
                If `response_format` is None and `llm_kwargs` does not request logprobs,
                this method acts as a normal LLM text generation call and a string is returned.
                If `response_format` is a Pydantic model, the model's JSON schema is
                used for structured output and a parsed pydantic model object is returned.
                If `llm_kwargs` requests logprobs, a LogProbsResponse object is returned.
                with the normal response text or Pydantic model structured output under
                attribute `content` and the logprobs under attribute `logprobs`.
                If `response_format` is a Pydantic model and `llm_kwargs` requests logprobs,`
                then the LogProbsResponse object is returned with the Pydantic model structured
                output under attribute `content` and the logprobs under attribute `logprobs`.
        """
        return await self._a_generate(
            *args,
            prompt=prompt,
            system_prompt=system_prompt,
            response_format=response_format,
            validate_response_fn=validate_response_fn,
            return_raw_response=return_raw_response,
            llm=llm,
            llm_kwargs=llm_kwargs,
            **kwargs,
        )
