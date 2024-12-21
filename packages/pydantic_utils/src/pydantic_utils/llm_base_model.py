import warnings
from collections.abc import Awaitable, Callable
from typing import Self

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field
from rag.llms.openai_like import OpenAILike
from rag.retry import create_retry_decorator
from utils import trimAndLoadJson

from pydantic_utils.prompts import default_system_prompt, prompt_revise_json_format

retry_decorator = create_retry_decorator(
    max_retries=10,
    random_exponential=True,
    stop_after_delay_seconds=60,
    min_seconds=1,
    max_seconds=20,
)


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
        default=5,
        description="Number of attempts to use LLM to correct output.",
    )
    system_prompt_template: Callable = Field(
        default=default_system_prompt,
        description="System prompt function. Overrides default LLM system prompt.",
    )

    @classmethod
    def class_name(cls) -> str:
        return "LLMBaseModel"

    @classmethod
    def from_defaults(
        cls,
        llm: OpenAILike | OpenAI | None = None,
        num_workers: int = 8,
        num_invalid_output_retries: int = 3,
    ) -> Self:
        return cls(
            llm=llm,
            num_workers=num_workers,
            num_invalid_output_retries=num_invalid_output_retries,
        )

    @retry_decorator
    def _generate(
        self,
        *args,
        prompt_fn: Callable[[str], str],
        pydantic_model: BaseModel | None = None,
        **kwargs,
    ) -> BaseModel | str:
        """Text generation using LLM and prompt function.  Any args and kwargs provided in
        the function call are passed as arguments to `prompt_fn`.
        If `pydantic_model` is None, this method acts as a normal LLM text generation call and
        the LLM response text is returned.
        If `pydantic_model` is provided, the model's JSON schema is used for constrained generation
        and a parsed pydantic model object is returned."""
        self.llm.system_prompt = self.system_prompt_template()
        # Guided JSON using Outlines if Pydantic Model provided
        if pydantic_model:
            if "extra_body" not in self.llm.additional_kwargs:
                self.llm.additional_kwargs["extra_body"] = {}
            self.llm.additional_kwargs["extra_body"] |= {
                "guided_json": pydantic_model.model_json_schema(),
                "guided_whitespace_pattern": " ",
            }
        messages = [ChatMessage(role=MessageRole.USER, content=prompt_fn(*args, **kwargs))]
        response = self.llm.chat(messages=messages)
        # Parse to Pydantic Object if Pydantic Model provided
        if pydantic_model:
            obj = self._parse_response_or_revise(response.message.content, pydantic_model)
            return obj
        else:
            return response.message.content

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
                for _ in range(self.num_invalid_output_retries):
                    try:
                        obj = self._generate(
                            text=text,
                            prompt_fn=prompt_revise_json_format,
                            pydantic_model=pydantic_model,
                        )
                        break
                    except Exception:
                        warnings.warn(
                            f"Failed to parse text object to pydantic model. Text: {text}.",
                            stacklevel=2,
                        )
            return obj

    @retry_decorator
    async def _a_generate(
        self,
        *args,
        prompt_fn: Callable[[str], str],
        pydantic_model: BaseModel | None = None,
        **kwargs,
    ) -> Awaitable[BaseModel | str]:
        """Text generation using LLM and prompt function.  Any args and kwargs provided in
        the function call are passed as arguments to `prompt_fn`.
        If `pydantic_model` is None, this method acts as a normal LLM text generation call and
        the LLM response text is returned.
        If `pydantic_model` is provided, the model's JSON schema is used for constrained generation
        and a parsed pydantic model object is returned."""
        self.llm.system_prompt = self.system_prompt_template()
        # Guided JSON using Outlines if Pydantic Model provided
        if pydantic_model:
            if "extra_body" not in self.llm.additional_kwargs:
                self.llm.additional_kwargs["extra_body"] = {}
            self.llm.additional_kwargs["extra_body"] |= {
                "guided_json": pydantic_model.model_json_schema(),
                "guided_whitespace_pattern": " ",
            }
        messages = [ChatMessage(role=MessageRole.USER, content=prompt_fn(*args, **kwargs))]
        response = await self.llm.achat(messages=messages)
        # Parse to Pydantic Object if Pydantic Model provided
        if pydantic_model:
            obj = self._parse_response_or_revise(response.message.content, pydantic_model)
            return obj
        else:
            return response.message.content

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
                for _ in range(self.num_invalid_output_retries):
                    try:
                        obj = await self._a_generate(
                            text=text,
                            prompt_fn=prompt_revise_json_format,
                            pydantic_model=pydantic_model,
                        )
                        break
                    except Exception:
                        warnings.warn(
                            f"Failed to parse text object to pydantic model. Text: {text}.",
                            stacklevel=2,
                        )
            return obj
