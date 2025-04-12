import logging
from collections.abc import Awaitable
from pathlib import Path
from typing import Self

from llama_index.core.llms import ChatResponse
from llama_index.llms.openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage
from pydantic import BaseModel, ConfigDict, Field
from pydantic_utils import LLM
from rag.llms.openai_like import OpenAILike
from tqdm.asyncio import tqdm
from utils import LazyFileLogger, run_jobs

from llm_judge.prompts import (
    prompt_determine_verdict_fewshot,
    prompt_determine_verdict_reasoning,
    prompt_json_structured_output,
    system_prompt_judge,
)
from llm_judge.schema import InputTextAndReferenceContext, SimpleVerdict, Verdict
from llm_judge.score_report import ScoreReport

SUPPORTED_LABEL = "Supported"
NOT_SUPPORTED_LABEL = "Not Supported"
NOT_ADDRESSED_LABEL = "Not Addressed"
VERDICT_LABELS = [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL, NOT_ADDRESSED_LABEL]


class Judge(BaseModel):
    """LLM-as-a-Judge which evaluates input text based on a reference context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: LLM | None = Field(description="LLM model to use for generation.")
    aux_llm: LLM | None = Field(
        description="Auxiliary LLM model to use for structured output generation."
    )
    is_reasoning_model: bool = Field(
        default=False,
        description="Whether LLM is a reasoning model or not.",
    )
    num_workers: int = Field(
        default=32, description="Number of async LLM generations to run in parallel."
    )
    num_invalid_output_retries: int = Field(
        default=10,
        description="Number of attempts to use LLM to correct output.",
    )
    include_summary_explanations: bool = Field(
        default=True,
        description="Include summary explanations for Supported, Not Supported, and "
        "Not Addressed texts.",
    )
    logger: logging.Logger | None = Field(
        default=None, description="Logger object.", exclude=True, repr=False
    )

    @classmethod
    def class_name(cls) -> str:
        return "Judge"

    @classmethod
    def from_defaults(
        cls,
        llm: OpenAILike | OpenAI | None = None,
        aux_llm: OpenAILike | OpenAI | None = None,
        is_reasoning_model: bool = False,
        num_workers: int = 16,
        num_invalid_output_retries: int = 10,
        log_filepath: str | Path | None = None,
        log_level: str = logging.DEBUG,
    ) -> Self:
        from utils import load_environment

        load_environment()
        log_filepath = Path(log_filepath) if log_filepath else None
        if log_filepath:
            logger = LazyFileLogger(name=__name__, level=log_level, log_file=log_filepath)
        else:
            logger = None

        if is_reasoning_model and not aux_llm:
            raise ValueError("Auxiliary LLM model required for parsing reasoning model outputs.")

        # Two LLMs if Reasoning Model and One LLM if Non-Reasoning Model.
        # - Reasoning Model: LLM generation w/ Reasoning w/o Structured Output,
        # followed by Auxillary-LLM generation to extract Structured Output
        # - Non-Reasoning Model: LLM generation w/ Structured Output
        llm_wrapper = LLM.from_defaults(
            llm=llm,
            system_prompt_template=system_prompt_judge,
            num_workers=num_workers,
            num_invalid_output_retries=num_invalid_output_retries,
        )
        aux_llm_wrapper = (
            LLM.from_defaults(
                llm=aux_llm,
                system_prompt_template=system_prompt_judge,
                num_workers=num_workers,
                num_invalid_output_retries=num_invalid_output_retries,
            )
            if aux_llm and is_reasoning_model
            else None
        )

        return cls(
            llm=llm_wrapper,
            aux_llm=aux_llm_wrapper,
            is_reasoning_model=is_reasoning_model,
            num_workers=num_workers,
            num_invalid_output_retries=num_invalid_output_retries,
            logger=logger,
        )

    def setup_logger(
        self, level: str = logging.DEBUG, log_file: str | Path | None = None
    ) -> LazyFileLogger:
        self.logger = LazyFileLogger(name=__name__, level=level, log_file=log_file)
        return self.logger

    def __call__(
        self,
        texts: list[str],
        references: list[str],
        proposition_ids: list[str] | None = None,
        proposition_type: str = "",
        fact_type: str = "",
        include_explanations: bool = True,
        show_progress: bool = False,
        **kwargs,
    ) -> ScoreReport:
        return self.evaluate(
            texts=texts,
            references=references,
            proposition_ids=proposition_ids,
            proposition_type=proposition_type,
            fact_type=fact_type,
            include_explanations=include_explanations,
            show_progress=show_progress,
            **kwargs,
        )

    async def acall(
        self,
        texts: list[str],
        references: list[str],
        proposition_ids: list[str] | None = None,
        proposition_type: str | None = None,
        fact_type: str | None = None,
        include_explanations: bool = True,
        show_progress: bool = False,
        **kwargs,
    ) -> Awaitable[ScoreReport]:
        return await self.a_evaluate(
            texts=texts,
            references=references,
            proposition_ids=proposition_ids,
            proposition_type=proposition_type,
            fact_type=fact_type,
            include_explanations=include_explanations,
            show_progress=show_progress,
            **kwargs,
        )

    def evaluate(
        self,
        texts: list[str],
        references: list[str],
        proposition_ids: list[str] | None = None,
        proposition_type: str | None = None,
        fact_type: str | None = None,
        include_explanations: bool = True,
        show_progress: bool = False,
        desc="Generating Verdicts",
        **kwargs,
    ) -> ScoreReport:
        """Evaluate each text against its corresponding reference."""
        proposition_ids = [None] * len(texts) if proposition_ids is None else proposition_ids

        if self.is_reasoning_model:
            verdicts = [
                self._determine_reasoning_verdict(
                    text=text, reference=reference, proposition_id=proposition_id
                )
                for text, reference, proposition_id in tqdm(
                    zip(texts, references, proposition_ids, strict=False),
                    show_progress=show_progress,
                    desc=desc,
                )
            ]
        else:
            verdicts = [
                self._determine_verdict(
                    text=text, reference=reference, proposition_id=proposition_id
                )
                for text, reference, proposition_id in tqdm(
                    zip(texts, references, proposition_ids, strict=False),
                    show_progress=show_progress,
                    desc=desc,
                )
            ]
        return ScoreReport.from_defaults(
            verdicts=verdicts,
            proposition_type=proposition_type,
            fact_type=fact_type,
            include_explanations=include_explanations,
            llm=self.aux_llm.llm if self.is_reasoning_model else self.llm.llm,
            num_workers=self.num_workers,
            num_invalid_output_retries=self.num_invalid_output_retries,
            **kwargs,
        )

    async def a_evaluate(
        self,
        texts: list[str],
        references: list[str],
        proposition_ids: list[str] | None = None,
        include_explanations: bool = True,
        show_progress: bool = False,
        desc: str = "Generating Verdicts",
        **kwargs,
    ) -> Awaitable[ScoreReport]:
        """Evaluate each text against its corresponding reference."""
        proposition_ids = [None] * len(texts) if proposition_ids is None else proposition_ids
        if self.is_reasoning_model:
            jobs = [
                self._a_determine_reasoning_verdict(
                    text=text, reference=reference, proposition_id=proposition_id
                )
                for text, reference, proposition_id in zip(
                    texts, references, proposition_ids, strict=False
                )
            ]
            verdicts = await run_jobs(
                jobs, workers=self.num_workers, show_progress=show_progress, desc=desc
            )
        else:
            jobs = [
                self._a_determine_verdict(
                    text=text, reference=reference, proposition_id=proposition_id
                )
                for text, reference, proposition_id in zip(
                    texts, references, proposition_ids, strict=False
                )
            ]
            verdicts = await run_jobs(
                jobs, workers=self.num_workers, show_progress=show_progress, desc=desc
            )
        return await ScoreReport.a_from_defaults(
            verdicts=verdicts,
            include_explanations=include_explanations,
            llm=self.aux_llm.llm if self.is_reasoning_model else self.llm.llm,
            num_workers=self.num_workers,
            num_invalid_output_retries=self.num_invalid_output_retries,
            **kwargs,
        )

    def validate_simple_verdict(self, obj: SimpleVerdict) -> bool | str:
        """Simple validation function that returns True if the verdict is in the set
        of {"Supported", "Not Supported", "Not Addressed"}. If not, an error string
        is returned."""
        try:
            # Capitalize first letter of each word in the label
            obj.verdict = obj.verdict.title()
            # Check if Verdict is one of the valid options
            return obj.verdict in VERDICT_LABELS
        except Exception as ex:
            log_str = f"Result Object: {obj}\n"
            if hasattr(obj, "verdict"):
                log_str += f"Verdict: {obj.verdict}\n"
            log_str += f"Exception: {ex}\n"
            if self.logger:
                self.logger.critical(log_str)
            return log_str

    def _determine_verdict(
        self,
        text: str,
        reference: str,
        proposition_id: str | None = None,
    ) -> Verdict:
        """Determines verdict on whether text is supported by reference."""
        # Format inputs as JSON string
        json_str = InputTextAndReferenceContext(text=text, reference=reference).model_dump_json(
            indent=4
        )

        # Several tries to generate verdict
        for i in range(self.num_invalid_output_retries):
            try:
                obj = None
                # Constrained generation using SimpleVerdict model JSON schema
                obj: SimpleVerdict = self.llm.call(
                    json_str,
                    prompt=prompt_determine_verdict_fewshot,
                    response_format=SimpleVerdict,
                    validate_response_fn=self.validate_simple_verdict,
                )
                # Convert to Verdict model to include text and reference
                obj: Verdict = Verdict.from_simple_verdict(
                    simple_verdict=obj,
                    text=text,
                    reference=reference,
                    proposition_id=proposition_id or "",
                )
                # Capitalize first letter of each word in the label
                obj.verdict = obj.verdict.title()
                # If verdict is valid, break out of loop
                if self.validate_simple_verdict(obj):
                    break
                else:
                    raise ValueError(f"Invalid Verdict: {obj.verdict}")
            except Exception as ex:
                if self.logger:
                    llm = self.llm.llm if isinstance(self.llm, LLM) else self.llm
                    log_str = (
                        f"Determine Verdict Try {i}. "
                        f"Temperature={llm.temperature}, "
                        f"Top_p={llm.additional_kwargs['top_p']}.\n"
                        f"Input JSON: {json_str}\n"
                    )
                    if obj is not None:
                        log_str += f"Result Object: {obj}\n"
                        if hasattr(obj, "verdict"):
                            log_str += f"Verdict: {obj.verdict}\n"
                    log_str += f"Exception: {ex}\n"
                    self.logger.critical(log_str)
        return obj

    async def _a_determine_verdict(
        self,
        text: str,
        reference: str,
        proposition_id: str | None = None,
    ) -> Awaitable[Verdict]:
        """Determines verdict on whether text is supported by reference."""
        # Format inputs as JSON string
        json_str = InputTextAndReferenceContext(text=text, reference=reference).model_dump_json(
            indent=4
        )
        # Several tries to generate verdict
        for i in range(self.num_invalid_output_retries):
            try:
                obj = None
                # Constrained generation using SimpleVerdict model JSON schema
                obj: SimpleVerdict = await self.llm.a_call(
                    json_str,
                    prompt=prompt_determine_verdict_fewshot,
                    response_format=SimpleVerdict,
                    validate_response_fn=self.validate_simple_verdict,
                )
                # Convert to Verdict model to include text and reference
                obj: Verdict = Verdict.from_simple_verdict(
                    simple_verdict=obj,
                    text=text,
                    reference=reference,
                    proposition_id=proposition_id or "",
                )
                # Capitalize first letter of each word in the label
                obj.verdict = obj.verdict.title()
                # If verdict is valid, break out of loop
                if self.validate_simple_verdict(obj):
                    break
                else:
                    raise ValueError(f"Invalid Verdict: {obj.verdict}")
            except Exception as ex:
                if self.logger:
                    llm = self.llm.llm if isinstance(self.llm, LLM) else self.llm
                    log_str = (
                        f"Determine Verdict Try {i}. "
                        f"Temperature={llm.temperature}, "
                        f"Top_p={llm.additional_kwargs['top_p']}.\n"
                        f"Input JSON: {json_str}\n"
                    )
                    if obj is not None:
                        log_str += f"Result Object: {obj}\n"
                        if hasattr(obj, "verdict"):
                            log_str += f"Verdict: {obj.verdict}\n"
                    log_str += f"Exception: {ex}\n"

                    self.logger.critical(log_str)
        return obj

    def _determine_reasoning_verdict(
        self,
        text: str,
        reference: str,
        proposition_id: str | None = None,
    ) -> Verdict:
        """Determines verdict on whether text is supported by reference."""
        # Format inputs as JSON string
        json_str = InputTextAndReferenceContext(text=text, reference=reference).model_dump_json(
            indent=4
        )
        # Several tries to generate verdict
        for i in range(self.num_invalid_output_retries):
            try:
                obj = None
                # Generation of Verdict & Reason using a Reasoning Model
                response: ChatResponse = self.llm.call(
                    json_str,
                    prompt=prompt_determine_verdict_reasoning,
                    return_raw_response=True,
                )
                chat_completion: ChatCompletion = response.raw
                message: ChatCompletionMessage = chat_completion.choices[0].message
                reasoning_chain: str = message.reasoning_content
                reasoning_final_answer: str = message.content
                # Make sure we have both content and reasoning_chain.
                if not reasoning_chain:
                    raise ValueError("No `reasoning_chain` found in response.")
                if not reasoning_final_answer:
                    raise ValueError("No `reasoning_final_answer` found in response.")

                # Content may not be valid JSON. Pass into Auxillary LLM with Structured Output
                obj: SimpleVerdict = self.aux_llm.call(
                    reasoning_final_answer,
                    prompt=prompt_json_structured_output,
                    response_format=SimpleVerdict,
                    validate_response_fn=self.validate_simple_verdict,
                )
                # Convert to Verdict model to include text and reference
                obj: Verdict = Verdict.from_simple_verdict(
                    simple_verdict=obj,
                    text=text,
                    reference=reference,
                    proposition_id=proposition_id or "",
                    reasoning_chain=reasoning_chain,
                    reasoning_final_answer=reasoning_final_answer,
                )
                # Capitalize first letter of each word in the label
                obj.verdict = obj.verdict.title()
                # If verdict is valid, break out of loop
                if self.validate_simple_verdict(obj):
                    break
                else:
                    raise ValueError(f"Invalid Verdict: {obj.verdict}")
            except Exception as ex:
                if self.logger:
                    llm = self.llm.llm if isinstance(self.llm, LLM) else self.llm
                    log_str = (
                        f"Determine Verdict Try {i}. "
                        f"Temperature={llm.temperature}, "
                        f"Top_p={llm.additional_kwargs['top_p']}.\n"
                        f"Input JSON: {json_str}\n"
                    )
                    if obj is not None:
                        log_str += f"Result Object: {obj}\n"
                        if hasattr(obj, "verdict"):
                            log_str += f"Verdict: {obj.verdict}\n"
                    log_str += f"Exception: {ex}\n"
                    self.logger.critical(log_str)
        return obj

    async def _a_determine_reasoning_verdict(
        self,
        text: str,
        reference: str,
        proposition_id: str | None = None,
    ) -> Awaitable[Verdict]:
        """Determines verdict on whether text is supported by reference."""
        # Format inputs as JSON string
        json_str = InputTextAndReferenceContext(text=text, reference=reference).model_dump_json(
            indent=4
        )
        # Several tries to generate verdict
        for i in range(self.num_invalid_output_retries):
            try:
                obj = None
                # Generation of Verdict & Reason using a Reasoning Model
                response: ChatResponse = await self.llm.a_call(
                    json_str,
                    prompt=prompt_determine_verdict_reasoning,
                    return_raw_response=True,
                )
                chat_completion: ChatCompletion = response.raw
                message: ChatCompletionMessage = chat_completion.choices[0].message
                reasoning_chain: str = message.reasoning_content
                reasoning_final_answer: str = message.content
                # Make sure we have both content and reasoning_chain.
                if not reasoning_chain:
                    raise ValueError("No `reasoning_chain` found in response.")
                if not reasoning_final_answer:
                    raise ValueError("No `reasoning_final_answer` found in response.")

                # Content may not be valid JSON. Pass into Auxillary LLM with Structured Output
                obj: SimpleVerdict = await self.aux_llm.a_call(
                    reasoning_final_answer,
                    prompt=prompt_json_structured_output,
                    response_format=SimpleVerdict,
                    validate_response_fn=self.validate_simple_verdict,
                )
                # Convert to Verdict model to include text and reference
                obj: Verdict = Verdict.from_simple_verdict(
                    simple_verdict=obj,
                    text=text,
                    reference=reference,
                    proposition_id=proposition_id or "",
                    reasoning_chain=reasoning_chain,
                    reasoning_final_answer=reasoning_final_answer,
                )
                # Capitalize first letter of each word in the label
                obj.verdict = obj.verdict.title()
                # If verdict is valid, break out of loop
                if self.validate_simple_verdict(obj):
                    break
                else:
                    raise ValueError(f"Invalid Verdict: {obj.verdict}")
            except Exception as ex:
                if self.logger:
                    llm = self.llm.llm if isinstance(self.llm, LLM) else self.llm
                    log_str = (
                        f"Determine Verdict Try {i}. "
                        f"Temperature={llm.temperature}, "
                        f"Top_p={llm.additional_kwargs['top_p']}.\n"
                        f"Input JSON: {json_str}\n"
                    )
                    if obj is not None:
                        log_str += f"Result Object: {obj}\n"
                        if hasattr(obj, "verdict"):
                            log_str += f"Verdict: {obj.verdict}\n"
                    log_str += f"Exception: {ex}\n"
                    self.logger.critical(log_str)
        return obj
