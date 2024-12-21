import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Self

from llama_index.core.bridge.pydantic import Field
from llama_index.llms.openai import OpenAI
from pydantic_utils import LLMBaseModel
from rag.llms.openai_like import OpenAILike
from tqdm.asyncio import tqdm
from utils import LazyFileLogger, run_jobs

from llm_judge.prompts import prompt_determine_verdict_fewshot, system_prompt_judge
from llm_judge.schema import InputTextAndReferenceContext, SimpleVerdict, Verdict
from llm_judge.score_report import ScoreReport

SUPPORTED_LABEL = "Supported"
NOT_SUPPORTED_LABEL = "Not Supported"
NOT_ADDRESSED_LABEL = "Not Addressed"
VERDICT_LABELS = [SUPPORTED_LABEL, NOT_SUPPORTED_LABEL, NOT_ADDRESSED_LABEL]


class Judge(LLMBaseModel):
    """LLM-as-a-Judge which evaluates input text based on a reference context."""

    llm: OpenAILike | OpenAI = Field(description="LLM model to use for generation.")
    num_workers: int = Field(
        default=32, description="Number of async LLM generations to run in parallel."
    )
    num_invalid_output_retries: int = Field(
        default=10,
        description="Number of attempts to use LLM to correct output.",
    )
    system_prompt_template: Callable = Field(
        default=system_prompt_judge,
        description="System prompt function for judge. Overrides default LLM system prompt.",
    )
    determine_verdict_template: Callable = Field(
        default=prompt_determine_verdict_fewshot,
        description=(
            "Message prompt function for determining a verdict on whether text is "
            "supported by reference. "
        ),
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
        system_prompt_template: Callable = system_prompt_judge,
        determine_verdict_template: Callable = prompt_determine_verdict_fewshot,
        num_workers: int = 16,
        num_invalid_output_retries: int = 3,
        log_filepath: str | Path | None = None,
        log_level: str = logging.DEBUG,
    ) -> Self:
        from utils import load_environment

        load_environment()
        logger = LazyFileLogger(name=__name__, level=log_level, log_file=log_filepath)
        return cls(
            llm=llm,
            system_prompt_template=system_prompt_template,
            determine_verdict_template=determine_verdict_template,
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
        node_ids: list[str] | None = None,
        **kwargs,
    ) -> ScoreReport:
        return self.evaluate(texts=texts, references=references, node_ids=node_ids, **kwargs)

    async def acall(
        self,
        texts: list[str],
        references: list[str],
        node_ids: list[str] | None = None,
        **kwargs,
    ) -> Awaitable[ScoreReport]:
        return await self.a_evaluate(
            texts=texts, references=references, node_ids=node_ids, **kwargs
        )

    def evaluate(
        self,
        texts: list[str],
        references: list[str],
        node_ids: list[str] | None = None,
        include_explanations: bool = True,
        show_progress: bool = False,
        desc="Generating Verdicts",
        **kwargs,
    ) -> ScoreReport:
        """Evaluate each text against its corresponding reference."""
        node_ids = [None] * len(texts) if node_ids is None else node_ids
        verdicts = [
            self._determine_verdict(text=text, reference=reference, node_id=node_id)
            for text, reference, node_id in tqdm(
                zip(texts, references, node_ids, strict=False),
                show_progress=show_progress,
                desc=desc,
            )
        ]
        return ScoreReport.from_defaults(
            verdicts=verdicts,
            include_explanations=include_explanations,
            llm=self.llm,
            num_workers=self.num_workers,
            num_invalid_output_retries=self.num_invalid_output_retries,
        )

    async def a_evaluate(
        self,
        texts: list[str],
        references: list[str],
        node_ids: list[str] | None = None,
        include_explanations: bool = True,
        show_progress: bool = False,
        desc: str = "Generating Verdicts",
        **kwargs,
    ) -> Awaitable[ScoreReport]:
        """Evaluate each text against its corresponding reference."""
        node_ids = [None] * len(texts) if node_ids is None else node_ids
        jobs = [
            self._a_determine_verdict(text=text, reference=reference, node_id=node_id)
            for text, reference, node_id in zip(texts, references, node_ids, strict=False)
        ]
        verdicts = await run_jobs(
            jobs, workers=self.num_workers, show_progress=show_progress, desc=desc
        )
        return await ScoreReport.a_from_defaults(
            verdicts=verdicts,
            include_explanations=include_explanations,
            llm=self.llm,
            num_workers=self.num_workers,
            num_invalid_output_retries=self.num_invalid_output_retries,
        )

    def _determine_verdict(self, text: str, reference: str, node_id: str | None = None) -> Verdict:
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
                obj: SimpleVerdict = self._generate(
                    json_str,
                    prompt_fn=self.determine_verdict_template,
                    pydantic_model=SimpleVerdict,
                )
                # Convert to Verdict model to include text and reference
                obj: Verdict = Verdict.from_simple_verdict(
                    simple_verdict=obj, text=text, reference=reference
                )
                # Capitalize first letter of each word in the label
                obj.verdict = obj.verdict.title()
                # If verdict is valid, break out of loop
                if obj.verdict in VERDICT_LABELS:
                    break
                else:
                    raise ValueError(f"Invalid Verdict: {obj.verdict}")
            except Exception as ex:
                # Construct the Log String
                log_str = (
                    f"Determine Verdict Try {i}. "
                    f"Temperature={self.llm.temperature}, "
                    f"Top_p={self.llm.additional_kwargs['top_p']}.\n"
                    f"Input JSON: {json_str}\n"
                )
                if obj is not None:
                    log_str += f"Result Object: {obj}\n"
                    if hasattr(obj, "verdict"):
                        log_str += f"Verdict: {obj.verdict}\n"
                log_str += f"Exception: {ex}\n"
                self.logger.critical(log_str)

                # If invalid verdict, usually because LLM generation gets stuck repeating a token
                # so we increase temperature and try again.  Cap max temperature at 1.0.
                # Set top_p to 1.0 for more token diversity.
                self.llm.temperature += 0.1
                if self.llm.temperature > 1.0:
                    self.llm.temperature = 1.0
                self.llm.additional_kwargs |= {"top_p": 1.0}
        if node_id is not None:
            obj.node_id = node_id
        return obj

    async def _a_determine_verdict(
        self, text: str, reference: str, node_id: str | None = None
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
                obj: SimpleVerdict = await self._a_generate(
                    json_str,
                    prompt_fn=self.determine_verdict_template,
                    pydantic_model=SimpleVerdict,
                )
                # Convert to Verdict model to include text and reference
                obj: Verdict = Verdict.from_simple_verdict(
                    simple_verdict=obj, text=text, reference=reference
                )
                # Capitalize first letter of each word in the label
                obj.verdict = obj.verdict.title()
                # If verdict is valid, break out of loop
                if obj.verdict in VERDICT_LABELS:
                    break
                else:
                    raise ValueError(f"Invalid Verdict: {obj.verdict}")
            except Exception as ex:
                # Construct the Log String
                log_str = (
                    f"Determine Verdict Try {i}. "
                    f"Temperature={self.llm.temperature}, "
                    f"Top_p={self.llm.additional_kwargs['top_p']}.\n"
                    f"Input JSON: {json_str}\n"
                )
                if obj is not None:
                    log_str += f"Result Object: {obj}\n"
                    if hasattr(obj, "verdict"):
                        log_str += f"Verdict: {obj.verdict}\n"
                log_str += f"Exception: {ex}\n"
                self.logger.critical(log_str)

                # If invalid verdict, usually because LLM generation gets stuck repeating a token
                # so we increase temperature and try again.  Cap max temperature at 1.0.
                # Set top_p to 1.0 for more token diversity.
                self.llm.temperature += 0.1
                if self.llm.temperature > 1.0:
                    self.llm.temperature = 1.0
                self.llm.additional_kwargs |= {"top_p": 1.0}
        if node_id is not None:
            obj.node_id = node_id
        return obj
