import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Self

from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_utils import LLMBaseModel
from rag.llms.openai_like import OpenAILike
from tqdm.asyncio import tqdm
from utils import LazyFileLogger, run_jobs

from proposition_validity.prompts import (
    prompt_contains_atomic_claims,
    prompt_contains_imperative_statement,
    prompt_contains_incomplete_statement,
    prompt_contains_interrogative_statement,
    prompt_contains_proposition,
    prompt_contains_vague_statement,
    system_prompt_judge,
)


class ClaimValidityResult(BaseModel):
    contains_atomic_claim: bool | None = None
    contains_proposition: bool | None = None
    contains_imperative_statement: bool | None = None
    contains_interrogative_statement: bool | None = None
    contains_incomplete_statement: bool | None = None
    contains_vague_statement: bool | None = None


class ContainsAtomicClaim(BaseModel):
    contains_atomic_claim: bool


class ContainsProposition(BaseModel):
    contains_proposition: bool


class ContainsImperativeStatement(BaseModel):
    contains_imperative_statement: bool


class ContainsInterrogativeStatement(BaseModel):
    contains_interrogative_statement: bool


class ContainsIncompleteStatement(BaseModel):
    contains_incomplete_statement: bool


class ContainsVagueStatement(BaseModel):
    contains_vague_statement: bool


class ClaimValidityJudge(LLMBaseModel):
    """LLM-as-a-Judge which makes a judgement on whether a claim is valid."""

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
    detect_claims_template: Callable = Field(
        default=prompt_contains_atomic_claims,
        description="Message prompt function to determine if text contains a claim.",
    )
    detect_proposition_template: Callable = Field(
        default=prompt_contains_proposition,
        description="Message prompt function to determine if text contains a proposition.",
    )
    detect_imperative_statement_template: Callable = Field(
        default=prompt_contains_imperative_statement,
        description="Message prompt function to determine if text contains an "
        "imperative statement.",
    )
    detect_interrogative_statement_template: Callable = Field(
        default=prompt_contains_interrogative_statement,
        description="Message prompt function to determine if text contains an "
        "interrogative statement.",
    )
    detect_incomplete_statement_template: Callable = Field(
        default=prompt_contains_incomplete_statement,
        description="Message prompt function to determine if text contains an "
        "incomplete statement.",
    )
    detect_vague_statement_template: Callable = Field(
        default=prompt_contains_vague_statement,
        description="Message prompt function to determine if text contains a " "vague statement.",
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
        detect_claims_template: Callable = prompt_contains_atomic_claims,
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
            detect_claims_template=detect_claims_template,
            num_workers=num_workers,
            num_invalid_output_retries=num_invalid_output_retries,
            logger=logger,
        )

    def setup_logger(
        self, level: str = logging.DEBUG, log_file: str | Path | None = None
    ) -> LazyFileLogger:
        log_file = log_file or self.log_filepath
        self.logger = LazyFileLogger(name=__name__, level=level, log_file=log_file)
        return self.logger

    def __call__(
        self,
        texts: list[str],
        **kwargs,
    ) -> list[ClaimValidityResult]:
        return self.evaluate(texts=texts, **kwargs)

    async def acall(
        self,
        texts: list[str],
        **kwargs,
    ) -> Awaitable[list[ClaimValidityResult]]:
        return await self.a_evaluate(texts=texts, **kwargs)

    def evaluate(
        self,
        texts: list[str],
        show_progress: bool = False,
        desc="Evaluating Text",
        **kwargs,
    ) -> list[ClaimValidityResult]:
        """Evaluate each text using all evaluation prompts."""
        results: list[ClaimValidityResult] = []
        for text in tqdm(texts, disable=not show_progress, desc=desc):
            result = ClaimValidityResult(
                contains_atomic_claim=self._determine_contains_atomic_claim(text=text),
                contains_proposition=self._determine_contains_proposition(text=text),
                contains_imperative_statement=self._determine_contains_imperative_statement(
                    text=text
                ),
                contains_interrogative_statement=self._determine_contains_interrogative_statement(
                    text=text
                ),
                contains_incomplete_statement=self._determine_contains_incomplete_statement(
                    text=text
                ),
                contains_vague_statement=self._determine_contains_vague_statement(text=text),
            )
            results.append(result)
        return results

    async def a_evaluate(
        self,
        texts: list[str],
        show_progress: bool = False,
        desc: str = "Evaluating Text",
        **kwargs,
    ) -> Awaitable[list[ClaimValidityResult]]:
        """Evaluate each text using all evaluation prompts."""

        async def job_fn(text: str) -> ClaimValidityResult:
            return ClaimValidityResult(
                contains_atomic_claim=await self._a_determine_contains_atomic_claim(text=text),
                contains_proposition=await self._a_determine_contains_proposition(text=text),
                contains_imperative_statement=await self._a_determine_contains_imperative_statement(
                    text=text
                ),
                contains_interrogative_statement=await self._a_determine_contains_interrogative_statement(  # noqa: E501
                    text=text
                ),
                contains_incomplete_statement=await self._a_determine_contains_incomplete_statement(
                    text=text
                ),
                contains_vague_statement=await self._a_determine_contains_vague_statement(
                    text=text
                ),
            )

        jobs = [job_fn(text) for text in texts]
        results: list[ClaimValidityResult] = await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc=desc,
        )
        return results

    def _determine_contains_atomic_claim(self, text: str) -> bool:
        obj = self._generate(
            text, prompt=self.detect_claims_template, response_format=ContainsAtomicClaim
        )
        contains_atomic_claim: bool = obj.contains_atomic_claim
        return contains_atomic_claim

    async def _a_determine_contains_atomic_claim(self, text: str) -> Awaitable[bool]:
        obj = await self._a_generate(
            text, prompt=self.detect_claims_template, response_format=ContainsAtomicClaim
        )
        contains_atomic_claim: bool = obj.contains_atomic_claim
        return contains_atomic_claim

    def _determine_contains_proposition(self, text: str) -> bool:
        obj = self._generate(
            text, prompt=self.detect_proposition_template, response_format=ContainsProposition
        )
        contains_proposition: bool = obj.contains_proposition
        return contains_proposition

    async def _a_determine_contains_proposition(self, text: str) -> Awaitable[bool]:
        obj = await self._a_generate(
            text, prompt=self.detect_proposition_template, response_format=ContainsProposition
        )
        contains_proposition: bool = obj.contains_proposition
        return contains_proposition

    def _determine_contains_imperative_statement(self, text: str) -> bool:
        obj = self._generate(
            text,
            prompt=self.detect_imperative_statement_template,
            response_format=ContainsImperativeStatement,
        )
        contains_imperative_statement: bool = obj.contains_imperative_statement
        return contains_imperative_statement

    async def _a_determine_contains_imperative_statement(self, text: str) -> Awaitable[bool]:
        obj = await self._a_generate(
            text,
            prompt=self.detect_imperative_statement_template,
            response_format=ContainsImperativeStatement,
        )
        contains_imperative_statement: bool = obj.contains_imperative_statement
        return contains_imperative_statement

    def _determine_contains_interrogative_statement(self, text: str) -> bool:
        obj = self._generate(
            text,
            prompt=self.detect_interrogative_statement_template,
            response_format=ContainsInterrogativeStatement,
        )
        contains_interrogative_statement: bool = obj.contains_interrogative_statement
        return contains_interrogative_statement

    async def _a_determine_contains_interrogative_statement(self, text: str) -> Awaitable[bool]:
        obj = await self._a_generate(
            text,
            prompt=self.detect_interrogative_statement_template,
            response_format=ContainsInterrogativeStatement,
        )
        contains_interrogative_statement: bool = obj.contains_interrogative_statement
        return contains_interrogative_statement

    def _determine_contains_incomplete_statement(self, text: str) -> bool:
        obj = self._generate(
            text,
            prompt=self.detect_incomplete_statement_template,
            response_format=ContainsIncompleteStatement,
        )
        contains_incomplete_statement: bool = obj.contains_incomplete_statement
        return contains_incomplete_statement

    async def _a_determine_contains_incomplete_statement(self, text: str) -> Awaitable[bool]:
        obj = await self._a_generate(
            text,
            prompt=self.detect_incomplete_statement_template,
            response_format=ContainsIncompleteStatement,
        )
        contains_incomplete_statement: bool = obj.contains_incomplete_statement
        return contains_incomplete_statement

    def _determine_contains_vague_statement(self, text: str) -> bool:
        obj = self._generate(
            text,
            prompt=self.detect_vague_statement_template,
            response_format=ContainsVagueStatement,
        )
        contains_vague_statement: bool = obj.contains_vague_statement
        return contains_vague_statement

    async def _a_determine_contains_vague_statement(self, text: str) -> Awaitable[bool]:
        obj = await self._a_generate(
            text,
            prompt=self.detect_vague_statement_template,
            response_format=ContainsVagueStatement,
        )
        contains_vague_statement: bool = obj.contains_vague_statement
        return contains_vague_statement
