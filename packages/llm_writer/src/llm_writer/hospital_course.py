from collections.abc import Awaitable, Callable, Coroutine
from copy import deepcopy

import pandas as pd
from llama_index.core.bridge.pydantic import ConfigDict, Field
from llama_index.llms.openai import OpenAI
from pydantic_utils.llm_base_model import LLMBaseModel
from rag.components import get_llm, get_semantic_node_parser
from rag.llms import OpenAILike
from rag.node_parser import count_tokens
from rag.schema import TextNode
from tqdm.asyncio import tqdm
from utils import run_jobs

from .prompts import (
    prompt_brief_hospital_course,
    prompt_combine_summaries,
    prompt_compact_brief_hospital_course,
    prompt_refine_brief_hospital_course,
    prompt_summarize_note,
    system_prompt_hospital_physician,
)


def semantic_split_text(text: str, max_chunk_size: int = 1000) -> list[str]:
    """Split text into smaller chunks using semantic text splitter."""
    semantic_parser = get_semantic_node_parser(max_chunk_size=max_chunk_size)
    text_node = TextNode(text=text)
    result_node_list = semantic_parser([text_node])
    return [node.text for node in result_node_list]


class HospitalCourseSummarizer(LLMBaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: OpenAILike | OpenAI | None = Field(
        default=None, description="LLM model to use for atomic claim extraction."
    )
    num_workers: int = Field(default=16, description="Number of notes to summarize in parallel.")
    max_parallel_note_chunks: int = Field(
        default=16, description="Number of note chunks to process in parallel."
    )
    system_prompt_template: Callable = Field(
        default=system_prompt_hospital_physician,
        description="System prompt function. Overrides default LLM system prompt.",
    )
    prompt_summarize_note_template: Callable = Field(
        default=prompt_summarize_note,
        description="Message prompt function for summarizing notes.",
    )
    prompt_combine_summaries_template: Callable = Field(
        default=prompt_combine_summaries,
        description="Message prompt function for combining summaries.",
    )
    prompt_brief_hospital_course_template: Callable = Field(
        default=prompt_brief_hospital_course,
        description="Message prompt function for writing the brief hospital course.",
    )
    prompt_refine_brief_hospital_course_template: Callable = Field(
        default=prompt_refine_brief_hospital_course,
        description="Message prompt function for refining the brief hospital course.",
    )
    prompt_compact_brief_hospital_course_template: Callable = Field(
        default=prompt_compact_brief_hospital_course,
        description="Message prompt function for compacting the brief hospital course.",
    )
    max_hospital_course_size: int = Field(
        default=1000,
        description="Maximum number of tokens allowed in the `brief hospital course`.",
    )
    max_chunk_size: int = Field(
        default=2000,
        description="Maximum number of tokens allowed in a chunk.",
    )
    splitter: Callable = Field(
        default=semantic_split_text,
        description="Function to split text into smaller chunks. "
        "Takes arguments: text, max_chunk_size.",
    )
    tokenizer_method: str | Callable = Field(
        default="llama3",
        description=(
            "Name of tokenizer (e.g. `llama3`, `o200k_base`, or "
            "huggingface with `namespace/name`"
        ),
    )

    @classmethod
    def from_defaults(
        cls,
        llm: OpenAILike | OpenAI | None,
        num_workers: int = 16,
        max_parallel_note_chunks: int = 8,
        system_prompt_template: Callable = system_prompt_hospital_physician,
        prompt_summarize_note_template: Callable = prompt_summarize_note,
        prompt_combine_summaries_template: Callable = prompt_combine_summaries,
        prompt_brief_hospital_course_template: Callable = prompt_brief_hospital_course,
        prompt_refine_brief_hospital_course_template: Callable = prompt_refine_brief_hospital_course,  # noqa: E501
        prompt_compact_brief_hospital_course_template: Callable = prompt_compact_brief_hospital_course,  # noqa: E501
        max_hospital_course_size: int = 1000,
        max_chunk_size: int = 2000,
        splitter: Callable = semantic_split_text,
        tokenizer_method: str | Callable = "llama3",
    ) -> "HospitalCourseSummarizer":
        if llm is None:
            llm = get_llm()
        return cls(
            llm=llm,
            num_workers=num_workers,
            max_parallel_note_chunks=max_parallel_note_chunks,
            system_prompt_template=system_prompt_template,
            prompt_combine_summaries_template=prompt_combine_summaries_template,
            prompt_summarize_note_template=prompt_summarize_note_template,
            prompt_brief_hospital_course_template=prompt_brief_hospital_course_template,
            prompt_refine_brief_hospital_course_template=prompt_refine_brief_hospital_course_template,
            prompt_compact_brief_hospital_course_template=prompt_compact_brief_hospital_course_template,
            max_hospital_course_size=max_hospital_course_size,
            max_chunk_size=max_chunk_size,
            splitter=splitter,
            tokenizer_method=tokenizer_method,
        )

    def __call__(self, df: pd.DataFrame, show_progress: bool = False) -> str:
        """Given a DataFrame of MIMIC-III notes for a single hospital admission,
        generate the "Brief Hospital Course" section of a discharge summary.

        The DataFrame is expected to conform to the MIMIC-III NOTEEVENTS table schema.
        """
        ## Generate summaries for each note in the DataFrame
        summaries = [
            self._summarize_text(row.TEXT)
            for row in tqdm(
                df.itertuples(), total=len(df), desc="Summarizing Notes", disable=not show_progress
            )
        ]
        ## Add Note Metadata Header to each Generated Summary
        summaries_with_metadata: list[str] = [
            self._add_note_metadata(text, row)
            for text, row in zip(summaries, df.itertuples(), strict=False)
        ]
        ## Repack Text Chunks
        repacked_summaries: list[str] = self._repack_text_chunks(
            texts=summaries_with_metadata, max_chunk_size=self.max_chunk_size
        )
        ## Iteratively Write Brief Hospital Course section from all notes
        hospital_course = self._write_hospital_course_from_summaries(repacked_summaries)
        return hospital_course

    async def acall(self, df: pd.DataFrame, show_progress: bool = False) -> Awaitable[str]:
        """Given a DataFrame of MIMIC-III notes for a single hospital admission,
        generate the "Brief Hospital Course" section of a discharge summary.

        The DataFrame is expected to conform to the MIMIC-III NOTEEVENTS table schema.
        """
        ## Generate summaries for each note in the DataFrame
        jobs: list[Coroutine] = [self._a_summarize_text(row.TEXT) for row in df.itertuples()]
        summaries = await run_jobs(
            jobs, workers=self.num_workers, desc="Summarizing Notes", show_progress=show_progress
        )
        ## Add Note Metadata Header to each Generated Summary
        summaries_with_metadata: list[str] = [
            self._add_note_metadata(text, row)
            for text, row in zip(summaries, df.itertuples(), strict=False)
        ]
        ## Repack Text Chunks
        repacked_summaries: list[str] = self._repack_text_chunks(
            texts=summaries_with_metadata, max_chunk_size=self.max_chunk_size
        )
        ## Iteratively Write Brief Hospital Course section from all notes
        hospital_course = await self._a_write_hospital_course_from_summaries(repacked_summaries)
        return hospital_course

    def _add_note_metadata(self, text: str, row: pd.Series) -> str:
        """Add metadata to a note text string."""
        date_str = row.CHARTDATE.strftime("%Y-%m-%d")  # All notes have Date
        time_str = (
            row.CHARTTIME.strftime("%X") if pd.notna(row.CHARTTIME) else "Unknown"
        )  # Not all notes have DateTime
        category = row.CATEGORY  # Note Category
        description = row.DESCRIPTION  # Note Description
        note_text_with_metadata = (
            f"Date: {date_str} | Time: {time_str} |"
            f"Note Category: {category} | Note Description: {description}\n"
            f"{text}"
        )
        return note_text_with_metadata

    def _repack_text_chunks(
        self, texts: list[str], max_chunk_size: int = 1000, delimiter: str = "\n\n"
    ) -> list[str]:
        """Pack text chunks into larger strings of length `max_chunk_size` or smaller.

            This method assumes all texts are smaller than `max_chunk_size`. If a text is larger
        than `max_chunk_size`, it will not be shortened or truncated and will be carried
        as-is as one of the output tet chunks.
        """
        packed_texts: list[str] = []
        running_text_chunk: str = ""
        running_token_count = 0
        for text_chunk in texts:
            text_token_ct = self._count_tokens(text_chunk)
            # If adding the next chunk would exceed the max chunk size, start a new chunk
            if running_token_count + text_token_ct > max_chunk_size:
                packed_texts.append(running_text_chunk)
                running_text_chunk = ""
                running_token_count = 0
            # Add the next chunk to the running chunk
            running_text_chunk += text_chunk + delimiter
            running_token_count += text_token_ct
        # Add the last chunk to the packed texts
        if running_text_chunk:
            packed_texts.append(running_text_chunk)
        return packed_texts

    def _write_hospital_course_from_summaries(self, summaries: list[str]) -> str:
        """Generate the "Brief Hospital Course" section of a discharge summary from
        a list of note summaries from the admission."""
        _summaries = deepcopy(summaries)
        ## Iteratively Write Brief Hospital Course section from all notes
        hosp_course_text: str = ""
        while _summaries:
            next_text = _summaries.pop(0)
            # Incorporate hospital note summaries into the brief hospital course
            if not hosp_course_text:
                hosp_course_text = self._write_brief_hospital_course(next_text)
            else:
                hosp_course_text = self._refine_brief_hospital_course(hosp_course_text, next_text)
            # If the brief hospital course text is too long, compact it
            if self._count_tokens(hosp_course_text) > self.max_hospital_course_size:
                hosp_course_text = self._compact_brief_hospital_course(hosp_course_text)
        return hosp_course_text

    async def _a_write_hospital_course_from_summaries(self, summaries: list[str]) -> Awaitable[str]:
        """Generate the "Brief Hospital Course" section of a discharge summary from
        a list of note summaries from the admission."""
        _summaries = deepcopy(summaries)
        ## Iteratively Write Brief Hospital Course section from all notes
        hosp_course_text: str = ""
        while _summaries:
            next_text = _summaries.pop(0)
            # Incorporate hospital note summaries into the brief hospital course
            if not hosp_course_text:
                hosp_course_text = await self._a_write_brief_hospital_course(next_text)
            else:
                hosp_course_text = await self._a_refine_brief_hospital_course(
                    hosp_course_text, next_text
                )
            # If the brief hospital course text is too long, compact it
            if self._count_tokens(hosp_course_text) > self.max_hospital_course_size:
                hosp_course_text = await self._a_compact_brief_hospital_course(hosp_course_text)
        return hosp_course_text

    def _count_tokens(self, text: str) -> int:
        """Count number of tokens in a text string."""
        return count_tokens(text, tokenizer=self.tokenizer_method)

    def _split_text(self, text: str) -> list[str]:
        """Split text into smaller chunks using semantic text splitter."""
        return self.splitter(text, self.max_chunk_size)

    def _summarize_text(self, text: str) -> str:
        """Summarize text using the LLM model."""
        token_ct = self._count_tokens(text)
        # If text is small enough, summarize it directly
        if token_ct <= self.max_chunk_size:
            return self._generate(text, prompt_fn=self.prompt_summarize_note_template)
        # If text is too large, split it into smaller chunks, summarize the chunks,
        # then combine them into a single summary
        else:
            chunks = self._split_text(text)
            summaries = [
                self._generate(chunk, prompt_fn=self.prompt_summarize_note_template)
                for chunk in chunks
            ]
            summaries_concat = "\n".join(summaries)
            combined_summary = self._generate(
                summaries_concat, prompt_fn=self.prompt_combine_summaries_template
            )
            return combined_summary

    async def _a_summarize_text(self, text: str) -> Awaitable[str]:
        """Summarize text using the LLM model."""
        token_ct = self._count_tokens(text)
        # If text is small enough, summarize it directly
        if token_ct <= self.max_chunk_size:
            return await self._a_generate(text, prompt_fn=self.prompt_summarize_note_template)
        # If text is too large, split it into smaller chunks, summarize the chunks,
        # then combine them into a single summary
        else:
            chunks = self._split_text(text)
            jobs = [
                self._a_generate(chunk, prompt_fn=self.prompt_summarize_note_template)
                for chunk in chunks
            ]
            summaries = await run_jobs(jobs, workers=self.max_parallel_note_chunks)
            summaries_concat = "\n".join(summaries)
            combined_summary = self._generate(
                summaries_concat, prompt_fn=self.prompt_combine_summaries_template
            )
            return combined_summary

    def _write_brief_hospital_course(self, text: str) -> str:
        """Write the "Brief Hospital Course" section of a discharge summary."""
        return self._generate(text, prompt_fn=self.prompt_brief_hospital_course_template)

    async def _a_write_brief_hospital_course(self, text: str) -> Awaitable[str]:
        """Write the "Brief Hospital Course" section of a discharge summary."""
        return await self._a_generate(text, prompt_fn=self.prompt_brief_hospital_course_template)

    def _refine_brief_hospital_course(self, brief_hospital_course: str, new_text: str) -> str:
        """Refine the "Brief Hospital Course" section of a discharge summary."""
        return self._generate(
            brief_hospital_course,
            new_text,
            prompt_fn=self.prompt_refine_brief_hospital_course_template,
        )

    async def _a_refine_brief_hospital_course(
        self, brief_hospital_course: str, new_text: str
    ) -> Awaitable[str]:
        """Refine the "Brief Hospital Course" section of a discharge summary."""
        return await self._a_generate(
            brief_hospital_course,
            new_text,
            prompt_fn=self.prompt_refine_brief_hospital_course_template,
        )

    def _compact_brief_hospital_course(self, brief_hospital_course: str) -> str:
        """Compact the "Brief Hospital Course" section to make it more concise."""
        return self._generate(
            brief_hospital_course, prompt_fn=self.prompt_compact_brief_hospital_course_template
        )

    async def _a_compact_brief_hospital_course(self, brief_hospital_course: str) -> Awaitable[str]:
        """Compact the "Brief Hospital Course" section to make it more concise."""
        return await self._a_generate(
            brief_hospital_course, prompt_fn=self.prompt_compact_brief_hospital_course_template
        )
