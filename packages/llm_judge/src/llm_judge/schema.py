from typing import Any

from pydantic import BaseModel, Field


class SimpleVerdict(BaseModel):
    """Verdict model for judge."""

    verdict: str = ""
    reason: str = ""

    @classmethod
    def class_name(cls) -> str:
        return "SimpleVerdict"

    def __str__(self) -> str:
        return f"Verdict: {self.verdict} | Reason: {self.reason}"

    def __repr__(self) -> str:
        return self.__str__()


class Verdict(BaseModel):
    """Verdict model for judge with additional fields for text and reference."""

    proposition_id: str | None = Field(
        default="", description="Proposition ID associated with the verdict."
    )
    verdict: str | None = Field(default="", description="Verdict result (e.g., Supported).")
    reason: str | None = Field(default="", description="Reason for the verdict.")
    reasoning_chain: str | None = Field(
        default="", description="Reasoning model's reasoning/thinking chain."
    )
    reasoning_final_answer: str | None = Field(
        default="", description="Reasoning model's final answer."
    )
    text: str | None = Field(default="", description="Text associated with the verdict.")
    reference: str | None = Field(
        default=None, description="Reference or citation for the verdict."
    )
    metadata_: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata for additional context.",
        alias="metadata",
    )  # Optional, Used to store any additional info

    @classmethod
    def class_name(cls) -> str:
        return "Verdict"

    @classmethod
    def from_simple_verdict(
        cls, simple_verdict: SimpleVerdict, text: str, reference: str, **kwargs
    ) -> "Verdict":
        return cls(
            proposition_id=kwargs.get("proposition_id", ""),
            verdict=simple_verdict.verdict,
            reason=simple_verdict.reason,
            reasoning_chain=kwargs.get("reasoning_chain", ""),
            reasoning_final_answer=kwargs.get("reasoning_final_answer", ""),
            text=text,
            reference=reference,
        )

    def __str__(self) -> str:
        return f"Verdict: {self.verdict} | Reason: {self.reason}"

    def __repr__(self) -> str:
        return self.__str__()

    def text_and_reference(self) -> str:
        return f"Text: {self.text}\nReference: \n{self.reference}"

    def report(self) -> str:
        report_str = (
            f"Verdict: {self.verdict}\nReason: {self.reason}\n"
            f"Text: {self.text}\nReference: \n{self.reference}"
        )
        if self.reasoning_chain:
            report_str += f"\nReasoning Chain: {self.reasoning_chain}"
        if self.reasoning_final_answer:
            report_str += f"\nReasoning Final Answer: {self.reasoning_final_answer}"
        return report_str


class InputTextAndReferenceContext(BaseModel):
    """Input model for judge."""

    text: str
    reference: str

    @classmethod
    def class_name(cls) -> str:
        return "InputTextAndReferenceContext"

    def __str__(self) -> str:
        return f"Text: {self.text}\nReference: {self.reference}"

    def __repr__(self) -> str:
        return self.__str__()


class InputTextsAndReferenceContexts(BaseModel):
    """Collection of input texts and paired references."""

    texts: list[str]
    references: list[str]
    raw_references: list[Any] | None = None

    @classmethod
    def class_name(cls) -> str:
        return "InputTextsAndReferenceContexts"


class Reasons(BaseModel):
    """Reasons model for judge."""

    reasons: list[str] = []

    @classmethod
    def class_name(cls) -> str:
        return "Reasons"

    def __str__(self) -> str:
        return f"Reasons: {self.reasons}"

    def __repr__(self) -> str:
        return self.__str__()


class ExplanationSummary(BaseModel):
    """Explanation summary model for judge."""

    summary: str = ""

    @classmethod
    def class_name(cls) -> str:
        return "ExplanationSummary"

    def __str__(self) -> str:
        return f"Summary: {self.summary}"

    def __repr__(self) -> str:
        return self.__str__()
