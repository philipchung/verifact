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

    node_id: str | None = Field(default="", description="Node ID associated with the verdict.")
    verdict: str | None = Field(default="", description="Verdict result (e.g., Supported).")
    reason: str | None = Field(default="", description="Reason for the verdict.")
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
            node_id=kwargs.get("node_id", ""),
            verdict=simple_verdict.verdict,
            reason=simple_verdict.reason,
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
        return (
            f"Verdict: {self.verdict}\nReason: {self.reason}\n"
            f"Text: {self.text}\nReference: \n{self.reference}"
        )


class InputTextAndReferenceContext(BaseModel):
    """Input model for judge."""

    node_id: str = ""
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
