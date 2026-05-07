from __future__ import annotations

from pydantic import BaseModel, Field


class Options(BaseModel):
    A: str = ""
    B: str = ""
    C: str = ""
    D: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"A": self.A, "B": self.B, "C": self.C, "D": self.D}


class GenerateRequest(BaseModel):
    article: str = Field(..., min_length=20)
    question: str | None = None
    options: Options | None = None
    question_count: int = Field(5, ge=5, le=10)


class VerifyRequest(BaseModel):
    article: str = Field(..., min_length=20)
    question: str = Field(..., min_length=3)
    options: Options
    selected_option: str = Field(..., pattern="^[ABCDabcd]$")
    correct_option: str | None = Field(None, pattern="^[ABCDabcd]$")


class Distractor(BaseModel):
    text: str
    score: float


class GeneratedQuestion(BaseModel):
    question: str
    options: dict[str, str]
    predicted_correct_option: str
    predicted_answer_text: str
    distractors: list[Distractor]
    hints: list[str]
    confidence: float
    latency_ms: float
    ai_generated: bool


class GenerateResponse(BaseModel):
    questions: list[GeneratedQuestion]
    latency_ms: float
    ai_generated: bool


class VerifyResponse(BaseModel):
    predicted_option: str
    selected_option: str
    is_correct: bool
    confidence: float
    option_scores: dict[str, float]
    explanation: str
    latency_ms: float
