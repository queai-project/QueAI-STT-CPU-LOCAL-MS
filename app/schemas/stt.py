from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TranscriptionOptions(BaseModel):
    model: str | None = None
    language: str | None = None
    task: Literal["transcribe", "translate"] = "transcribe"
    beam_size: int = Field(default=3, ge=1, le=5)


class PrepareModelRequest(BaseModel):
    model: str


class JobSubmissionResponse(BaseModel):
    job_id: str
    status: str


class SegmentResponse(BaseModel):
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    kind: Literal["audio", "video"] | str
    model: str | None = None
    language: str | None = None
    task: str | None = None
    progress: float = 0.0
    message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    original_filename: str | None = None
    duration_seconds: float | None = None
    text: str | None = None
    result_ready: bool = False
    segments: list[SegmentResponse] = []