"""Data models shared by the pipeline, the API and the evaluation harness."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A piece of a source document, the unit that gets embedded and retrieved."""

    id: str
    text: str
    source: str
    page: int | None = None
    chunk_index: int = 0
    url: str | None = None


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float


class Citation(BaseModel):
    index: int = Field(description="Number used in the answer, e.g. [1]")
    source: str
    page: int | None = None
    url: str | None = None
    snippet: str


class TraceStep(BaseModel):
    node: str
    detail: str
    duration_ms: float


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    use_web: bool | None = Field(
        default=None, description="Override the server's web-search setting for this request"
    )


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool = Field(description="True when the answer cites at least one source")
    used_web: bool
    rewrites: int
    trace: list[TraceStep]
    latency_ms: float


class IngestResponse(BaseModel):
    files: int
    chunks_added: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_backend: str
    indexed_chunks: int
