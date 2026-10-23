"""The state that flows through the agent graph. Each node reads it and returns a partial update."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from jolt.schemas import Citation, RetrievedChunk, TraceStep


class AgentState(TypedDict, total=False):
    question: str
    web_allowed: bool
    query: str  # current search query (rewritten on retries)
    rewrites: int
    documents: list[RetrievedChunk]
    sufficient: bool
    missing: str
    used_web: bool
    answer: str
    citations: list[Citation]
    # operator.add makes LangGraph append each node's steps instead of overwriting them
    trace: Annotated[list[TraceStep], operator.add]
