"""The agent's steps. Each method takes the current state and returns the fields it changes."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from typing import Any

from jolt import prompts
from jolt.agent.state import AgentState
from jolt.config import Settings
from jolt.llm import LLM, parse_json
from jolt.retrieval import HybridRetriever
from jolt.schemas import Chunk, Citation, RetrievedChunk, TraceStep
from jolt.web_search import WebSearcher

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(\d+)\]")


def _step(node: str, detail: str, started: float) -> list[TraceStep]:
    elapsed = (time.perf_counter() - started) * 1000
    logger.info("%-11s %6.0f ms  %s", node, elapsed, detail)
    return [TraceStep(node=node, detail=detail, duration_ms=round(elapsed, 1))]


def _query_from(raw: str, fallback: str) -> str:
    data = parse_json(raw)
    query = data.get("query") if data else None
    return query.strip() if isinstance(query, str) and query.strip() else fallback


class AgentNodes:
    def __init__(
        self,
        llm: LLM,
        retriever: HybridRetriever,
        web: WebSearcher,
        settings: Settings,
        on_web_results: Callable[[list[Chunk]], None] | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.web = web
        self.settings = settings
        self.on_web_results = on_web_results

    def plan(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        question = state["question"]
        raw = self.llm.chat(prompts.planner(question), max_tokens=96, json_mode=True)
        query = _query_from(raw, fallback=question)
        return {
            "query": query,
            "rewrites": 0,
            "used_web": False,
            "trace": _step("plan", f"search query: {query!r}", t0),
        }

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        docs = self.retriever.retrieve(state["query"])
        return {"documents": docs, "trace": _step("retrieve", f"{len(docs)} chunks", t0)}

    def grade(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        docs = state.get("documents", [])
        if not docs:
            return {
                "sufficient": False,
                "missing": "no documents found",
                "trace": _step("grade", "no documents to grade", t0),
            }
        raw = self.llm.chat(prompts.grader(state["question"], docs), max_tokens=160, json_mode=True)
        verdict = parse_json(raw)
        if verdict is None:
            # Unparseable grade: keep everything and let the answer step decide or abstain.
            logger.warning("Grader returned non-JSON output; keeping all %d chunks", len(docs))
            return {
                "sufficient": True,
                "trace": _step("grade", "unparseable verdict, kept all", t0),
            }
        picked = {int(n) for n in verdict.get("relevant", []) if str(n).isdigit()}
        kept = [d for i, d in enumerate(docs, start=1) if i in picked]
        sufficient = bool(verdict.get("sufficient")) and bool(kept)
        return {
            "documents": kept,
            "sufficient": sufficient,
            "missing": str(verdict.get("missing") or ""),
            "trace": _step("grade", f"kept {len(kept)}/{len(docs)}, sufficient={sufficient}", t0),
        }

    def rewrite(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        previous = state["query"]
        raw = self.llm.chat(
            prompts.rewriter(state["question"], previous, state.get("missing", "")),
            max_tokens=96,
            json_mode=True,
        )
        query = _query_from(raw, fallback=state["question"])
        if query.lower() == previous.lower() and state.get("missing"):
            query = f"{query} {state['missing']}"
        rewrites = state.get("rewrites", 0) + 1
        return {
            "query": query,
            "rewrites": rewrites,
            "trace": _step("rewrite", f"attempt {rewrites}: {query!r}", t0),
        }

    def web_search(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = self.web.search(state["question"], self.settings.web_max_results)
        if results and self.on_web_results:
            self.on_web_results(results)
        web_docs = [RetrievedChunk(chunk=c, score=0.0) for c in results]
        return {
            "documents": [*state.get("documents", []), *web_docs],
            "used_web": True,
            "trace": _step("web_search", f"{len(results)} results", t0),
        }

    def generate(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        docs = state.get("documents", [])
        if not docs:
            return {
                "answer": prompts.ABSTAIN,
                "citations": [],
                "trace": _step("generate", "no sources, abstained", t0),
            }
        answer = self.llm.chat(prompts.answerer(state["question"], docs), max_tokens=700)
        cited = sorted({int(n) for n in _CITATION.findall(answer) if 1 <= int(n) <= len(docs)})
        citations = [
            Citation(
                index=n,
                source=docs[n - 1].chunk.source,
                page=docs[n - 1].chunk.page,
                url=docs[n - 1].chunk.url,
                snippet=docs[n - 1].chunk.text[:240],
            )
            for n in cited
        ]
        return {
            "answer": answer,
            "citations": citations,
            "trace": _step("generate", f"{len(citations)} citations", t0),
        }
