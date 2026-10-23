"""Offline stand-ins for tests, CI and smoke runs. Not used unless configured."""

from __future__ import annotations

import json
import math
import re

from jolt.llm import Message
from jolt.prompts import ABSTAIN
from jolt.schemas import Chunk

_SOURCE = re.compile(r"\[(\d+)\] \([^)]*\)\n(.*?)(?=\n\n\[\d+\] \(|\n\nQuestion: |\Z)", re.DOTALL)
_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "do",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 2}


class ExtractiveFakeLLM:
    """A rule-based stand-in for an LLM.

    It recognises which agent step is calling from the system prompt and answers by keyword
    overlap and sentence extraction. It lets the whole graph run end to end without a model,
    which is what the tests and CI need. It is not a measure of answer quality.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def chat(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        system, user = messages[0]["content"], messages[-1]["content"]
        role = system.split(" of a research assistant")[0].removeprefix("You are the ").strip()
        self.calls.append(role)
        if role in ("planner", "query rewriter"):
            question = user.split("\n")[0].removeprefix("Question: ")
            return json.dumps({"query": question})
        question = user.split("Question: ")[-1].split("\n")[0]
        sources = _SOURCE.findall(user)
        keywords = _keywords(question)
        scored = [(len(keywords & _keywords(text)), int(num), text) for num, text in sources]
        threshold = max(1, math.ceil(len(keywords) / 2))
        relevant = [num for score, num, _ in scored if score >= threshold]
        if role == "grader":
            return json.dumps(
                {
                    "relevant": relevant,
                    "sufficient": bool(relevant),
                    "missing": "" if relevant else question,
                }
            )
        # answer writer: return the best-matching sentence from the best source
        best = max(scored, default=None)
        if not best or best[0] < threshold:
            return ABSTAIN
        sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", best[2].strip()) if s.strip()]
        sentence = max(sentences, key=lambda s: len(keywords & _keywords(s)))
        return f"{sentence.strip()} [{best[1]}]"


class StaticWebSearcher:
    """Returns canned web results; records the queries it received."""

    def __init__(self, results: list[Chunk] | None = None) -> None:
        self.results = results or []
        self.queries: list[str] = []

    def search(self, query: str, max_results: int) -> list[Chunk]:
        self.queries.append(query)
        return self.results[:max_results]
