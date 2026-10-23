"""Prompt templates for each agent step. Kept in one place so they can be reviewed and versioned."""

from __future__ import annotations

from jolt.llm import Message
from jolt.schemas import RetrievedChunk

ABSTAIN = "I don't know based on the available sources."


def format_sources(docs: list[RetrievedChunk], max_chars: int = 900) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        c = doc.chunk
        where = c.url or (f"{c.source}, p.{c.page}" if c.page else c.source)
        lines.append(f"[{i}] ({where})\n{c.text[:max_chars]}")
    return "\n\n".join(lines)


def planner(question: str) -> list[Message]:
    return [
        {
            "role": "system",
            "content": (
                "You are the planner of a research assistant. Turn the user's question into one "
                "focused search query for a document index: keep names, numbers and technical "
                "terms, drop filler words.\n"
                'Reply with JSON only: {"query": "<search query>"}'
            ),
        },
        {"role": "user", "content": question},
    ]


def grader(question: str, docs: list[RetrievedChunk]) -> list[Message]:
    return [
        {
            "role": "system",
            "content": (
                "You are the grader of a research assistant. Decide which numbered sources are "
                "relevant to the question, and whether together they contain enough information "
                "to answer it.\n"
                "Reply with JSON only: "
                '{"relevant": [<source numbers>], "sufficient": true|false, '
                '"missing": "<what information is still needed, or empty>"}'
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nSources:\n{format_sources(docs, 600)}",
        },
    ]


def rewriter(question: str, previous_query: str, missing: str) -> list[Message]:
    return [
        {
            "role": "system",
            "content": (
                "You are the query rewriter of a research assistant. The last search did not "
                "find enough information. Write a different search query: use synonyms, expand "
                "acronyms, or target the missing information.\n"
                'Reply with JSON only: {"query": "<new search query>"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\nPrevious query: {previous_query}\n"
                f"Missing information: {missing or 'unspecified'}"
            ),
        },
    ]


def answerer(question: str, docs: list[RetrievedChunk]) -> list[Message]:
    return [
        {
            "role": "system",
            "content": (
                "You are the answer writer of a research assistant. Answer the question using "
                "only the numbered sources. Cite every claim with its source number in square "
                "brackets, e.g. [1] or [2][3]. Be concise. If the sources do not contain the "
                f'answer, reply exactly: "{ABSTAIN}"'
            ),
        },
        {"role": "user", "content": f"Sources:\n{format_sources(docs)}\n\nQuestion: {question}"},
    ]
