"""Web search fallback, used when the local knowledge base cannot answer a question."""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

from jolt.schemas import Chunk

logger = logging.getLogger(__name__)


class WebSearcher(Protocol):
    def search(self, query: str, max_results: int) -> list[Chunk]: ...


class DuckDuckGoSearcher:
    """Keyless web search through the ``ddgs`` package. Failures return no results, never raise."""

    def search(self, query: str, max_results: int) -> list[Chunk]:
        try:
            from ddgs import DDGS

            hits = DDGS().text(query, max_results=max_results)
        except Exception as exc:  # network errors, rate limits, parser changes
            logger.warning("Web search failed for %r: %s", query, exc)
            return []
        chunks: list[Chunk] = []
        for i, hit in enumerate(hits or []):
            body = (hit.get("body") or "").strip()
            url = hit.get("href") or hit.get("url")
            if not body:
                continue
            chunks.append(
                Chunk(
                    id=hashlib.sha1(f"{url}|{body}".encode()).hexdigest()[:16],
                    text=body[:1500],
                    source=hit.get("title") or url or "web",
                    url=url,
                    chunk_index=i,
                )
            )
        return chunks


class NullSearcher:
    def search(self, query: str, max_results: int) -> list[Chunk]:
        return []
