"""Builds every component from settings and exposes the two operations: ask and ingest."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from jolt.agent import build_graph
from jolt.agent.nodes import AgentNodes
from jolt.config import Settings
from jolt.embeddings import Embedder, build_embedder
from jolt.ingest import chunk_file, iter_files
from jolt.llm import LLM, build_llm
from jolt.retrieval import HybridRetriever, Reranker, build_reranker
from jolt.schemas import AskResponse, Chunk, IngestResponse
from jolt.store import VectorStore
from jolt.web_search import DuckDuckGoSearcher, NullSearcher, WebSearcher

logger = logging.getLogger(__name__)


class JoltService:
    def __init__(
        self,
        settings: Settings,
        *,
        llm: LLM,
        embedder: Embedder,
        reranker: Reranker,
        web: WebSearcher,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.store = store if store is not None else VectorStore.load(settings.index_dir, embedder)
        self.retriever = HybridRetriever(
            self.store, reranker, settings.retrieve_k, settings.rerank_top_n
        )
        self._write_lock = threading.Lock()
        nodes = AgentNodes(
            llm,
            self.retriever,
            web,
            settings,
            on_web_results=self._learn if settings.learn_from_web else None,
        )
        self.graph = build_graph(nodes)

    @classmethod
    def from_settings(cls, settings: Settings) -> JoltService:
        web: WebSearcher = DuckDuckGoSearcher() if settings.web_search_enabled else NullSearcher()
        return cls(
            settings,
            llm=build_llm(settings),
            embedder=build_embedder(settings),
            reranker=build_reranker(settings),
            web=web,
        )

    def ask(self, question: str, use_web: bool | None = None) -> AskResponse:
        started = time.perf_counter()
        web_allowed = self.settings.web_search_enabled if use_web is None else use_web
        state: dict[str, Any] = self.graph.invoke(
            {"question": question.strip(), "web_allowed": web_allowed, "trace": []},
            config={"recursion_limit": 12 + 3 * self.settings.max_rewrites},
        )
        citations = state.get("citations", [])
        return AskResponse(
            answer=state.get("answer", ""),
            citations=citations,
            grounded=bool(citations),
            used_web=state.get("used_web", False),
            rewrites=state.get("rewrites", 0),
            trace=state.get("trace", []),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
        )

    def ingest(self, paths: list[Path], rebuild: bool = False) -> IngestResponse:
        files = list(iter_files(paths))
        chunks: list[Chunk] = []
        for path in files:
            root = next((p for p in paths if p.is_dir() and path.is_relative_to(p)), None)
            chunks.extend(
                chunk_file(path, self.settings.chunk_size, self.settings.chunk_overlap, root)
            )
        with self._write_lock:
            if rebuild:
                self.store.clear()
            added = self.store.add(chunks)
            self.store.save(self.settings.index_dir)
        logger.info("Ingested %d files, %d new chunks", len(files), added)
        return IngestResponse(files=len(files), chunks_added=added, total_chunks=len(self.store))

    def _learn(self, chunks: list[Chunk]) -> None:
        with self._write_lock:
            if self.store.add(chunks):
                self.store.save(self.settings.index_dir)

    def graph_mermaid(self) -> str:
        return str(self.graph.get_graph().draw_mermaid())
