"""Hybrid retrieval: dense + BM25 candidates, merged with reciprocal rank fusion, then reranked."""

from __future__ import annotations

import logging
from typing import Protocol

from jolt.config import Settings
from jolt.schemas import RetrievedChunk
from jolt.store import VectorStore

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Merge ranked lists of ids. Each id scores sum(1 / (k + rank)) over the lists it appears in.

    RRF only uses positions, so it can combine scores on different scales (cosine vs BM25)
    without any tuning. k=60 is the value from the original paper (Cormack et al., 2009).
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class PassthroughReranker:
    """Keeps the fused order. Used when no cross-encoder is available."""

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        return candidates[:top_n]


class CrossEncoderReranker:
    """Scores each (query, passage) pair jointly; slower than embeddings but more precise."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install 'jolt[embeddings]'"
            ) from exc
        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self._model.predict([(query, c.chunk.text) for c in candidates])
        rescored = [
            RetrievedChunk(chunk=c.chunk, score=float(s))
            for c, s in zip(candidates, scores, strict=True)
        ]
        return sorted(rescored, key=lambda c: c.score, reverse=True)[:top_n]


def build_reranker(settings: Settings) -> Reranker:
    if settings.reranker_backend == "cross-encoder":
        return CrossEncoderReranker(settings.reranker_model)
    return PassthroughReranker()


class HybridRetriever:
    def __init__(self, store: VectorStore, reranker: Reranker, retrieve_k: int, top_n: int) -> None:
        self.store = store
        self.reranker = reranker
        self.retrieve_k = retrieve_k
        self.top_n = top_n

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        dense = [i for i, _ in self.store.dense_search(query, self.retrieve_k)]
        sparse = [i for i, _ in self.store.sparse_search(query, self.retrieve_k)]
        fused = reciprocal_rank_fusion([dense, sparse])[: self.retrieve_k]
        candidates = [RetrievedChunk(chunk=self.store.chunks[i], score=score) for i, score in fused]
        logger.debug(
            "query=%r dense=%d sparse=%d fused=%d", query, len(dense), len(sparse), len(fused)
        )
        return self.reranker.rerank(query, candidates, self.top_n)
