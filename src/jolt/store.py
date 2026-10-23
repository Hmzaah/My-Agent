"""Persistent chunk store with a dense (FAISS) index and a sparse (BM25) index."""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from jolt.embeddings import Embedder
from jolt.schemas import Chunk

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\w+")
STORE_VERSION = 1


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class VectorStore:
    """Holds chunks plus two search indexes over them.

    Dense search finds passages with similar meaning; BM25 finds exact terms (names, part
    numbers, acronyms) that embeddings often blur. The retriever fuses both.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._ids: set[str] = set()
        # inner product on unit vectors = cosine similarity
        self._index: faiss.Index = faiss.IndexFlatIP(embedder.dim)
        self._bm25: BM25Okapi | None = None
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        with self._lock:
            self.chunks = []
            self._ids = set()
            self._index = faiss.IndexFlatIP(self.embedder.dim)
            self._bm25 = None

    def add(self, chunks: list[Chunk]) -> int:
        """Embed and index new chunks, skipping ones already present. Returns the count added."""
        with self._lock:
            new = [c for c in chunks if c.id not in self._ids]
            if not new:
                return 0
            vectors = self.embedder.embed([c.text for c in new])
            self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
            self.chunks.extend(new)
            self._ids.update(c.id for c in new)
            self._bm25 = None  # rebuilt lazily on the next sparse search
            return len(new)

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        with self._lock:
            if not self.chunks:
                return []
            vector = self.embedder.embed([query])
            scores, ids = self._index.search(vector, min(k, len(self.chunks)))
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i != -1]

    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        with self._lock:
            if not self.chunks:
                return []
            if self._bm25 is None:
                self._bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
            scores = self._bm25.get_scores(tokenize(query))
            order = np.argsort(scores)[::-1][:k]
            return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    # --- persistence --------------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        with self._lock:
            directory.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(directory / "index.faiss"))
            with (directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
                for chunk in self.chunks:
                    f.write(chunk.model_dump_json() + "\n")
            meta = {
                "version": STORE_VERSION,
                "embedder": self.embedder.name,
                "dim": self.embedder.dim,
                "chunks": len(self.chunks),
            }
            (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedder: Embedder) -> VectorStore:
        store = cls(embedder)
        meta_path = directory / "meta.json"
        if not meta_path.exists():
            return store
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("embedder") != embedder.name or meta.get("dim") != embedder.dim:
            raise ValueError(
                f"Index at {directory} was built with '{meta.get('embedder')}' "
                f"(dim {meta.get('dim')}) but the current embedder is '{embedder.name}' "
                f"(dim {embedder.dim}). Re-run `jolt ingest --rebuild`."
            )
        store._index = faiss.read_index(str(directory / "index.faiss"))
        with (directory / "chunks.jsonl").open(encoding="utf-8") as f:
            store.chunks = [Chunk.model_validate_json(line) for line in f if line.strip()]
        store._ids = {c.id for c in store.chunks}
        if store._index.ntotal != len(store.chunks):
            raise ValueError(f"Corrupt index at {directory}: vector and chunk counts differ")
        logger.info("Loaded %d chunks from %s", len(store.chunks), directory)
        return store
