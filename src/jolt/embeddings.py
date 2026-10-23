"""Text embedding backends. All return L2-normalised float32 vectors (cosine = dot product)."""

from __future__ import annotations

import itertools
import re
import zlib
from typing import Protocol

import numpy as np

from jolt.config import Settings

_TOKEN = re.compile(r"[a-z0-9]+")


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise RuntimeError(
                "sentence-transformers is not installed. Run: pip install 'jolt[embeddings]'"
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, batch_size=32, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedder:
    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key.get_secret_value(),
        )
        self.name = settings.embedding_model
        self.dim = len(
            self._client.embeddings.create(model=self.name, input=["probe"]).data[0].embedding
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        data = self._client.embeddings.create(model=self.name, input=texts).data
        return _normalise(np.array([d.embedding for d in data]))


class HashingEmbedder:
    """Deterministic bag-of-words embedder with no model download.

    Used in tests and CI. It captures word overlap only, not meaning, so use a real
    embedding model for anything you want to measure.
    """

    def __init__(self, dim: int = 512) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = _TOKEN.findall(text.lower())
            for token in tokens + [a + " " + b for a, b in itertools.pairwise(tokens)]:
                h = zlib.crc32(token.encode())
                out[row, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return _normalise(out)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(settings.embedding_model)
    if settings.embedding_backend == "openai":
        return OpenAIEmbedder(settings)
    return HashingEmbedder()
