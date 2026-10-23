from __future__ import annotations

from pathlib import Path

import pytest

from jolt.config import Settings
from jolt.embeddings import HashingEmbedder
from jolt.retrieval import PassthroughReranker
from jolt.service import JoltService
from jolt.testing import ExtractiveFakeLLM, StaticWebSearcher

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "data" / "sample"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        llm_backend="fake",
        embedding_backend="hashing",
        reranker_backend="none",
        index_dir=tmp_path / "index",
        upload_dir=tmp_path / "uploads",
        web_search_enabled=False,
        chunk_size=500,
        chunk_overlap=80,
    )


def make_service(settings: Settings, web: StaticWebSearcher | None = None) -> JoltService:
    return JoltService(
        settings,
        llm=ExtractiveFakeLLM(),
        embedder=HashingEmbedder(),
        reranker=PassthroughReranker(),
        web=web or StaticWebSearcher(),
    )


@pytest.fixture
def service(settings: Settings) -> JoltService:
    svc = make_service(settings)
    svc.ingest([SAMPLE_DIR])
    return svc
