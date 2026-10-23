from pathlib import Path

import pytest

from jolt.embeddings import HashingEmbedder
from jolt.retrieval import HybridRetriever, PassthroughReranker, reciprocal_rank_fusion
from jolt.schemas import Chunk
from jolt.store import VectorStore


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(id=f"c{i}", text=text, source=f"doc{i}.md")


@pytest.fixture
def store() -> VectorStore:
    s = VectorStore(HashingEmbedder())
    s.add(
        [
            _chunk(0, "The cyclic prefix removes inter-symbol interference in OFDM."),
            _chunk(1, "RRAM stores bits as the resistance of a hafnium oxide layer."),
            _chunk(2, "BM25 ranks documents by exact term overlap with the query."),
            _chunk(3, "Photoresist is exposed through a photomask during lithography."),
        ]
    )
    return s


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    ids = [i for i, _ in fused]
    assert ids[0] == 1  # ranked high in both lists
    assert set(ids) == {1, 2, 3, 4}
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_add_deduplicates(store: VectorStore):
    assert store.add([_chunk(0, "duplicate id")]) == 0
    assert len(store) == 4


def test_dense_and_sparse_find_the_right_chunk(store: VectorStore):
    assert store.dense_search("hafnium oxide resistance", 1)[0][0] == 1
    assert store.sparse_search("photomask lithography", 1)[0][0] == 3


def test_hybrid_retriever_returns_top_n(store: VectorStore):
    retriever = HybridRetriever(store, PassthroughReranker(), retrieve_k=4, top_n=2)
    results = retriever.retrieve("what does the cyclic prefix remove in OFDM")
    assert len(results) == 2
    assert results[0].chunk.id == "c0"


def test_empty_store_returns_nothing():
    retriever = HybridRetriever(VectorStore(HashingEmbedder()), PassthroughReranker(), 5, 3)
    assert retriever.retrieve("anything") == []


def test_save_and_load_roundtrip(store: VectorStore, tmp_path: Path):
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path, HashingEmbedder())
    assert [c.id for c in loaded.chunks] == [c.id for c in store.chunks]
    assert loaded.dense_search("photomask", 1)[0][0] == 3


def test_load_rejects_mismatched_embedder(store: VectorStore, tmp_path: Path):
    store.save(tmp_path)
    with pytest.raises(ValueError, match=r"[Rr]e-run"):
        VectorStore.load(tmp_path, HashingEmbedder(dim=256))


def test_load_missing_directory_gives_empty_store(tmp_path: Path):
    assert len(VectorStore.load(tmp_path / "nope", HashingEmbedder())) == 0
