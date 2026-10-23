import itertools
from pathlib import Path

import pytest

from jolt.ingest import chunk_file, iter_files, split_text

from .conftest import SAMPLE_DIR


def test_split_respects_size_and_keeps_all_words():
    text = " ".join(f"word{i}." for i in range(600))
    chunks = split_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 60 for c in chunks)  # overlap tail may add a little
    joined = " ".join(chunks)
    assert all(f"word{i}." in joined for i in range(600))


def test_split_overlaps_consecutive_chunks():
    text = "\n\n".join(f"Paragraph {i} has some text about topic {i}." for i in range(40))
    chunks = split_text(text, chunk_size=200, chunk_overlap=60)
    for prev, nxt in itertools.pairwise(chunks):
        first_word = nxt.split()[0]
        assert first_word in prev


def test_split_rejects_bad_overlap():
    with pytest.raises(ValueError):
        split_text("abc", chunk_size=100, chunk_overlap=100)


def test_split_empty_text():
    assert split_text("   \n  ") == []


def test_iter_files_filters_and_sorts(tmp_path: Path):
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "skip.docx").write_text("x")
    names = [p.name for p in iter_files([tmp_path])]
    assert names == ["a.txt", "b.md"]


def test_chunk_file_metadata_and_stable_ids():
    path = SAMPLE_DIR / "ofdm_basics.md"
    first = chunk_file(path, 500, 80, root=SAMPLE_DIR)
    second = chunk_file(path, 500, 80, root=SAMPLE_DIR)
    assert first and [c.id for c in first] == [c.id for c in second]
    assert {c.source for c in first} == {"ofdm_basics.md"}
    assert [c.chunk_index for c in first] == list(range(len(first)))
