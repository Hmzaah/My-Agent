"""Loading documents and splitting them into overlapping chunks with source metadata."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from jolt.schemas import Chunk

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
_SEPARATORS = ("\n\n", "\n", ". ", " ")


def iter_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Yield supported files from a mix of file and directory paths, in a stable order."""
    for path in paths:
        if path.is_dir():
            yield from sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
            )
        elif path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path
        else:
            logger.warning("Skipping unsupported or missing path: %s", path)


def load_file(path: Path) -> list[tuple[str, int | None]]:
    """Return (text, page_number) pairs. Text files are a single page with no number."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(page.extract_text() or "", i + 1) for i, page in enumerate(reader.pages)]
    return [(path.read_text(encoding="utf-8", errors="replace"), None)]


def _split_recursive(text: str, size: int, separators: tuple[str, ...]) -> list[str]:
    """Split text into pieces no longer than ``size``, preferring natural boundaries."""
    if len(text) <= size:
        return [text]
    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = separators[0], separators[1:]
    pieces: list[str] = []
    parts = text.split(sep)
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part += sep  # keep the separator so paragraphs, lines and punctuation survive
        if not part.strip():
            continue
        pieces.extend(_split_recursive(part, size, rest) if len(part) > size else [part])
    return pieces


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    """Pack natural pieces greedily into chunks of about ``chunk_size`` characters.

    Each chunk after the first starts with the last ``chunk_overlap`` characters of the
    previous one (cut at a word boundary) so facts spanning a boundary are not lost.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    current = ""
    for piece in _split_recursive(text, chunk_size - chunk_overlap, _SEPARATORS):
        if current.strip() and len(current) + len(piece) > chunk_size:
            chunks.append(current.strip())
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            cut = min((i for i in (tail.find(" "), tail.find("\n")) if i >= 0), default=-1)
            current = tail[cut + 1 :] if cut >= 0 else tail
        current += piece
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if c]


def _chunk_id(source: str, page: int | None, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}|{page}|{index}|{text}".encode()).hexdigest()
    return digest[:16]


def chunk_file(
    path: Path, chunk_size: int, chunk_overlap: int, root: Path | None = None
) -> list[Chunk]:
    source = str(path.relative_to(root)) if root and path.is_relative_to(root) else path.name
    chunks: list[Chunk] = []
    for text, page in load_file(path):
        for piece in split_text(text, chunk_size, chunk_overlap):
            index = len(chunks)
            chunks.append(
                Chunk(
                    id=_chunk_id(source, page, index, piece),
                    text=piece,
                    source=source,
                    page=page,
                    chunk_index=index,
                )
            )
    return chunks
