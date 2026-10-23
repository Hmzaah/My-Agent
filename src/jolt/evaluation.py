"""Evaluation harness: retrieval quality, answer quality, abstention and latency on a labelled set.

Dataset format (JSON Lines), one question per line:

    {"id": "q1", "question": "...", "answerable": true,
     "expected_sources": ["ofdm_basics.md"], "answer_keywords": ["cyclic prefix"]}

``answerable: false`` rows check that the agent says it does not know instead of guessing.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jolt.prompts import ABSTAIN
from jolt.service import JoltService


@dataclass
class EvalItem:
    id: str
    question: str
    answerable: bool = True
    expected_sources: list[str] = field(default_factory=list)
    answer_keywords: list[str] = field(default_factory=list)


@dataclass
class ItemResult:
    id: str
    answerable: bool
    retrieved_sources: list[str]
    first_hit_rank: int | None
    abstained: bool
    keyword_recall: float | None
    citation_precision: float | None
    used_web: bool
    rewrites: int
    latency_ms: float
    answer: str


@dataclass
class Report:
    created_at: str
    settings: dict[str, object]
    n_answerable: int
    n_unanswerable: int
    hit_at_k: float
    mrr: float
    answer_rate: float
    abstention_accuracy: float
    keyword_recall: float
    citation_precision: float
    latency_p50_ms: float
    latency_p95_ms: float
    mean_rewrites: float
    items: list[ItemResult]

    def to_markdown(self) -> str:
        k = self.settings.get("rerank_top_n")
        rows = [
            (
                "Retrieval hit@" + str(k),
                f"{self.hit_at_k:.0%}",
                "an expected source is in the top chunks",
            ),
            ("Retrieval MRR", f"{self.mrr:.3f}", "1 / rank of the first correct chunk"),
            ("Answer rate", f"{self.answer_rate:.0%}", "answerable questions not refused"),
            (
                "Abstention accuracy",
                f"{self.abstention_accuracy:.0%}",
                "unanswerable questions refused",
            ),
            (
                "Keyword recall",
                f"{self.keyword_recall:.0%}",
                "expected facts present in the answer",
            ),
            (
                "Citation precision",
                f"{self.citation_precision:.0%}",
                "citations pointing at a correct source",
            ),
            (
                "Latency p50 / p95",
                f"{self.latency_p50_ms:.0f} / {self.latency_p95_ms:.0f} ms",
                "end to end",
            ),
            ("Mean query rewrites", f"{self.mean_rewrites:.2f}", "retries per question"),
        ]
        table = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
        model = (
            self.settings.get("llm_model_path")
            if self.settings.get("llm_backend") == "llamacpp"
            else self.settings.get("llm_model")
        )
        return (
            f"## Evaluation ({self.n_answerable} answerable + "
            f"{self.n_unanswerable} unanswerable questions)\n\n"
            f"LLM: `{self.settings.get('llm_backend')}` / `{model}` · "
            f"embeddings: `{self.settings.get('embedding_model')}` · "
            f"reranker: `{self.settings.get('reranker_backend')}`\n\n"
            "| Metric | Value | Meaning |\n|---|---|---|\n" + table + "\n"
        )


def load_dataset(path: Path) -> list[EvalItem]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(EvalItem(**json.loads(line)))
    return items


def _is_abstention(answer: str) -> bool:
    return ABSTAIN.lower().rstrip(".") in answer.lower()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def evaluate_item(service: JoltService, item: EvalItem, use_web: bool) -> ItemResult:
    retrieved = [d.chunk.source for d in service.retriever.retrieve(item.question)]
    expected = set(item.expected_sources)
    first_hit = next((i for i, s in enumerate(retrieved, start=1) if s in expected), None)

    response = service.ask(item.question, use_web=use_web)
    abstained = _is_abstention(response.answer)
    answer = response.answer.lower()
    recall = (
        sum(kw.lower() in answer for kw in item.answer_keywords) / len(item.answer_keywords)
        if item.answerable and item.answer_keywords
        else None
    )
    precision = (
        sum(c.source in expected for c in response.citations) / len(response.citations)
        if item.answerable and response.citations and expected
        else None
    )
    return ItemResult(
        id=item.id,
        answerable=item.answerable,
        retrieved_sources=retrieved,
        first_hit_rank=first_hit,
        abstained=abstained,
        keyword_recall=recall,
        citation_precision=precision,
        used_web=response.used_web,
        rewrites=response.rewrites,
        latency_ms=response.latency_ms,
        answer=response.answer,
    )


def run_evaluation(
    service: JoltService, dataset: Path, out_dir: Path | None = None, use_web: bool = False
) -> Report:
    items = load_dataset(dataset)
    results = [evaluate_item(service, item, use_web) for item in items]
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]

    def mean(values: list[float]) -> float:
        return statistics.fmean(values) if values else 0.0

    s = service.settings
    report = Report(
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        settings={
            "llm_backend": s.llm_backend,
            "llm_model": s.llm_model,
            "llm_model_path": str(s.llm_model_path),
            "embedding_model": s.embedding_model if s.embedding_backend != "hashing" else "hashing",
            "reranker_backend": s.reranker_backend,
            "rerank_top_n": s.rerank_top_n,
            "max_rewrites": s.max_rewrites,
        },
        n_answerable=len(answerable),
        n_unanswerable=len(unanswerable),
        hit_at_k=mean([1.0 if r.first_hit_rank else 0.0 for r in answerable]),
        mrr=mean([1.0 / r.first_hit_rank if r.first_hit_rank else 0.0 for r in answerable]),
        answer_rate=mean([0.0 if r.abstained else 1.0 for r in answerable]),
        abstention_accuracy=mean([1.0 if r.abstained else 0.0 for r in unanswerable]),
        keyword_recall=mean([r.keyword_recall for r in answerable if r.keyword_recall is not None]),
        citation_precision=mean(
            [r.citation_precision for r in answerable if r.citation_precision is not None]
        ),
        latency_p50_ms=_percentile([r.latency_ms for r in results], 0.5),
        latency_p95_ms=_percentile([r.latency_ms for r in results], 0.95),
        mean_rewrites=mean([float(r.rewrites) for r in results]),
        items=results,
    )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        (out_dir / f"report-{stamp}.json").write_text(
            json.dumps(asdict(report), indent=2), encoding="utf-8"
        )
        (out_dir / f"report-{stamp}.md").write_text(report.to_markdown(), encoding="utf-8")
    return report
