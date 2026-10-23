from pathlib import Path

from jolt.evaluation import load_dataset, run_evaluation
from jolt.service import JoltService

DATASET = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"


def test_dataset_is_well_formed():
    items = load_dataset(DATASET)
    assert len(items) >= 20
    assert all(i.expected_sources and i.answer_keywords for i in items if i.answerable)
    assert any(not i.answerable for i in items)


def test_run_evaluation_produces_bounded_metrics_and_files(service: JoltService, tmp_path: Path):
    report = run_evaluation(service, DATASET, out_dir=tmp_path)
    for value in (
        report.hit_at_k,
        report.mrr,
        report.answer_rate,
        report.abstention_accuracy,
        report.keyword_recall,
        report.citation_precision,
    ):
        assert 0.0 <= value <= 1.0
    assert report.n_answerable + report.n_unanswerable == len(load_dataset(DATASET))
    assert report.abstention_accuracy == 1.0  # the fake model must not invent answers
    assert "| Retrieval MRR |" in report.to_markdown()
    assert len(list(tmp_path.glob("report-*.json"))) == 1
