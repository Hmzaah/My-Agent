import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jolt.cli import app
from jolt.config import get_settings

from .conftest import SAMPLE_DIR

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def offline_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for key, value in {
        "JOLT_LLM_BACKEND": "fake",
        "JOLT_EMBEDDING_BACKEND": "hashing",
        "JOLT_RERANKER_BACKEND": "none",
        "JOLT_WEB_SEARCH_ENABLED": "false",
        "JOLT_INDEX_DIR": str(tmp_path / "index"),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)  # ignore any .env in the repo
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_ingest_ask_eval_graph(offline_env, tmp_path: Path):
    runner = CliRunner()

    result = runner.invoke(app, ["ingest", str(SAMPLE_DIR)])
    assert result.exit_code == 0, result.output
    assert "Indexed 6 files" in result.output

    result = runner.invoke(app, ["ask", "Which IEEE task group is standardising WLAN sensing?"])
    assert result.exit_code == 0, result.output
    assert "802.11bf" in result.output and "isac.md" in result.output

    result = runner.invoke(app, ["ask", "What is the boiling point of mercury?", "--json"])
    body = json.loads(result.output)
    assert body["grounded"] is False

    out = tmp_path / "results"
    dataset = ROOT / "eval" / "questions.jsonl"
    result = runner.invoke(app, ["eval", str(dataset), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Abstention accuracy" in result.output
    assert list(out.glob("report-*.md"))

    result = runner.invoke(app, ["graph"])
    assert "web_search" in result.output
