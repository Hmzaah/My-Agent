"""Backends that normally talk to the network, tested against fakes."""

import json

import httpx
import openai
import pytest

from jolt.config import Settings
from jolt.embeddings import HashingEmbedder
from jolt.llm import OpenAICompatibleLLM, build_llm
from jolt.testing import ExtractiveFakeLLM
from jolt.web_search import DuckDuckGoSearcher


def _openai_with_transport(monkeypatch: pytest.MonkeyPatch, handler) -> None:
    real = openai.OpenAI

    def patched(**kwargs):
        return real(http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kwargs)

    monkeypatch.setattr(openai, "OpenAI", patched)


def _completion(content: str) -> dict:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def test_openai_backend_sends_json_mode(monkeypatch: pytest.MonkeyPatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=_completion(' {"query": "ofdm"} '))

    _openai_with_transport(monkeypatch, handler)
    llm = OpenAICompatibleLLM(Settings(_env_file=None, llm_backend="openai", llm_model="m"))
    out = llm.chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert out == '{"query": "ofdm"}'
    assert seen[0]["response_format"] == {"type": "json_object"}
    assert seen[0]["model"] == "m"


def test_openai_backend_falls_back_when_json_mode_rejected(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "unsupported"}})
        return httpx.Response(200, json=_completion("ok"))

    _openai_with_transport(monkeypatch, handler)
    llm = OpenAICompatibleLLM(Settings(_env_file=None, llm_backend="openai"))
    assert llm.chat([{"role": "user", "content": "x"}], json_mode=True) == "ok"
    assert llm.chat([{"role": "user", "content": "x"}], json_mode=True) == "ok"
    assert len(calls) == 3  # one rejected call, then JSON mode stays off


def test_llamacpp_backend_reports_missing_model(tmp_path):
    pytest.importorskip("llama_cpp")
    settings = Settings(_env_file=None, llm_model_path=tmp_path / "missing.gguf")
    with pytest.raises(FileNotFoundError):
        build_llm(settings)


def test_fake_backend_is_selectable():
    assert isinstance(build_llm(Settings(_env_file=None, llm_backend="fake")), ExtractiveFakeLLM)


def test_web_search_failure_returns_empty(monkeypatch: pytest.MonkeyPatch):
    class Boom:
        def text(self, *args, **kwargs):
            raise RuntimeError("rate limited")

    monkeypatch.setattr("ddgs.DDGS", Boom)
    assert DuckDuckGoSearcher().search("anything", 3) == []


def test_web_search_maps_results(monkeypatch: pytest.MonkeyPatch):
    class Fake:
        def text(self, query, max_results):
            return [
                {"title": "T", "href": "https://e.org", "body": "Body text."},
                {"title": "Empty", "href": "https://e.org/2", "body": ""},
            ]

    monkeypatch.setattr("ddgs.DDGS", Fake)
    chunks = DuckDuckGoSearcher().search("q", 3)
    assert len(chunks) == 1 and chunks[0].url == "https://e.org" and chunks[0].source == "T"


def test_hashing_embedder_is_normalised_and_deterministic():
    e = HashingEmbedder(dim=64)
    a = e.embed(["the cyclic prefix", "the cyclic prefix", ""])
    assert a.shape == (3, 64)
    assert abs(float((a[0] ** 2).sum()) - 1.0) < 1e-5
    assert (a[0] == a[1]).all() and not a[2].any()
