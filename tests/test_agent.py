import json

from jolt.config import Settings
from jolt.llm import parse_json
from jolt.prompts import ABSTAIN
from jolt.schemas import Chunk
from jolt.service import JoltService
from jolt.testing import ExtractiveFakeLLM, StaticWebSearcher

from .conftest import make_service


def nodes_of(response) -> list[str]:
    return [s.node for s in response.trace]


def test_answerable_question_is_grounded_and_cited(service: JoltService):
    r = service.ask("What causes molecular absorption of terahertz signals?")
    assert r.grounded
    assert "water vapour" in r.answer.lower()
    assert r.citations[0].source == "terahertz_communication.md"
    assert nodes_of(r) == ["plan", "retrieve", "grade", "generate"]
    assert r.rewrites == 0 and not r.used_web


def test_unanswerable_question_rewrites_then_abstains(service: JoltService):
    r = service.ask("Who won the 2018 FIFA World Cup final?")
    assert r.answer == ABSTAIN
    assert not r.grounded and r.citations == []
    assert r.rewrites == service.settings.max_rewrites
    assert nodes_of(r).count("retrieve") == service.settings.max_rewrites + 1


def test_web_fallback_when_index_is_empty(settings: Settings):
    web = StaticWebSearcher(
        [
            Chunk(
                id="w1",
                text="Argentina won the 2022 FIFA World Cup final against France.",
                source="FIFA results",
                url="https://example.org/fifa",
            )
        ]
    )
    svc = make_service(settings, web=web)  # nothing ingested
    r = svc.ask("Who won the 2022 FIFA World Cup final?", use_web=True)
    assert r.used_web and r.grounded
    assert r.citations[0].url == "https://example.org/fifa"
    assert r.rewrites == 0  # rewriting an empty index is pointless, so it goes straight to the web
    assert web.queries == ["Who won the 2022 FIFA World Cup final?"]


def test_web_not_used_when_disabled_per_request(settings: Settings):
    web = StaticWebSearcher([Chunk(id="w", text="x", source="s")])
    svc = make_service(settings, web=web)
    r = svc.ask("Anything at all?", use_web=False)
    assert not r.used_web and web.queries == []
    assert r.answer == ABSTAIN


def test_learn_from_web_adds_results_to_index(settings: Settings):
    settings.learn_from_web = True
    web = StaticWebSearcher(
        [
            Chunk(
                id="w1",
                text="The Kaveri river flows through Karnataka and Tamil Nadu.",
                source="web",
            )
        ]
    )
    svc = make_service(settings, web=web)
    svc.ask("Which states does the Kaveri river flow through?", use_web=True)
    assert len(svc.store) == 1
    assert (settings.index_dir / "chunks.jsonl").exists()


class GarbageGraderLLM(ExtractiveFakeLLM):
    def chat(self, messages, **kwargs):
        if messages[0]["content"].startswith("You are the grader"):
            return "sure, these look fine"
        return super().chat(messages, **kwargs)


def test_unparseable_grader_output_keeps_documents(settings: Settings, service: JoltService):
    svc = JoltService(
        settings,
        llm=GarbageGraderLLM(),
        embedder=service.store.embedder,
        reranker=service.retriever.reranker,
        web=StaticWebSearcher(),
        store=service.store,
    )
    r = svc.ask("What is the conductive filament in an HfO2 RRAM cell made of?")
    assert r.grounded and r.rewrites == 0


def test_many_rewrites_do_not_hit_recursion_limit(settings: Settings):
    settings.max_rewrites = 6
    svc = make_service(settings)
    svc.store.add([Chunk(id="a", text="Unrelated text about cooking rice.", source="a.md")])
    r = svc.ask("How do transistors switch?")
    assert r.rewrites == 6 and r.answer == ABSTAIN


def test_parse_json_handles_fences_and_prose():
    assert parse_json('{"a": 1}') == {"a": 1}
    assert parse_json('Here you go:\n```json\n{"query": "x"}\n```') == {"query": "x"}
    assert parse_json("no json here") is None
    assert parse_json(json.dumps([1, 2])) is None
