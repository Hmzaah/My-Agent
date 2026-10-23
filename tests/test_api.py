import pytest
from fastapi.testclient import TestClient

from jolt.api import create_app
from jolt.service import JoltService


@pytest.fixture
def client(service: JoltService):
    with TestClient(create_app(service)) as c:
        yield c


def test_health(client: TestClient, service: JoltService):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] == len(service.store) > 0


def test_ask_returns_answer_with_citations(client: TestClient):
    resp = client.post(
        "/v1/ask", json={"question": "Which IEEE task group is standardising WLAN sensing?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "802.11bf" in body["answer"]
    assert body["citations"][0]["source"] == "isac.md"
    assert resp.headers["x-request-id"]


def test_ask_validates_input(client: TestClient):
    assert client.post("/v1/ask", json={"question": ""}).status_code == 422


def test_upload_and_query_new_document(client: TestClient):
    doc = b"Jolt's mascot is a lightning bolt named Sparky who loves FAISS indexes."
    resp = client.post("/v1/documents", files={"files": ("mascot.md", doc, "text/markdown")})
    assert resp.status_code == 200 and resp.json()["chunks_added"] == 1
    answer = client.post("/v1/ask", json={"question": "What is the name of Jolt's mascot?"}).json()
    assert "Sparky" in answer["answer"]


def test_upload_rejects_unsupported_type(client: TestClient):
    resp = client.post(
        "/v1/documents", files={"files": ("x.exe", b"MZ", "application/octet-stream")}
    )
    assert resp.status_code == 415


def test_graph_endpoint_returns_mermaid(client: TestClient):
    text = client.get("/v1/graph").text
    for node in ("plan", "retrieve", "grade", "rewrite", "web_search", "generate"):
        assert node in text
