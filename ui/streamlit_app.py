"""Chat UI for Jolt. Talks to the HTTP API, so the model runs in one place.

Run the API first (`jolt serve`), then: streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("JOLT_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Jolt", page_icon="⚡", layout="wide")


@st.cache_data(ttl=10)
def health() -> dict | None:
    try:
        return httpx.get(f"{API_URL}/health", timeout=5).json()
    except httpx.HTTPError:
        return None


with st.sidebar:
    st.header("⚡ Jolt")
    info = health()
    if info:
        st.success(f"API online · {info['indexed_chunks']} chunks indexed")
        st.caption(f"LLM backend: {info['llm_backend']} · v{info['version']}")
    else:
        st.error(f"API not reachable at {API_URL}")
    use_web = st.toggle("Allow web search", value=True)
    uploads = st.file_uploader(
        "Add documents", type=["pdf", "txt", "md"], accept_multiple_files=True
    )
    if uploads and st.button("Index files", use_container_width=True):
        files = [("files", (f.name, f.getvalue())) for f in uploads]
        resp = httpx.post(f"{API_URL}/v1/documents", files=files, timeout=600)
        if resp.is_success:
            st.success(f"Added {resp.json()['chunks_added']} chunks")
            health.clear()
        else:
            st.error(resp.text)
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

st.title("⚡ Jolt")
st.caption("Ask questions over your documents. Every claim is cited.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render(msg: dict) -> None:
    st.markdown(msg["content"])
    data = msg.get("data")
    if not data:
        return
    if data["citations"]:
        with st.expander(f"Sources ({len(data['citations'])})"):
            for c in data["citations"]:
                where = c["url"] or (f"{c['source']}, p.{c['page']}" if c["page"] else c["source"])
                st.markdown(f"**[{c['index']}]** {where}")
                st.caption(c["snippet"])
    with st.expander("Agent trace"):
        for step in data["trace"]:
            st.text(f"{step['node']:<11} {step['duration_ms']:>7.0f} ms  {step['detail']}")
        st.caption(
            f"{data['latency_ms']:.0f} ms total · {data['rewrites']} rewrites · "
            f"web used: {data['used_web']}"
        )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render(msg)

if question := st.chat_input("Ask a question"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            resp = httpx.post(
                f"{API_URL}/v1/ask", json={"question": question, "use_web": use_web}, timeout=300
            )
            resp.raise_for_status()
            data = resp.json()
            msg = {"role": "assistant", "content": data["answer"], "data": data}
        except httpx.HTTPError as exc:
            msg = {"role": "assistant", "content": f"Request failed: {exc}"}
        render(msg)
        st.session_state.messages.append(msg)
