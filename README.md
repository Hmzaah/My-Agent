# ⚡ Jolt | Agentic AI Research Engine

**Jolt** is a local, GPU-accelerated **Agentic RAG (Retrieval-Augmented Generation)** system. It doesn't just read files; it thinks, plans, critiques its own answers, and **autonomously browses the web** to learn what it doesn't know.

<div align="center">
  <img src="https://img.shields.io/badge/AI-Phi--3%20Mini-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Hardware-RTX%203050-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Stack-Streamlit%20%7C%20Llama.cpp%20%7C%20FAISS-orange?style=for-the-badge" />
</div>

---

## 🧠 The Architecture

Jolt uses a **Multi-Agent Chain** to ensure accuracy rather than just speed.

```mermaid
graph TD
    User(User Query) --> Planner[🧠 Planner Agent]
    Planner --> Retrieve[📂 Local Retriever]
    Retrieve --> Rerank[📉 Reranker]
    Rerank --> Distill[⚗️ Distiller]
    Distill --> Critic{⚖️ Critic Agent}
    
    Critic -- "Sufficient" --> Synth[💬 Synthesizer]
    Critic -- "Insufficient" --> Web[🌐 Web Search Agent]
    
    Web --> Save[💾 Auto-Memory Save]
    Save --> Synth
    
    Synth --> Output(Final Answer)
eof
