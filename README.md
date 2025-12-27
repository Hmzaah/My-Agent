# Deep Thinking Agentic RAG (Local CPU) 🧠

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Architecture](https://img.shields.io/badge/Architecture-Agentic%20Loop-orange)
![Hardware](https://img.shields.io/badge/Hardware-CPU%20Only-green)
![Status](https://img.shields.io/badge/Status-Active-success)

A fully local, privacy-focused **Agentic RAG (Retrieval-Augmented Generation)** system that runs entirely on a CPU. Unlike standard RAG pipelines that linearly fetch and answer, this agent uses a **"Deep Thinking" Loop** to self-correct, refine queries, and reject poor data before answering.

## 🤖 What Makes This "Agentic"?

Standard RAG systems follow a straight line: `Query -> Retrieve -> Answer`. If the retrieval is bad, the answer is bad.

**This project implements a Cognitive Architecture:**
1.  **Metacognition (The Critic):** The agent reads the retrieved documents and judges: *"Does this actually answer the user's plan?"* If not, it rejects them.
2.  **Adaptive Loop:** If the data is rejected, the agent **rewrites its own query** and tries a different search strategy.
3.  **Context-Aware Planner:** It understands conversation history (e.g., "Who is *his* wife?") by resolving pronouns before planning.
4.  **Unified Brain:** Uses a single **Phi-3 Mini (3.8B)** model for Planning, Critiquing, and Synthesizing, keeping RAM usage low (~4GB).

## 📊 Architecture Flowchart

```mermaid
graph TD
    User[User Query] --> Planner[🧠 Planner]
    Planner -- "Rewrites & Plans" --> LoopStart{Start Cycle}
    
    LoopStart --> Retriever[🔍 Retriever]
    Retriever --> Reranker[Reranker]
    Reranker --> Distiller[📝 Distiller]
    Distiller --> Critic[⚖️ Critic]
    
    Critic -- "FAIL: Missing Info" --> Refine[Refine Query]
    Refine --> LoopStart
    
    Critic -- "PASS: Sufficient" --> Synthesizer[💬 Synthesizer]
    Synthesizer --> Output[Final Answer]
    
    subgraph "Unified Brain (Phi-3)"
    Planner
    Critic
    Distiller
    Synthesizer
    end
