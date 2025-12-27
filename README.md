# Deep Thinking Agentic RAG (Local CPU) 🧠

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)
![Architecture](https://img.shields.io/badge/Architecture-Agentic%20Loop-orange?style=flat-square)
![Hardware](https://img.shields.io/badge/Hardware-CPU%20Only-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat-square)

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
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': '#BB2528',
      'primaryTextColor': '#fff',
      'primaryBorderColor': '#7C0000',
      'lineColor': '#F8B229',
      'secondaryColor': '#006100',
      'tertiaryColor': '#fff'
    },
    'flowchart': { 'curve': 'stepAfter' }
  }
}%%

graph TD
    User([👤 User Query]) --> Planner[🧠 Planner]
    Planner -- "Rewrites & Plans" --> LoopStart{Start Cycle}
    
    LoopStart --> Retriever[🔍 Retriever]
    Retriever --> Reranker[⚡ Reranker]
    Reranker --> Distiller[📝 Distiller]
    Distiller --> Critic[⚖️ Critic]
    
    Critic -- "FAIL: Missing Info" --> Refine[🔄 Refine Query]
    Refine --> LoopStart
    
    Critic -- "PASS: Sufficient" --> Synthesizer[💬 Synthesizer]
    Synthesizer --> Output([🏁 Final Answer])
    
    subgraph "Unified Brain (Phi-3)"
    Planner
    Critic
    Distiller
    Synthesizer
    end
    
    %% Professional Styling
    classDef plain fill:#fff,stroke:#333,stroke-width:1px,color:#333;
    classDef brain fill:#f9f9f9,stroke:#666,stroke-width:2px,color:#333,stroke-dasharray: 5 5;
    classDef decision fill:#fff3cd,stroke:#e0a800,stroke-width:2px,color:#333;
    classDef success fill:#d4edda,stroke:#28a745,stroke-width:2px,color:#155724;
    classDef fail fill:#f8d7da,stroke:#dc3545,stroke-width:2px,color:#721c24;

    class Planner,Distiller,Synthesizer brain;
    class Critic decision;
    class Output success;
    class Refine fail;
