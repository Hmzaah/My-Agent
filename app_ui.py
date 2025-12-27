import streamlit as st
import os
import sys
import time
from llama_cpp import Llama

sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from agents.planner import Planner
from agents.retriever import Retriever
from agents.reranker import Reranker
from agents.distiller import Distiller
from agents.critic import Critic
from agents.synthesizer import Synthesizer
from agents.web_searcher import WebSearcher

# --- CONFIG ---
st.set_page_config(page_title="Jolt | Deep Thinking", page_icon="⚡", layout="wide")
st.markdown("<style>.stChatInput {border-radius: 20px;} h1 {color: #FF4B4B;}</style>", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚡ Jolt")
    st.caption("Mode: Best Effort (Robust)")
    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.code("Model: Qwen 2.5 (3B)\nLogic: Persist Context", language="yaml")

# --- INIT ---
@st.cache_resource
def init_brain():
    if not os.path.exists("knowledge_base"): os.makedirs("knowledge_base")
    chunk_path = "knowledge_base/doc_chunks.txt"
    chunks = []
    if os.path.exists(chunk_path):
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = f.read().split("\n")
    
    # LOAD MODEL
    llm = Llama(
        model_path="models/qwen2.5-3b.gguf",
        n_ctx=2048,
        n_gpu_layers=-1, 
        verbose=False
    )
    
    return {
        "planner": Planner(llm),
        "retriever": Retriever("faiss_index.bin", chunks, "sentence-transformers/all-MiniLM-L6-v2") if chunks else None,
        "reranker": Reranker(),
        "distiller": Distiller(llm),
        "critic": Critic(llm),
        "web": WebSearcher(),
        "synth": Synthesizer(llm)
    }

if "agents" not in st.session_state:
    st.session_state.agents = init_brain()
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- CHAT ---
st.title("⚡ Jolt")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="⚡" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if "trace" in msg and msg["trace"]:
            with st.expander("🧠 View Thinking Process"):
                st.code(msg["trace"])

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="⚡"):
        msg_ph = st.empty()
        full_trace = ""
        
        with st.status("⚡ Thinking...", expanded=True) as status:
            
            # --- THE LOOP ---
            final_context = "" # Will hold the BEST context found so far
            current_plan = ""
            feedback = None
            max_attempts = 2
            attempt = 0
            
            while attempt < max_attempts:
                attempt += 1
                status.write(f"🔄 **Cycle {attempt}: Planning...**")
                
                # 1. PLAN
                current_plan = st.session_state.agents["planner"].generate_plan(prompt, [], feedback)
                full_trace += f"\n[Plan {attempt}]: {current_plan}\n"
                
                # 2. ACTION
                context_found = ""
                source = "Web" # Default to Web
                
                # Try Local (Attempt 1 only)
                if attempt == 1 and st.session_state.agents["retriever"]:
                    status.write("📂 Checking Memory...")
                    raw = st.session_state.agents["retriever"].retrieve(prompt)
                    if raw:
                        ranked = st.session_state.agents["reranker"].rerank(prompt, raw)
                        distilled = st.session_state.agents["distiller"].distill(ranked)
                        if len(distilled) > 50:
                            context_found = distilled
                            source = "Local"
                # Try Web (If Local failed or Attempt > 1)
                if not context_found:
                    status.write("🌐 Searching Web...")
                    source = "Web"
                    # SEARCH FIX: Only search the PROMPT, not the PLAN
                    context_found = st.session_state.agents["web"].search(prompt)
                    if context_found:
                        st.session_state.agents["web"].save_knowledge(prompt, context_found)

                # --- CRITICAL FIX: ALWAYS SAVE CONTEXT ---
                if context_found:
                    final_context = context_found 

                # 3. CRITIQUE
                status.write("⚖️ Critiquing Evidence...")
                valid, critique = st.session_state.agents["critic"].evaluate_sufficiency(current_plan, context_found)
                
                if valid:
                    status.write("✅ Evidence is sufficient!")
                    break # Success!
                else:
                    status.write(f"❌ Rejection: {critique}")
                    feedback = critique 
                    full_trace += f"[Critique {attempt}]: Failed - {critique}\n"
            
            status.update(label="Thinking Complete", state="complete")
            
            # 4. SYNTHESIZE (Best Effort)
            response = st.session_state.agents["synth"].generate_response(prompt, current_plan, final_context, "Final Attempt")
            
            msg_ph.markdown(response)
            
            if final_context:
                with st.expander("🔎 Source Data Used"):
                    st.info(f"Source: {source} (Best Available)")
                    st.code(final_context[:1000] + "...")

        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "trace": full_trace
        })
