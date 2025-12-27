import streamlit as st
import os
import sys
import time
from llama_cpp import Llama

# Add app folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from agents.planner import Planner
from agents.retriever import Retriever
from agents.reranker import Reranker
from agents.distiller import Distiller
from agents.critic import Critic
from agents.synthesizer import Synthesizer
from agents.web_searcher import WebSearcher

# --- Jolt Configuration ---
st.set_page_config(
    page_title="Jolt | AI Assistant", 
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Jolt" aesthetic
st.markdown("""
<style>
    .stChatInput {border-radius: 20px;}
    .reportview-container {background: #0e1117;}
    h1 {color: #FF4B4B;}
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Control Center ---
with st.sidebar:
    st.header("⚡ Jolt")
    st.caption("v1.0.0 | Local Research Engine")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("GPU", "On", delta_color="normal")
    with col2:
        st.metric("Net", "Active", delta_color="normal")
        
    st.markdown("---")
    
    if st.button("🧹 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🧠 Rebuild Memory", use_container_width=True):
        with st.status("Indexing...", expanded=True) as status:
            os.system("python build_faiss_index.py")
            status.update(label="Memory Updated!", state="complete")
            time.sleep(1)
            st.rerun()
            
    st.markdown("---")
    st.code("Model: Phi-3 Mini\nVRAM: 4GB Limit\nMode: Auto-Learning", language="yaml")

# --- Initialization ---
@st.cache_resource
def initialize_system():
    if not os.path.exists("knowledge_base"):
        os.makedirs("knowledge_base")
        
    chunk_path = os.path.join("knowledge_base", "doc_chunks.txt")
    chunks = []
    if os.path.exists(chunk_path):
        with open(chunk_path, "r", encoding="utf-8") as f:
            chunks = f.read().split("\n")
    
    # Load Engine (GPU Mode)
    llm_engine = Llama(
        model_path="models/phi-3-mini-4k.gguf",
        n_ctx=4096,
        n_gpu_layers=-1, 
        verbose=False
    )
    
    return {
        "planner": Planner(llm_engine),
        "synthesizer": Synthesizer(llm_engine),
        "distiller": Distiller(llm_engine),
        "critic": Critic(llm_engine),
        "retriever": Retriever("faiss_index.bin", chunks, "sentence-transformers/all-MiniLM-L6-v2") if chunks else None,
        "reranker": Reranker(),
        "web_searcher": WebSearcher()
    }

if "agents" not in st.session_state:
    with st.spinner("⚡ Jolt is powering up..."):
        st.session_state.agents = initialize_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main Chat Interface ---
st.title("⚡ Jolt")

for msg in st.session_state.messages:
    avatar = "⚡" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "context" in msg and msg["context"]:
            with st.expander("🔎 Inspect Knowledge Source"):
                st.info(msg["source_type"])
                st.code(msg["context"], language="text")

if prompt := st.chat_input("What do you want to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    if st.session_state.agents:
        with st.chat_message("assistant", avatar="⚡"):
            msg_ph = st.empty()
            
            with st.status("⚡ Jolt is thinking...", expanded=True) as status:
                
                # 1. Plan
                chat_history = [(m["content"], "") for m in st.session_state.messages if m["role"] == "user"]
                plan = st.session_state.agents["planner"].generate_plan(prompt, chat_history)
                status.write(f"📋 **Plan:** {plan}")
                
                # 2. Check Local
                final_context = ""
                source_lbl = "Local Memory"
                
                if st.session_state.agents["retriever"]:
                    status.write("📂 Checking archives...")
                    retrieved = st.session_state.agents["retriever"].retrieve(prompt)
                    ranked = st.session_state.agents["reranker"].rerank(prompt, retrieved)
                    distilled = st.session_state.agents["distiller"].distill(ranked)
                    valid, _ = st.session_state.agents["critic"].evaluate_sufficiency(plan, distilled)
                    
                    if valid:
                        status.write("✅ Found locally.")
                        final_context = distilled
                
                # 3. Check Web (if needed)
                if not final_context:
                    status.write("🌐 Browsing live web...")
                    web_context = st.session_state.agents["web_searcher"].search(prompt)
                    
                    if web_context:
                        status.write("✅ Found on web.")
                        final_context = web_context
                        source_lbl = "Web Search (New Learning)"
                        st.session_state.agents["web_searcher"].save_knowledge(prompt, web_context)
                    else:
                        status.write("❌ No data found.")
                
                status.update(label="Ready", state="complete", expanded=False)

            # 4. Generate
            response = st.session_state.agents["synthesizer"].generate_response(
                query=prompt,
                plan=plan,
                context=final_context,
                critique="Strict Fact Check"
            )
            
            # Typing Animation
            full_res = ""
            for chunk in response.split():
                full_res += chunk + " "
                msg_ph.markdown(full_res + "▌")
                time.sleep(0.01)
            msg_ph.markdown(full_res)
            
            # Context Footer
            if final_context:
                with st.expander("🔎 Inspect Knowledge Source"):
                    st.info(f"Source: {source_lbl}")
                    st.code(final_context, language="text")
            
            if source_lbl == "Web Search (New Learning)":
                st.toast("Jolt learned something new.", icon="💾")

        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_res,
            "context": final_context,
            "source_type": source_lbl
        })
