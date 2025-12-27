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

# --- Page Config ---
st.set_page_config(
    page_title="Deep Thinking Agent (GPU)", 
    page_icon="🚀", 
    layout="wide"
)

# --- Sidebar Control Center ---
with st.sidebar:
    st.title("🚀 Deep Thinking Agent")
    st.markdown("---")
    st.markdown("**System Status:**")
    st.success("🟢 AI Model Loaded (Phi-3)")
    st.success("🟢 GPU Acceleration: ON")
    st.success("🟢 Vector DB Active")
    
    st.markdown("---")
    if st.button("🗑️ Clear Conversation", type="primary"):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    st.caption("Architecture: Agentic RAG\nModel: Phi-3 Mini 4K\nHardware: RTX 3050 (Turbo)")

# --- Initialization (Cached) ---
@st.cache_resource
def initialize_system():
    chunk_path = os.path.join("knowledge_base", "doc_chunks.txt")
    if not os.path.exists(chunk_path):
        st.error("❌ Knowledge base not found! Run build_faiss_index.py first.")
        return None
        
    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = f.read().split("\n")
    
    # LOAD MODEL ON GPU (n_gpu_layers=-1 means "All layers to VRAM")
    llm_engine = Llama(
        model_path="models/phi-3-mini-4k.gguf",
        n_ctx=4096,
        n_gpu_layers=-1,  # <--- THE TURBO SWITCH
        verbose=False
    )
    
    return {
        "planner": Planner(llm_engine),
        "synthesizer": Synthesizer(llm_engine),
        "distiller": Distiller(llm_engine),
        "critic": Critic(llm_engine),
        "retriever": Retriever("faiss_index.bin", chunks, "sentence-transformers/all-MiniLM-L6-v2"),
        "reranker": Reranker()
    }

if "agents" not in st.session_state:
    with st.spinner("🚀 Revving up the RTX 3050... (Loading Model)"):
        st.session_state.agents = initialize_system()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Render Chat History ---
st.header("💬 Agentic Chat Interface (GPU Accelerated)")

for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "context" in msg and msg["context"]:
            with st.expander("🕵️ View Source Documents"):
                st.code(msg["context"], language="text")

# --- Main Interaction Loop ---
if prompt := st.chat_input("Ask me anything..."):
    # 1. User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. AI Processing
    if st.session_state.agents:
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            status = st.status("🧠 **Deep Thinking in progress...**", expanded=True)
            
            # Prepare history
            chat_history_tuples = [(m["content"], "") for m in st.session_state.messages if m["role"] == "user"]
            
            # A. PLAN
            status.write("🤔 **Planner:** Strategizing...")
            plan = st.session_state.agents["planner"].generate_plan(prompt, chat_history_tuples)
            status.write(f"📋 **Plan:** {plan}")
            
            # B. EXECUTE LOOP
            final_context = ""
            critique_feedback = ""
            current_query = prompt
            max_retries = 2
            
            for attempt in range(max_retries + 1):
                status.write(f"🔍 **Cycle {attempt+1}:** Searching knowledge base...")
                
                # Retrieve & Rerank
                retrieved_docs = st.session_state.agents["retriever"].retrieve(current_query, top_k=5)
                ranked_docs = st.session_state.agents["reranker"].rerank(current_query, retrieved_docs)
                distilled_context = st.session_state.agents["distiller"].distill(ranked_docs)
                
                # Critic
                status.write("⚖️ **Critic:** Verifying facts...")
                is_sufficient, feedback = st.session_state.agents["critic"].evaluate_sufficiency(plan, distilled_context)
                
                if is_sufficient:
                    status.write(f"✅ **Verified:** Found sufficient data.")
                    final_context = distilled_context
                    critique_feedback = "Verified."
                    break
                else:
                    status.write(f"❌ **Refining:** {feedback}")
                    if attempt < max_retries:
                        current_query = f"{prompt} {feedback}"
                    else:
                        status.write("⚠️ **Limit:** Proceeding with best available info.")
                        final_context = distilled_context
                        critique_feedback = f"Insufficient: {feedback}"
            
            status.update(label="✅ Thinking Complete", state="complete", expanded=False)
            
            # C. SYNTHESIZE
            response = st.session_state.agents["synthesizer"].generate_response(
                query=prompt,
                plan=plan,
                context=final_context,
                critique=critique_feedback
            )
            
            # Typewriter effect
            full_response = ""
            for chunk in response.split():
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01) # Faster typing speed for GPU
            message_placeholder.markdown(full_response)
            
            # Show Sources
            if final_context:
                with st.expander("🕵️ View Source Documents"):
                    st.code(final_context, language="text")
            
        # 3. Save AI Message
        st.session_state.messages.append({
            "role": "assistant", 
            "content": full_response,
            "context": final_context
        })
