import os
import sys
from llama_cpp import Llama

# Ensure app is in path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from agents.planner import Planner
from agents.retriever import Retriever
from agents.reranker import Reranker
from agents.distiller import Distiller
from agents.critic import Critic
from agents.synthesizer import Synthesizer

def load_chunks():
    chunk_path = os.path.join("knowledge_base", "doc_chunks.txt")
    if not os.path.exists(chunk_path):
        print(f"[Error] Chunk file not found.")
        sys.exit(1)
    with open(chunk_path, "r", encoding="utf-8") as f:
        chunks = f.read().split("\n")
    return chunks

def main():
    print("--- Deep Thinking Agentic RAG (Context-Aware) ---")
    
    # 1. Load Resources
    chunks = load_chunks()
    print("[System] Loading Unified AI Brain (Phi-3)...")
    llm_engine = Llama(
        model_path="models/phi-3-mini-4k.gguf",
        n_ctx=4096,
        n_gpu_layers=0,
        verbose=False
    )
    
    # 2. Initialize Agents
    planner = Planner(llm_engine)
    synthesizer = Synthesizer(llm_engine)
    distiller = Distiller(llm_engine)
    critic = Critic(llm_engine)
    retriever = Retriever("faiss_index.bin", chunks, "sentence-transformers/all-MiniLM-L6-v2")
    reranker = Reranker()
    
    chat_history = [] 
    
    print("[System] Agents ready. Type 'exit' to quit.\n")

    while True:
        user_query = input("\nUser: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        
        # 3. Plan (Context-Aware)
        print(f"[Planner] Thinking...")
        plan = planner.generate_plan(user_query, chat_history)
        print(f"  Plan: {plan}")

        # 4. Agentic Loop
        max_retries = 2
        final_context = ""
        critique_feedback = ""
        current_query = user_query

        for attempt in range(max_retries + 1):
            try:
                retrieved_docs = retriever.retrieve(current_query, top_k=8)
                ranked_docs = reranker.rerank(current_query, retrieved_docs)
                distilled_context = distiller.distill(ranked_docs)
                is_sufficient, feedback = critic.evaluate_sufficiency(plan, distilled_context)
                
                if is_sufficient:
                    print(f"  [Critic] PASS: {feedback}")
                    final_context = distilled_context
                    critique_feedback = "Verified."
                    break 
                else:
                    print(f"  [Critic] FAIL: {feedback}")
                    if attempt < max_retries:
                        print(f"  [Loop] Refining search...")
                        current_query = f"{user_query} {feedback}"
                    else:
                        final_context = distilled_context
                        critique_feedback = f"Insufficient: {feedback}"
            except ValueError:
                break

        # 5. Synthesize
        print("  [Synthesizer] responding...")
        answer = synthesizer.generate_response(
            query=user_query,
            plan=plan,
            context=final_context,
            critique=critique_feedback
        )
        
        print(f"\nAI: {answer}")
        chat_history.append((user_query, answer))

if __name__ == "__main__":
    main()
    # ... (This logic is already largely there, we are just refreshing the file to be safe)
    # RERUN the Part 3 block from the previous step if you want the full file clean.
    # OR just proceed to test with the updated Planner/Synthesizer.
