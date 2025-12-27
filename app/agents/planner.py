class Planner:
    def __init__(self, llm_engine):
        self.llm = llm_engine

    def generate_plan(self, query, history=None):
        """
        Step 1: Rewrite query to resolve pronouns (he/it/that).
        Step 2: Generate search plan.
        """
        # Context Construction
        context_str = "No prior context."
        if history:
            last_q, last_a = history[-1]
            # Keep context short to avoid confusing the model
            context_str = f"User: {last_q}\nAI: {last_a}"

        # 1. REWRITE STEP
        rewrite_prompt = (
            f"Instruct: You are a helpful assistant. Rewrite the 'Current Question' to be a standalone sentence. "
            f"Replace pronouns (it, he, she, they) with the specific names they refer to from the Context.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Current Question: {query}\n\n"
            f"Standalone Question:"
        )
        
        output = self.llm(rewrite_prompt, max_tokens=60, stop=["\n"], echo=False, temperature=0.1)
        rewritten_query = output["choices"][0]["text"].strip()
        
        # Fallback if it failed to rewrite
        if len(rewritten_query) < 5: rewritten_query = query
        
        # 2. PLANNING STEP
        plan_prompt = (
            f"Instruct: Break down this question into 2-3 logical search steps. "
            f"Return ONLY the steps separated by '->'.\n\n"
            f"Question: {rewritten_query}\n\n"
            f"Output:"
        )

        output = self.llm(plan_prompt, max_tokens=100, stop=["\n", "Instruct:"], echo=False, temperature=0.1)
        raw_plan = output["choices"][0]["text"].strip()

        # Formatting fix
        if "->" not in raw_plan:
            return f"Search for {rewritten_query}"
            
        return raw_plan
