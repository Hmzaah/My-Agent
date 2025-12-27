class Synthesizer:
    def __init__(self, llm_engine):
        self.model = llm_engine

    def generate_response(self, query, plan, context, critique):
        # We allow General Knowledge but force the model to label its source.
        prompt = (
            f"Instruct: You are a helpful AI assistant. Answer the user's question accurately.\n"
            f"--- LOGIC RULES ---\n"
            f"1. ANALYZE the Context. If it contains the answer, use it. Start answer with: '[Source: Local DB]'\n"
            f"2. IF the Context is empty or irrelevant, use your own General Knowledge. Start answer with: '[Source: General Knowledge]'\n"
            f"3. Do NOT invent fake facts. If you truly don't know, say 'I don't know'.\n\n"
            f"--- Plan ---\n{plan}\n\n"
            f"--- Context ---\n{context}\n\n"
            f"--- Question ---\n{query}\n\n"
            f"Output:"
        )
        return self.generate(prompt)

    def generate(self, prompt):
        output = self.model(
            prompt,
            max_tokens=500,
            temperature=0.3,    # Slight creativity allowed for General Knowledge
            repeat_penalty=1.1, # Prevent loops
            stop=["Instruct:", "User:", "Output:", "---"],
            echo=False
        )
        return output["choices"][0]["text"].strip()
