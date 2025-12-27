class Planner:
    def __init__(self, llm_engine):
        self.llm = llm_engine

    def generate_plan(self, query, history, feedback=None):
        # If we have feedback (previous failure), we need a NEW plan.
        instruction = "Break this down into simple search steps."
        if feedback:
            instruction = f"PREVIOUS PLAN FAILED: {feedback}. Generate a NEW, different plan to find the answer."

        prompt = f'''<|im_start|>system
You are a strategist. {instruction}
Reply ONLY with the numbered steps.
<|im_end|>
<|im_start|>user
Query: {query}
<|im_end|>
<|im_start|>assistant
'''
        response = self.llm(prompt, max_tokens=256, stop=["<|im_end|>"], echo=False)
        return response['choices'][0]['text'].strip()
