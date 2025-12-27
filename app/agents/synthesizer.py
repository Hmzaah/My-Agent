class Synthesizer:
    def __init__(self, llm_engine):
        self.llm = llm_engine

    def generate_response(self, query, plan, context, critique):
        # FAIL-SAFE: If no context, refuse to hallucinate.
        if not context:
            return "❌ I tried to search the web, but I couldn't find any reliable information. I will not guess."

        prompt = f'''<|im_start|>system
You are a factual research assistant.
1. Answer the question using ONLY the [CONTEXT] below.
2. If the context contradicts your memory, TRUST THE CONTEXT.
3. If the answer is not in the context, say "I don't know."
<|im_end|>
<|im_start|>user
[QUESTION]: {query}
[CONTEXT]: 
{context}
<|im_end|>
<|im_start|>assistant
'''
        response = self.llm(
            prompt, 
            max_tokens=1024, 
            stop=["<|im_end|>"], 
            echo=False,
            temperature=0.1 # Very strict
        )
        return response['choices'][0]['text'].strip()
