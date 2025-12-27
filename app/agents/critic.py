class Critic:
    def __init__(self, llm_engine):
        self.llm = llm_engine

    def evaluate_sufficiency(self, plan, context_found):
        # If context is empty, instant fail (save token cost)
        if not context_found or len(context_found) < 50:
            return False, "Context is empty or too short."

        prompt = f'''<|im_start|>system
You are a strict evaluator. 
Check if the [CONTEXT] contains the answer to the [PLAN].
1. If the info is there, reply "YES".
2. If the info is missing/irrelevant, reply "NO" and explain why.
<|im_end|>
<|im_start|>user
[PLAN]: {plan}
[CONTEXT]: {context_found[:2000]}
<|im_end|>
<|im_start|>assistant
'''
        response = self.llm(prompt, max_tokens=64, stop=["<|im_end|>"], echo=False)
        output = response['choices'][0]['text'].strip()
        
        if "YES" in output.upper():
            return True, "Sufficient"
        else:
            return False, output
