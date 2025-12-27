from llama_cpp import Llama

class Synthesizer:
    def __init__(self, llm_engine):
        self.llm = llm_engine

    def generate_response(self, query, plan, context, critique):
        # STRICT MODE: If no context is provided, refuse to answer.
        if not context or len(context) < 10:
            return "I could not find reliable information about this in my local memory or via web search. I will not guess to avoid giving false information."

        prompt = f'''
        <|system|>
        You are a strict, fact-based research assistant. 
        Your ONLY job is to answer the user's question using the provided [CONTEXT] below.
        
        RULES:
        1. If the answer is NOT in the [CONTEXT], say "I don't know."
        2. DO NOT use your own outside knowledge. ONLY use the provided text.
        3. Cite the source (e.g., "[Source: Web Search]" or "[Source: Local]").
        4. Be direct and concise.

        [PLAN]: {plan}
        [CRITIQUE]: {critique}
        
        [CONTEXT]:
        {context}
        <|end|>
        <|user|>
        {query}
        <|end|>
        <|assistant|>
        '''
        
        response = self.llm(
            prompt, 
            max_tokens=512, 
            stop=["<|end|>", "[CONTEXT]"], 
            echo=False,
            temperature=0.1  # Low temp = Less creativity, more facts
        )
        
        return response['choices'][0]['text'].strip()
