# app/llm_adapter.py
class LLMAdapter:
    def __init__(self):
        pass

    def generate(self, prompt: str) -> str:
        prompt = prompt.lower()
        if "hello" in prompt:
            return "Hello! I am your AI agent."
        elif "how are you" in prompt:
            return "I'm just code, running perfectly!"
        else:
            return f"[Stub reply] You said: {prompt}"
