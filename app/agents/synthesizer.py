from llama_cpp import Llama

class Synthesizer:
    def __init__(self):
        model_path = "models/phi-3-mini-4k.gguf"
        self.model = Llama(model_path=model_path, n_ctx=4096, verbose=False)

    def generate(self, prompt):
        output = self.model(
            prompt,
            max_tokens=300,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "assistant:"],
        )
        return output["choices"][0]["text"].strip()
