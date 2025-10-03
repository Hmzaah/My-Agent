from transformers import pipeline

class ChatService:
    def __init__(self):
        """
        Initializes the ChatService by loading the GPT4All model from Hugging Face.
        Make sure your Hugging Face token is saved and authenticated.
        """
        # Load the text-generation pipeline
        self.chatbot = pipeline(
            "text-generation",
            model="TheBloke/GPT4All-13B-snoozy-GPTQ",  # updated model
            device=0,  # GPU: 0 for first GPU, -1 for CPU
            use_auth_token=True  # use the HF token you saved
        )

    def generate_response(self, prompt: str, max_length: int = 512):
        """
        Generates a response from the model for a given prompt.
        """
        output = self.chatbot(prompt, max_length=max_length, do_sample=True, top_p=0.95, temperature=0.7)
        return output[0]['generated_text']
