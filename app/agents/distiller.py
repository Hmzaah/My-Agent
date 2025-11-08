from transformers import T5Tokenizer, T5ForConditionalGeneration

class Distiller:
    def __init__(self, model_name="weijiahaha/t5-small-summarization"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def distill(self, docs):
        combined_text = " ".join(docs)
        input_text = f"summarize: {combined_text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True)
        outputs = self.model.generate(**inputs, max_length=150)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
