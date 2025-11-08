from transformers import T5ForConditionalGeneration, T5Tokenizer

class Planner:
    def __init__(self, model_name="t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def plan(self, query):
        # Convert query to sub-tasks
        input_text = f"plan: {query}"
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=64)
        plan_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return [step.strip() for step in plan_text.split(";") if step.strip()]
