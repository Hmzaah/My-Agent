from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. Model and tokenizer
model_name = "gpt2"  # small and CPU-friendly
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. Set device to CPU
device = "cpu"
model.to(device)

# 3. Text generation function
def generate_text(prompt, max_length=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_p=0.9,
        top_k=50
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 4. Interactive loop
print("=== AI Agent (type 'exit' or 'quit' to stop) ===")
while True:
    prompt = input("You: ")
    if prompt.lower() in ["exit", "quit"]:
        print("Exiting AI agent. Goodbye!")
        break
    response = generate_text(prompt)
    print("AI:", response)
