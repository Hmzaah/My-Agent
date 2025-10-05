# agent.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ===== Device Setup =====
device = "cpu"  # change to "cuda" later if you have a GPU
print(f"Device set to use {device}\n")

# ===== Load Model & Tokenizer =====
model_name = "microsoft/DialoGPT-medium"
print("Loading model... this may take a minute.")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

print("\n🤖 Smart Chat Agent Ready! Type 'exit' to quit.\n")

# ===== Chat Loop =====
chat_history_ids = None  # for conversation context

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Exiting agent. Goodbye!")
        break

    # Encode user input and append to chat history
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt").to(device)
    if chat_history_ids is not None:
        bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1)
    else:
        bot_input_ids = new_input_ids

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=500,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,        # enables more varied replies
        top_p=0.95,            # nucleus sampling
        temperature=0.7        # creativity control
    )

    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"Agent: {bot_response}\n")
