import os
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ---------------------------
# Config
# ---------------------------
DEVICE = "cpu"  # stays on CPU
KNOWLEDGE_BASE = "knowledge_base"  # folder with PDFs and facts.txt

# ---------------------------
# Load documents
# ---------------------------
def load_pdfs(folder):
    docs = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if file.lower().endswith(".pdf"):
            reader = PyPDF2.PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            docs.append(text)
        elif file.lower().endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

print("Loading documents...")
documents = load_pdfs(KNOWLEDGE_BASE)
print(f"Loaded {len(documents)} documents.")

# ---------------------------
# Embed documents
# ---------------------------
embedder = SentenceTransformer('all-MiniLM-L6-v2')
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

# FAISS index
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))
print(f"FAISS index with {index.ntotal} entries ready.")

# ---------------------------
# Load LLM
# ---------------------------
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
chat = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)

# ---------------------------
# Helper functions
# ---------------------------
MAX_PROMPT_LENGTH = 500  # tokens
MAX_NEW_TOKENS = 150
MAX_HISTORY = 2  # keep last 2 turns

chat_history = []

def get_context(query, k=2):
    q_emb = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(np.array(q_emb), k)
    context = ""
    for idx in indices[0]:
        context += documents[idx] + "\n"
    return context

def build_prompt(user_input, context):
    prompt = f"Context: {context}\n"
    for u, a in chat_history[-MAX_HISTORY:]:
        prompt += f"You: {u}\nAgent: {a}\n"
    prompt += f"You: {user_input}\nAgent:"
    # Trim tokens if too long
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
    if tokens.shape[1] > MAX_PROMPT_LENGTH:
        tokens = tokens[:, -MAX_PROMPT_LENGTH:]
        prompt = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return prompt

# ---------------------------
# Chat loop
# ---------------------------
print("🤖 RAG Agent Ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    context = get_context(user_input, k=2)
    if context.strip() == "":
        context = "Use your general knowledge to answer."

    prompt = build_prompt(user_input, context)
    
    response = chat(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        top_p=0.9
    )[0]["generated_text"]

    # Only return the newly generated part
    response_text = response[len(prompt):].strip()
    if response_text == "":
        response_text = "Sorry, I don't know the answer to that."

    print(f"Agent: {response_text}")
    chat_history.append((user_input, response_text))
