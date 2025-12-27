import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

def load_documents(folder="knowledge_base"):
    texts = []
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        if fname.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        elif fname.endswith(".pdf"):
            reader = PdfReader(path)
            text = " ".join(page.extract_text() or "" for page in reader.pages)
            texts.append(text)
    return texts

def chunk_text(text, size=400):
    words = text.split()
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size)]

def build_index():
    print("📚 Loading documents...")
    docs = load_documents()
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    print(f"🧠 Total chunks: {len(chunks)}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "faiss_index.bin")

    with open("knowledge_base/doc_chunks.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(chunks))

    print("✅ FAISS index built successfully!")

if __name__ == "__main__":
    build_index()
