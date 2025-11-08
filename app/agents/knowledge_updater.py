import os
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def update_knowledge_base(topics):
    # 1. Load data from Wikipedia
    print(f"📘 Fetching topics: {topics}")
    loader = WikipediaLoader(query=topics, load_max_docs=3)
    docs = loader.load()

    # 2. Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # 3. Create embeddings
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Store locally in FAISS
    db_path = os.path.join(os.getcwd(), "../../knowledge_base/faiss_index")
    print(f"💾 Saving to {db_path}")
    db = FAISS.from_documents(chunks, embedder)
    db.save_local(db_path)

    print("✅ Knowledge base updated successfully!")

if __name__ == "__main__":
    topics = input("Enter topics to add (comma-separated): ").split(",")
    update_knowledge_base([t.strip() for t in topics])
