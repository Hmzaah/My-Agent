from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class Retriever:
    def __init__(self):
        # Load existing FAISS index
        index_path = "knowledge_base/faiss_index"
        self.embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.db = FAISS.load_local(index_path, self.embedder, allow_dangerous_deserialization=True)
        self.docs = [d.page_content for d in self.db.similarity_search("test", k=1)]  # sample init check

    def retrieve(self, query, k=3):
        """Return top k relevant docs from FAISS index."""
        results = self.db.similarity_search(query, k=k)
        return [r.page_content for r in results]
