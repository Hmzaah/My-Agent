from typing import List
import faiss
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, index_path: str, documents: List[str], model_name: str):
        self.documents = documents
        self.embedder = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)

    def retrieve(self, query: str, top_k: int = 8) -> List[str]:
        """
        Retrieve top_k documents for a query.
        Enforces minimum context depth to avoid thin-context failures.
        """
        if top_k < 5:
            raise ValueError("top_k must be >= 5 for sufficient retrieval context")

        query_embedding = self.embedder.encode([query])
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            results.append(self.documents[idx])

        if len(results) < 3:
            raise RuntimeError(
                f"Retriever returned insufficient documents: {len(results)}"
            )

        return results
