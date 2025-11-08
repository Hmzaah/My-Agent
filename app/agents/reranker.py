from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query, docs):
        # Returns docs sorted by cross-encoder score
        scores = self.model.predict([[query, d] for d in docs])
        ranked_docs = [d for _, d in sorted(zip(scores, docs), reverse=True)]
        return ranked_docs
