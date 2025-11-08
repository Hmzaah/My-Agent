class Critic:
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def evaluate(self, scores, docs):
        # Filter docs based on relevance threshold
        filtered = [doc for score, doc in zip(scores, docs) if score > self.threshold]
        return filtered
