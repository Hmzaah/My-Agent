class Distiller:
    def __init__(self, llm_engine):
        # Use the shared Phi-3 brain
        self.llm = llm_engine

    def distill(self, retrieved_docs):
        """
        Uses Phi-3 to summarize and extract key facts from the retrieved chunks.
        """
        if not retrieved_docs:
            return "No documents retrieved."

        # Combine docs, but truncate if too long for the context window
        combined_text = "\n\n".join(retrieved_docs)[:3000]

        prompt = (
            f"Instruct: You are an expert analyst. Your goal is to extract the most relevant information "
            f"from the context below to help answer a user's request. \n"
            f"Ignore irrelevant details. If the context has no useful info, say 'No relevant information found'.\n\n"
            f"--- Raw Context ---\n{combined_text}\n\n"
            f"Output:"
        )

        output = self.llm(
            prompt,
            max_tokens=300,
            stop=["Instruct:", "---"],
            echo=False,
            temperature=0.3
        )
        
        return output["choices"][0]["text"].strip()
