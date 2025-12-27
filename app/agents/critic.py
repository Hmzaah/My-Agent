class Critic:
    def __init__(self, llm_engine):
        self.llm = llm_engine

    def evaluate_sufficiency(self, plan, context):
        """
        Uses Phi-3 to JUDGE if the context is sufficient for the plan.
        Returns: (bool, feedback_string)
        """
        # 1. Fast fail for empty context
        if not context or len(context.split()) < 10:
            return False, "Context is empty or near-empty."

        # 2. LLM Judgement
        prompt = (
            f"Instruct: You are a strict logic reviewer. Compare the PLAN and the CONTEXT below. "
            f"Determine if the Context contains enough information to answer the Plan.\n"
            f"If YES, output 'PASS'. If NO, output 'FAIL' followed by a short reason.\n\n"
            f"--- PLAN ---\n{plan}\n\n"
            f"--- CONTEXT ---\n{context}\n\n"
            f"Output:"
        )

        output = self.llm(
            prompt,
            max_tokens=50, # Keep it short
            stop=["\n", "Instruct:"],
            echo=False,
            temperature=0.0 # Strict logic
        )
        
        response = output["choices"][0]["text"].strip()
        
        if response.startswith("PASS"):
            return True, "Context is sufficient."
        else:
            # Extract the reason (e.g., "FAIL: Missing popularity stats")
            reason = response.replace("FAIL", "").strip().strip(":")
            if not reason: reason = "Context missing key details."
            return False, reason
