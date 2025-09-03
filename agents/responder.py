import ollama


class ResponderAgent:
	def __init__(self, model: str = "llama2") -> None:
		self.model = model

	def respond(self, question: str, reasoning_summary: str, max_tokens: int = 512) -> str:
		prompt = (
			"You are a helpful assistant. Using the reasoning notes below, craft a clear, "
			"concise, and well-structured answer for the user. Keep it grounded in the notes.\n\n"
			f"Question: {question}\n\nReasoning Notes:\n{reasoning_summary}\n\nFinal Answer:"
		)
		response = ollama.chat(
			model=self.model,
			messages=[{"role": "user", "content": prompt}],
			options={"num_predict": max_tokens},
		)
		return response["message"]["content"].strip()

