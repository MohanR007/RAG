import ollama


class ResponderAgent:
	def __init__(self, model: str = "llama2") -> None:
		self.model = model

	def respond(self, question: str, reasoning_summary: str, max_tokens: int = 768) -> str:
		prompt = (
			"You are a helpful assistant. Using the reasoning notes below, craft a complete, "
			"authentic, and elegant answer in a conversational ChatGPT-like style. Keep it grounded in the notes.\n\n"
			"Formatting rules (must follow):\n"
			"- Start with a concise 1–2 sentence introduction that answers directly.\n"
			"- Use short paragraphs (2–3 sentences) and clear section headings.\n"
			"- Use bulleted or numbered lists for structured points.\n"
			"- Include a simple example or mini snippet if it improves clarity (use a proper code block).\n"
			"- Keep emojis minimal and helpful (e.g., ✅, 💡) — optional.\n"
			"- Finish with a brief takeaway line.\n\n"
			"Template (adapt naturally; do not be rigid):\n"
			"### Answer\n"
			"<2-sentence intro>\n\n"
			"#### Key Points\n"
			"- <bullet point 1>\n- <bullet point 2>\n- <bullet point 3>\n\n"
			"#### Example (optional)\n"
			"```\n<concise illustrative example>\n```\n\n"
			"Takeaway: <one line>\n\n"
			f"Question: {question}\n\nReasoning Notes:\n{reasoning_summary}\n\nFinal Answer:"
		)
		response = ollama.chat(
			model=self.model,
			messages=[{"role": "user", "content": prompt}],
			options={
				"num_predict": max_tokens,
				"temperature": 0.2,
				"top_p": 0.9,
				"top_k": 50,
				"repeat_penalty": 1.1,
				"num_ctx": 8192,
				"num_gpu": 999,
				"num_thread": 8
			},
		)
		return response["message"]["content"].strip()

