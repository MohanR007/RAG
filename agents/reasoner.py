from typing import List, Dict, Any
import ollama


class ReasonerAgent:
	def __init__(self, model: str = "mistral") -> None:
		self.model = model

	def reason(self, question: str, passages: List[Dict[str, Any]], max_tokens: int = 640) -> str:
		context_blocks = []
		for idx, p in enumerate(passages, start=1):
			block = f"[Passage {idx}]\n{p.get('text','')}\n"
			context_blocks.append(block)
		context = "\n".join(context_blocks)

		prompt = (
			"You are a careful analyst. Given the user question and retrieved passages, "
			"extract only the most relevant facts, remove redundancies, and produce a concise "
			"but complete bullet list of key points grounded in the passages. If information is missing, say so and state reasonable assumptions. "
			"Ensure coverage of all major aspects needed for a self-contained final answer.\n\n"
			f"Question: {question}\n\nPassages:\n{context}\n\nOutput:" 
		)

		response = ollama.chat(
			model=self.model,
			messages=[{"role": "user", "content": prompt}],
			options={
				"num_predict": max_tokens,
				"temperature": 0.2,
				"top_p": 0.9,
				"top_k": 50,
				"repeat_penalty": 1.15,
				"num_ctx": 8192,
				"num_gpu": 999,
				"num_thread": 8
			},
		)
		return response["message"]["content"].strip()

