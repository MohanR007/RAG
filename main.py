import argparse
from typing import List, Dict, Any

from agents.retriever import RetrieverAgent
from agents.reasoner import ReasonerAgent
from agents.responder import ResponderAgent
from db_setup import ensure_db_initialized


def run_pipeline(question: str, top_k: int = 4,
                 retriever: RetrieverAgent | None = None,
                 reasoner: ReasonerAgent | None = None,
                 responder: ResponderAgent | None = None) -> Dict[str, Any]:
	ensure_db_initialized()
	retriever = retriever or RetrieverAgent()
	reasoner = reasoner or ReasonerAgent(model="mistral")
	responder = responder or ResponderAgent(model="llama2")

	retrieved: List[Dict[str, Any]] = retriever.retrieve(question=question, top_k=top_k)
	reasoned: str = reasoner.reason(question=question, passages=retrieved)
	final_answer: str = responder.respond(question=question, reasoning_summary=reasoned)

	return {
		"question": question,
		"retrieved": retrieved,
		"reasoned": reasoned,
		"answer": final_answer,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Multi-Agentic RAG with Ollama (Local)")
	parser.add_argument("--question", type=str, default=None, help="Single question to answer")
	parser.add_argument("--top_k", type=int, default=4, help="Number of passages to retrieve")
	args = parser.parse_args()

	if args.question:
		result = run_pipeline(question=args.question, top_k=args.top_k)
		print("Answer:\n" + result["answer"]) 
		return

	print("Interactive mode. Type 'exit' to quit.")
	while True:
		try:
			q = input("\nYour question: ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\nBye.")
			break
		if q.lower() in {"exit", "quit"}:
			print("Bye.")
			break
		if not q:
			continue
		result = run_pipeline(question=q, top_k=args.top_k)
		print("\nAnswer:\n" + result["answer"]) 


if __name__ == "__main__":
	main()
