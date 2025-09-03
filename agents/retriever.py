from typing import List, Dict, Any
import os

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

PERSIST_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma")
COLLECTION_NAME = "local_docs"


class RetrieverAgent:
	def __init__(self, collection_name: str = COLLECTION_NAME, embed_model: str = "nomic-embed-text") -> None:
		self.client = chromadb.PersistentClient(path=PERSIST_PATH, settings=Settings(anonymized_telemetry=False))
		# Create embedding function and set model name
		embedding_function = OllamaEmbeddingFunction()
		embedding_function.model_name = embed_model
		self.collection = self.client.get_or_create_collection(
			name=collection_name,
			embedding_function=embedding_function,
		)

	def retrieve(self, question: str, top_k: int = 4) -> List[Dict[str, Any]]:
		result = self.collection.query(query_texts=[question], n_results=top_k)
		passages: List[str] = result.get("documents", [[]])[0]
		metadatas: List[Dict[str, Any]] = result.get("metadatas", [[]])[0]
		ids: List[str] = result.get("ids", [[]])[0]
		return [
			{"id": doc_id, "text": text, "metadata": meta}
			for doc_id, text, meta in zip(ids, passages, metadatas)
		]

