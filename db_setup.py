import argparse
import os
from typing import List, Dict
import uuid

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# File processing imports
import PyPDF2
from docx import Document

PERSIST_PATH = os.path.join(os.path.dirname(__file__), "chroma")
COLLECTION_NAME = "local_docs"


def get_client() -> chromadb.Client:
	return chromadb.PersistentClient(path=PERSIST_PATH, settings=Settings(anonymized_telemetry=False))


def get_embedding_function() -> OllamaEmbeddingFunction:
	"""Get Ollama embedding function with default configuration"""
	# Configure to use a common embedding model
	embedding_function = OllamaEmbeddingFunction()
	# Set the model name directly
	embedding_function.model_name = 'nomic-embed-text'
	return embedding_function


def get_or_create_collection(rebuild: bool = False):
	client = get_client()
	if rebuild:
		try:
			client.delete_collection(COLLECTION_NAME)
		except Exception:
			pass
	collection = client.get_or_create_collection(
		name=COLLECTION_NAME,
		embedding_function=get_embedding_function(),
	)
	return collection


def load_sample_documents() -> List[Dict[str, str]]:
	"""Returns a small set of sample documents. Extend or replace with your corpus."""
	docs: List[Dict[str, str]] = [
		{
			"id": "doc_1",
			"text": (
				"This repository demonstrates a local Retrieval-Augmented Generation (RAG) pipeline. "
				"It uses ChromaDB for vector storage and Ollama for local LLM inference."
			),
			"metadata": {"source": "README", "topic": "project"},
		},
		{
			"id": "doc_2",
			"text": (
				"The Retriever Agent queries ChromaDB to fetch semantically relevant passages based on the user question."
			),
			"metadata": {"source": "design", "topic": "retriever"},
		},
		{
			"id": "doc_3",
			"text": (
				"The Reasoner Agent analyzes the retrieved passages, filters noise, and synthesizes the key facts needed to answer."
			),
			"metadata": {"source": "design", "topic": "reasoner"},
		},
		{
			"id": "doc_4",
			"text": (
				"The Responder Agent crafts a concise, user-friendly answer grounded in the reasoned context."
			),
			"metadata": {"source": "design", "topic": "responder"},
		},
	]
	return docs


def ingest_documents(rebuild: bool = False) -> int:
	collection = get_or_create_collection(rebuild=rebuild)
	docs = load_sample_documents()
	ids = [d["id"] for d in docs]
	texts = [d["text"] for d in docs]
	metas = [d["metadata"] for d in docs]
	# Upsert to be idempotent
	collection.upsert(ids=ids, documents=texts, metadatas=metas)
	return len(docs)


def extract_text_from_file(file_path: str) -> str:
	"""Extract text from various file formats"""
	file_ext = os.path.splitext(file_path)[1].lower()
	
	try:
		if file_ext == '.pdf':
			with open(file_path, 'rb') as file:
				reader = PyPDF2.PdfReader(file)
				text = ""
				for page in reader.pages:
					text += page.extract_text() + "\n"
				return text.strip()
		
		elif file_ext == '.txt':
			with open(file_path, 'r', encoding='utf-8') as file:
				return file.read().strip()
		
		elif file_ext == '.docx':
			doc = Document(file_path)
			text = ""
			for paragraph in doc.paragraphs:
				text += paragraph.text + "\n"
			return text.strip()
		
		else:
			raise ValueError(f"Unsupported file format: {file_ext}")
	
	except Exception as e:
		raise Exception(f"Error extracting text from {file_path}: {str(e)}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
	"""Split text into overlapping chunks"""
	if len(text) <= chunk_size:
		return [text]
	
	chunks = []
	start = 0
	
	while start < len(text):
		end = start + chunk_size
		chunk = text[start:end]
		
		# Try to break at sentence boundary
		if end < len(text):
			last_period = chunk.rfind('.')
			last_newline = chunk.rfind('\n')
			break_point = max(last_period, last_newline)
			
			if break_point > start + chunk_size // 2:  # Only break if we're not too far back
				chunk = chunk[:break_point + 1]
				end = start + break_point + 1
		
		chunks.append(chunk.strip())
		start = end - overlap
	
	return chunks


def ingest_uploaded_documents(file_paths: List[str], base_doc_id: str = None) -> int:
	"""Ingest uploaded documents into ChromaDB"""
	collection = get_or_create_collection(rebuild=False)
	
	if base_doc_id is None:
		base_doc_id = str(uuid.uuid4())
	
	all_docs = []
	doc_count = 0
	
	for file_path in file_paths:
		try:
			# Extract text
			text = extract_text_from_file(file_path)
			if not text.strip():
				continue
			
			# Chunk the text
			chunks = chunk_text(text)
			
			# Create documents for each chunk
			for i, chunk in enumerate(chunks):
				doc_id = f"{base_doc_id}_{os.path.basename(file_path)}_{i}"
				all_docs.append({
					"id": doc_id,
					"text": chunk,
					"metadata": {
						"source": os.path.basename(file_path),
						"chunk_index": i,
						"total_chunks": len(chunks),
						"file_type": os.path.splitext(file_path)[1].lower()
					}
				})
				doc_count += 1
		
		except Exception as e:
			print(f"Error processing {file_path}: {str(e)}")
			continue
	
	if all_docs:
		# Batch insert all documents
		ids = [doc["id"] for doc in all_docs]
		texts = [doc["text"] for doc in all_docs]
		metadatas = [doc["metadata"] for doc in all_docs]
		
		collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
	
	return doc_count


def ensure_db_initialized() -> None:
	client = get_client()
	try:
		client.get_collection(COLLECTION_NAME)
	except Exception:
		ingest_documents(rebuild=False)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Setup and ingest documents into ChromaDB")
	parser.add_argument("--rebuild", action="store_true", help="Rebuild the Chroma collection from scratch")
	args = parser.parse_args()

	os.makedirs(PERSIST_PATH, exist_ok=True)
	count = ingest_documents(rebuild=args.rebuild)
	print(f"Ingested {count} documents into collection '{COLLECTION_NAME}' at {PERSIST_PATH}")
