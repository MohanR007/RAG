import streamlit as st
import tempfile
import os
from typing import List, Dict, Any
import uuid

from agents.retriever import RetrieverAgent
from agents.reasoner import ReasonerAgent
from agents.responder import ResponderAgent
from db_setup import ensure_db_initialized, ingest_uploaded_documents


def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False


def process_uploaded_files(uploaded_files) -> int:
	"""Process uploaded files and ingest them into ChromaDB"""
	if not uploaded_files:
		return 0
	
	try:
		# Ensure DB is initialized
		ensure_db_initialized()
	except Exception as e:
		st.error(f"❌ Database initialization failed: {str(e)}")
		return 0
	
	# Process each file
	processed_count = 0
	for uploaded_file in uploaded_files:
		# Save uploaded file to temporary location
		with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
			tmp_file.write(uploaded_file.getvalue())
			tmp_file_path = tmp_file.name
		
		try:
			# Ingest the file
			doc_id = str(uuid.uuid4())
			ingest_uploaded_documents([tmp_file_path], doc_id)
			processed_count += 1
			st.success(f"✅ Processed: {uploaded_file.name}")
		except Exception as e:
			error_msg = str(e)
			if "model" in error_msg.lower() and "not found" in error_msg.lower():
				st.error(f"❌ Error processing {uploaded_file.name}: Embedding model not found. Please run: ollama pull nomic-embed-text")
			elif "Connection refused" in error_msg or "Failed to establish a new connection" in error_msg:
				st.error(f"❌ Error processing {uploaded_file.name}: Cannot connect to Ollama. Please ensure Ollama is running with 'ollama serve'")
			else:
				st.error(f"❌ Error processing {uploaded_file.name}: {error_msg}")
		finally:
			# Clean up temporary file
			if os.path.exists(tmp_file_path):
				os.unlink(tmp_file_path)
	
	return processed_count


def run_rag_pipeline(question: str) -> Dict[str, Any]:
	"""Run the RAG pipeline: Retriever → Reasoner → Responder"""
	try:
		# Initialize agents
		retriever = RetrieverAgent()
		reasoner = ReasonerAgent(model="mistral")
		responder = ResponderAgent(model="llama2")
		
		# Run pipeline
		retrieved = retriever.retrieve(question=question, top_k=4)
		reasoned = reasoner.reason(question=question, passages=retrieved)
		final_answer = responder.respond(question=question, reasoning_summary=reasoned)
		
		return {
			"question": question,
			"retrieved": retrieved,
			"reasoned": reasoned,
			"answer": final_answer,
		}
	except Exception as e:
		# Check if it's an Ollama connection issue
		if "Connection refused" in str(e) or "Failed to establish a new connection" in str(e):
			raise Exception("❌ Cannot connect to Ollama. Please ensure Ollama is running with 'ollama serve'")
		elif "model" in str(e).lower() and "not found" in str(e).lower():
			raise Exception("❌ Required Ollama model not found. Please run: ollama pull mistral && ollama pull llama2")
		else:
			raise Exception(f"❌ Error in RAG pipeline: {str(e)}")


def main():
    st.set_page_config(
        page_title="Multi-Agentic RAG with Ollama",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Multi-Agentic RAG with Ollama")
    st.markdown("Upload documents and ask questions using local AI agents!")
    
    initialize_session_state()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📄 Document Upload")
        st.markdown("Upload documents to add them to the knowledge base.")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or DOCX files"
        )
        
        if st.button("📥 Ingest Documents", type="primary"):
            if uploaded_files:
                with st.spinner("Processing documents..."):
                    processed_count = process_uploaded_files(uploaded_files)
                    if processed_count > 0:
                        st.success(f"Successfully processed {processed_count} document(s)!")
                        st.session_state.db_initialized = True
            else:
                st.warning("Please select files to upload.")
        
        st.markdown("---")
        st.markdown("### 📊 Database Status")
        if st.session_state.db_initialized:
            st.success("✅ Database initialized")
        else:
            st.info("ℹ️ Upload documents to initialize database")
        
        # Check Ollama status
        st.markdown("### 🤖 Ollama Status")
        try:
            import ollama
            # Try to list models to check if Ollama is running
            models = ollama.list()
            if models.get('models'):
                st.success("✅ Ollama running")
                st.info(f"📋 Available models: {len(models['models'])}")
            else:
                st.warning("⚠️ Ollama running but no models found")
        except Exception:
            st.error("❌ Ollama not running")
            st.info("💡 Run 'ollama serve' to start")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Question input
        question = st.text_area(
            "Enter your question:",
            placeholder="What would you like to know?",
            height=100
        )
        
        if st.button("🔍 Get Answer", type="primary", disabled=not question.strip()):
            if not st.session_state.db_initialized:
                st.warning("⚠️ Please upload and ingest documents first!")
            else:
                with st.spinner("Processing your question..."):
                    try:
                        result = run_rag_pipeline(question)
                        
                        # Display answer
                        st.markdown("### 📝 Answer")
                        st.write(result["answer"])
                        
                        # Store in session state for chat history
                        st.session_state.messages.append({
                            "role": "user",
                            "content": question
                        })
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": result["answer"]
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error processing question: {str(e)}")
    
    with col2:
        st.header("📋 Chat History")
        
        if st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        else:
            st.info("No conversation yet. Ask a question to get started!")
        
        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <small>Multi-Agentic RAG with Ollama | Local AI Agents</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
