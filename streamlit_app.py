import streamlit as st
import tempfile
import os
from typing import List, Dict, Any
import uuid
import json
from datetime import datetime

from agents.retriever import RetrieverAgent
from agents.reasoner import ReasonerAgent
from agents.responder import ResponderAgent
from db_setup import ensure_db_initialized, ingest_uploaded_documents


def initialize_session_state():
    """Initialize session state variables"""
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    if 'db_initialized' not in st.session_state:
        st.session_state.db_initialized = False
    if 'sidebar_collapsed' not in st.session_state:
        st.session_state.sidebar_collapsed = False
    if 'agents' not in st.session_state:
        st.session_state.agents = None


def get_agents():
    """Create or retrieve cached agents for the session."""
    if st.session_state.agents is None:
        retriever = RetrieverAgent()
        reasoner = ReasonerAgent(model="mistral")
        responder = ResponderAgent(model="llama2")
        st.session_state.agents = {
            'retriever': retriever,
            'reasoner': reasoner,
            'responder': responder,
        }
    return st.session_state.agents


def split_user_questions(text: str) -> List[str]:
    """Split a compound prompt into separate questions.
    Rules:
    - Prefer numbered lines like '1. ...', '2. ...'
    - Fallback: non-empty lines become separate questions
    - Trim whitespace; ignore empty pieces
    """
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    numbered = []
    for ln in lines:
        # Accept patterns like '1. question', '2) question', '1 - question'
        if len(ln) > 2 and (ln[0].isdigit() and (ln[1] in ".)" or (ln[1] == ' ' and ln[2] in "-.") )):
            # Remove leading numbering tokens
            cleaned = ln
            # Common prefixes
            for pref in [") ", ")", ". ", ".", " - ", " -", " "]:
                if cleaned[:2].isdigit() if False else False:  # placeholder to keep indentation structure
                    pass
            # Minimal robust cleanup
            idx = 0
            while idx < len(cleaned) and cleaned[idx].isdigit():
                idx += 1
            while idx < len(cleaned) and cleaned[idx] in [')', '.', '-', ' ']:
                idx += 1
            cleaned = cleaned[idx:].strip()
            if cleaned:
                numbered.append(cleaned)
        else:
            numbered.append(ln)
    return numbered


def create_new_chat():
    """Create a new chat and return its ID"""
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        'id': chat_id,
        'title': 'New Chat',
        'messages': [],
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    st.session_state.current_chat_id = chat_id
    return chat_id


def get_chat_title(messages: List[Dict]) -> str:
    """Generate a title from the first user message"""
    for message in messages:
        if message['role'] == 'user':
            content = message['content'].strip()
            if content:
                # Truncate to 50 characters
                return content[:50] + '...' if len(content) > 50 else content
    return 'New Chat'


def update_chat_title(chat_id: str):
    """Update the chat title based on messages"""
    if chat_id in st.session_state.chats:
        chat = st.session_state.chats[chat_id]
        if chat['messages']:
            chat['title'] = get_chat_title(chat['messages'])
            chat['updated_at'] = datetime.now().isoformat()


def switch_to_chat(chat_id: str):
    """Switch to a specific chat"""
    st.session_state.current_chat_id = chat_id


def delete_chat(chat_id: str):
    """Delete a chat"""
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        # If we deleted the current chat, switch to the first available chat or create a new one
        if st.session_state.current_chat_id == chat_id:
            if st.session_state.chats:
                st.session_state.current_chat_id = list(st.session_state.chats.keys())[0]
            else:
                create_new_chat()


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
        # Initialize or reuse agents
        agents = get_agents()
        retriever: RetrieverAgent = agents['retriever']
        reasoner: ReasonerAgent = agents['reasoner']
        responder: ResponderAgent = agents['responder']
        
        # Run pipeline (reduced retrieval for speed)
        retrieved = retriever.retrieve(question=question, top_k=2)
        # Slightly higher budget for more complete reasoning
        reasoned = reasoner.reason(question=question, passages=retrieved, max_tokens=640)
        # Encourage precise, point-wise, compact formatting in the final answer
        formatting_note = (
            "\n\nFormatting requirements: Provide a precise, accurate answer in well-aligned markdown. "
            "Keep it to ≤ 8 bullet points, use short headings if needed, avoid redundancy, and keep it crisp."
        )
        final_answer = responder.respond(
            question=question,
            reasoning_summary=f"{reasoned}{formatting_note}",
            max_tokens=768
        )
        
        return {
            "question": question,
            "retrieved": retrieved,
            "reasoned": f"{reasoned}{formatting_note}",
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


def render_chat_sidebar():
    """Render the collapsible chat history sidebar"""
    with st.sidebar:
        # Toggle button for sidebar collapse
        if st.button("📋" if not st.session_state.sidebar_collapsed else "📋", key="sidebar_toggle"):
            st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
        
        if not st.session_state.sidebar_collapsed:
            st.header("💬 Chat History")
            
            # New Chat button
            if st.button("🆕 New Chat", type="primary", use_container_width=True):
                create_new_chat()
                st.rerun()
            
            st.markdown("---")
            
            # Chat list
            if st.session_state.chats:
                for chat_id, chat in st.session_state.chats.items():
                    # Create a unique key for each chat button
                    is_current = chat_id == st.session_state.current_chat_id
                    
                    # Chat button with title and timestamp
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(
                            chat['title'],
                            key=f"chat_{chat_id}",
                            use_container_width=True,
                            type="primary" if is_current else "secondary"
                        ):
                            switch_to_chat(chat_id)
                            st.rerun()
                    
                    with col2:
                        if st.button("🗑️", key=f"delete_{chat_id}", help="Delete chat"):
                            delete_chat(chat_id)
                            st.rerun()
                    
                    # Show message count
                    if chat['messages']:
                        st.caption(f"{len(chat['messages'])} messages")
            else:
                st.info("No chats yet. Start a conversation!")
            
            st.markdown("---")
            
            # Document upload section
            st.header("📄 Document Upload")
            st.markdown("Upload documents to add them to the knowledge base.")
            
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'txt', 'docx'],
                accept_multiple_files=True,
                help="Upload PDF, TXT, or DOCX files",
                key="file_uploader"
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


def render_chat_messages():
    """Render chat messages with proper styling"""
    if not st.session_state.current_chat_id or st.session_state.current_chat_id not in st.session_state.chats:
        return
    
    chat = st.session_state.chats[st.session_state.current_chat_id]
    
    # Display chat messages
    for message in chat['messages']:
        with st.chat_message(message["role"]):
            # Use markdown for better formatting
            st.markdown(message["content"])
    
    # Auto-scroll to bottom
    st.markdown(
        """
        <script>
            // Auto-scroll to bottom of chat
            window.scrollTo(0, document.body.scrollHeight);
        </script>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(
        page_title="Multi-Agentic RAG Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling with system color theme support
    st.markdown("""
    <style>
    /* Base chat bubble styling */
    [data-testid="chatMessage"] {
        border-radius: 12px !important;
        padding: 10px 12px !important;
        margin: 10px 0 !important;
        border: 1px solid transparent;
    }
    /* Hide avatars/icons for a clean alignment */
    [data-testid="chatMessage"] [data-testid*="avatar"],
    [data-testid="chatMessage"] svg,
    [data-testid="chatMessage"] img { display: none !important; }

    /* Light theme (system preference) */
    @media (prefers-color-scheme: light) {
      html, body, .stApp { color: #0f172a; background: #ffffff; }
      [data-testid="chatMessage"] { background: #f8fafc; color: #0f172a; border-color: #e2e8f0; }
      /* Code blocks */
      pre, code { background: #f1f5f9 !important; color: #0f172a !important; }
      /* Inputs & buttons */
      .stTextInput > div > div > input { color: #0f172a; }
      .stButton > button { border-radius: 20px; }
    }

    /* Dark theme (system preference) */
    @media (prefers-color-scheme: dark) {
      html, body, .stApp { color: #e5e7eb; background: #0b0f19; }
      [data-testid="chatMessage"] { background: #111827; color: #e5e7eb; border-color: #1f2937; }
      /* Code blocks */
      pre, code { background: #0f172a !important; color: #e5e7eb !important; }
      /* Inputs & buttons */
      .stTextInput > div > div > input { color: #e5e7eb; }
      .stButton > button { border-radius: 20px; }
    }
    
    /* Print stylesheet: show only chat conversation reliably */
    @media print {
      @page { size: A4; margin: 12mm; }
      html, body, .stApp { background: #ffffff !important; color: #000000 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
      .block-container { max-width: 100% !important; padding: 0 0 !important; }
      /* Hide Streamlit chrome */
      header, footer, aside, nav,
      [data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="stSidebarContent"], .stSidebar,
      [data-testid="stToolbar"], [data-testid="stBottomBlockContainer"], [data-testid="stChatInput"],
      .stFileUploader, .stTextInput, .stTextArea, .stButton { display: none !important; }
      /* Show only the print-chat wrapper among main content */
      .block-container > *:not(.print-chat) { display: none !important; }
      .print-chat { display: block !important; }
      /* Style chat bubbles for paper (support both possible test IDs) */
      .print-chat [data-testid="chatMessage"],
      .print-chat [data-testid="stChatMessage"] {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        margin: 8px 0 !important;
        padding: 10px 12px !important;
        page-break-inside: avoid;
      }
      .print-chat [data-testid*="avatar"], .print-chat img, .print-chat svg { display: none !important; }
      /* Code blocks readable on paper */
      .print-chat pre, .print-chat code { background: #f3f4f6 !important; color: #111827 !important; border: 1px solid #e5e7eb !important; }
      /* Headings spacing */
      .print-chat h1, .print-chat h2, .print-chat h3, .print-chat h4, .print-chat h5, .print-chat h6 { margin: 10px 0 6px 0 !important; page-break-after: avoid; }
    }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Ensure we have at least one chat
    if not st.session_state.chats:
        create_new_chat()
    
    # Render sidebar
    render_chat_sidebar()
    
    # Main chat area
    if st.session_state.current_chat_id and st.session_state.current_chat_id in st.session_state.chats:
        chat = st.session_state.chats[st.session_state.current_chat_id]
        
        # Chat header
        st.header(f"💬 {chat['title']}")
        
        # Chat messages container (wrapped for printing)
        st.markdown("<div class='print-chat'>", unsafe_allow_html=True)
        chat_container = st.container()
        
        with chat_container:
            render_chat_messages()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Input area
        st.markdown("---")
        
        # Question input with better styling
        question = st.text_area(
            "Ask a question:",
            placeholder="What would you like to know?",
            height=100,
            key="question_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("🚀 Send", type="primary", disabled=not question.strip())
        
        with col2:
            if st.button("🗑️ Clear Chat"):
                chat['messages'] = []
                update_chat_title(st.session_state.current_chat_id)
                st.rerun()
        
        # Process question
        if send_button:
            if not st.session_state.db_initialized:
                st.warning("⚠️ Please upload and ingest documents first!")
            else:
                with st.spinner("Processing your question..."):
                    try:
                        sub_questions = split_user_questions(question)
                        if len(sub_questions) <= 1:
                            result = run_rag_pipeline(question)
                            # Add messages to current chat
                            chat['messages'].append({
                                "role": "user",
                                "content": question
                            })
                            chat['messages'].append({
                                "role": "assistant", 
                                "content": result["answer"]
                            })
                        else:
                            # Process each sub-question and combine
                            combined_markdown = ["### ✅ Answers (Multiple Questions)"]
                            chat['messages'].append({
                                "role": "user",
                                "content": "\n".join(f"{i+1}. {q}" for i, q in enumerate(sub_questions))
                            })
                            for i, q in enumerate(sub_questions, start=1):
                                res = run_rag_pipeline(q)
                                combined_markdown.append(f"\n### Q{i}: {q}")
                                combined_markdown.append(res["answer"])  # already well-formatted
                            chat['messages'].append({
                                "role": "assistant",
                                "content": "\n\n".join(combined_markdown)
                            })
                        
                        # Update chat title and timestamp
                        update_chat_title(st.session_state.current_chat_id)
                        
                        # Rerun to show new messages
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error processing question: {str(e)}")
    
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
