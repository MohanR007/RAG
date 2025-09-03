## Multi-Agentic RAG with Ollama (Local Setup)

### Prerequisites
- Python 3.10+
- Ollama installed and running locally (`ollama serve`)
- Pull required models:
  ```bash
  ollama pull mistral
  ollama pull llama2
  ollama pull nomic-embed-text
  ```

### Install
```bash
# Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt:
# .venv\Scripts\activate.bat
# Linux/Mac:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Ingest sample docs into Chroma
```bash
python db_setup.py --rebuild
```

### Run

#### Command Line Interface
```bash
python main.py --question "What is this project about?"
```
Or start an interactive loop:
```bash
python main.py
```

#### Streamlit Web Interface
```bash
streamlit run streamlit_app.py
```
Then open your browser to `http://localhost:8501`

**Features:**
- 📄 Upload PDF, TXT, or DOCX files
- 💬 Ask questions with a chat interface
- 📋 View conversation history
- 🔍 Real-time document processing and ingestion

### Testing
```bash
# Test the complete system
python test_system.py
```

### Notes
- Uses ChromaDB persistent storage at `./chroma`.
- Embeddings via Ollama model `nomic-embed-text`.
- Agents use Ollama chat models (default: `mistral` for Reasoner, `llama2` for Responder).
- **Troubleshooting**: If you get connection errors, ensure Ollama is running with `ollama serve`.

