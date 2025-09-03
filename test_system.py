#!/usr/bin/env python3
"""
Test script for the Multi-Agentic RAG system
"""

def test_imports():
    """Test if all modules can be imported"""
    try:
        from agents.retriever import RetrieverAgent
        from agents.reasoner import ReasonerAgent
        from agents.responder import ResponderAgent
        from db_setup import ensure_db_initialized, ingest_uploaded_documents
        from streamlit_app import main, process_uploaded_files, run_rag_pipeline
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_database():
    """Test database initialization"""
    try:
        from db_setup import ensure_db_initialized
        ensure_db_initialized()
        print("✅ Database initialization successful")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def test_agents():
    """Test agent initialization"""
    try:
        from agents.retriever import RetrieverAgent
        from agents.reasoner import ReasonerAgent
        from agents.responder import ResponderAgent
        
        retriever = RetrieverAgent()
        reasoner = ReasonerAgent(model="mistral")
        responder = ResponderAgent(model="llama2")
        
        print("✅ All agents initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Multi-Agentic RAG System...")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Database Setup", test_database),
        ("Agent Initialization", test_agents),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"   ⚠️  {test_name} failed")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 To run the Streamlit app:")
        print("   streamlit run streamlit_app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n💡 Common solutions:")
        print("   1. Ensure Ollama is running: ollama serve")
        print("   2. Pull required models: ollama pull mistral llama2 nomic-embed-text")
        print("   3. Install dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
