"""Integration tests for the LangGraph RAG pipeline."""

import pytest
import socket
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
from asc.vectorstore.chroma_store import ASCVectorStore
from asc.retrieval.retriever import AdaptiveSemanticRetriever
from asc.retrieval.rag_pipeline import run_rag

def is_ollama_running() -> bool:
    """Checks if Ollama is running on the local port 11434."""
    try:
        with socket.create_connection(("localhost", 11434), timeout=1.0):
            return True
    except OSError:
        return False

@pytest.fixture
def rag_components(temp_chroma_dir):
    """Initializes vector store, retriever, and LLM for integration testing."""
    if not is_ollama_running():
        pytest.skip("Ollama is not running on localhost:11434")
        
    # Initialize ASCVectorStore
    vector_store = ASCVectorStore(
        collection_name="test_rag_integration_collection",
        persist_directory=temp_chroma_dir,
        embedding_model="nomic-embed-text"
    )
    vector_store.reset()
    
    # Initialize Retriever
    retriever = AdaptiveSemanticRetriever(
        vector_store=vector_store,
        k=2,
        coherence_rerank=True,
        boundary_expand=False
    )
    
    # Initialize LLM
    llm = OllamaLLM(model="llama3.2:3b", base_url="http://localhost:11434")
    
    return vector_store, retriever, llm

@pytest.mark.integration
@pytest.mark.asyncio
async def test_rag_pipeline_integration(rag_components) -> None:
    """Tests the full RAG pipeline index -> retrieve -> answer process on a 3-document corpus."""
    vector_store, retriever, llm = rag_components
    
    # 3-document corpus
    doc1 = Document(
        page_content="Adaptive Semantic Chunking (ASC) is a method that uses causal language model perplexity to identify boundaries in text.",
        metadata={"source": "asc_info.txt", "chunk_index": 0, "avg_perplexity": 2.5, "boundary_z_score": 0.5}
    )
    doc2 = Document(
        page_content="Photosynthesis converts carbon dioxide and water into oxygen and sugars.",
        metadata={"source": "bio_info.txt", "chunk_index": 0, "avg_perplexity": 1.5, "boundary_z_score": 0.2}
    )
    doc3 = Document(
        page_content="Quantum computing uses qubits to perform complex operations.",
        metadata={"source": "qc_info.txt", "chunk_index": 0, "avg_perplexity": 3.0, "boundary_z_score": 0.8}
    )
    
    # 1. Full Index
    await vector_store.add_documents([doc1, doc2, doc3])
    
    # 2. Test Answerable Question
    # Query about ASC
    question = "What does Adaptive Semantic Chunking use to identify boundaries?"
    result = await run_rag(question, retriever, llm)
    
    assert result is not None
    assert "answer" in result
    assert "entailment_score" in result
    
    answer = result["answer"].lower()
    # The answer should be generated and relevant
    assert len(answer) > 0
    assert "perplexity" in answer or "causal" in answer or "surprisal" in answer or "boundaries" in answer
    
    # Entailment score >= 0.5 for answerable questions
    assert result["entailment_score"] >= 0.5
    
    # 3. Test Unanswerable Question
    # Query completely unrelated to the indexed documents
    unrelated_question = "What is the capital of France?"
    result_unrelated = await run_rag(unrelated_question, retriever, llm)
    
    assert result_unrelated is not None
    assert "answer" in result_unrelated
    
    unrelated_answer = result_unrelated["answer"].lower()
    # "I don't know" response for unanswerable questions
    assert "don't know" in unrelated_answer or "do not know" in unrelated_answer or "no relevant context" in unrelated_answer or "unknown" in unrelated_answer
