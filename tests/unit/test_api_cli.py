"""Unit tests for the FastAPI API and Typer CLI."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from asc.api.app import app
from main import app as cli_app

# Create test client for FastAPI
client = TestClient(app)
runner = CliRunner()


@pytest.fixture
def mock_chunker():
    """Mock the AdaptiveSemanticChunker to prevent actual LLM/Ollama calls."""
    with patch("asc.api.app.AdaptiveSemanticChunker") as mock_cls:
        mock_instance = MagicMock()
        mock_instance._score_sentences = AsyncMock(return_value=[1.0, 2.0, 5.0, 1.0, 1.0])
        # Return mock document chunks
        from langchain_core.documents import Document
        mock_docs = [
            Document(page_content="Chunk 1", metadata={"chunk_index": 0, "avg_perplexity": 2.0}),
            Document(page_content="Chunk 2", metadata={"chunk_index": 1, "avg_perplexity": 1.5})
        ]
        mock_instance._to_documents.return_value = mock_docs
        mock_instance.sentence_overlap = 1
        
        # Mock boundary detector return value
        mock_instance.detector.detect_boundaries.return_value = ([0, 3, 5], {"z_scores": [0,0,3,0,0]})
        
        mock_cls.return_value = mock_instance
        yield mock_instance


def test_api_chunk_endpoint(mock_chunker) -> None:
    """Verifies that the POST /chunk endpoint segments text correctly."""
    payload = {
        "text": "This is sentence one. This is sentence two. Surprise spike! This is sentence four. This is sentence five.",
        "source": "api_test.txt",
        "visualize": False
    }
    
    response = client.post("/chunk", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["total_chunks"] == 2
    assert len(data["chunks"]) == 2
    assert data["chunks"][0]["text"] == "Chunk 1"
    assert data["avg_perplexity"] == 2.0


def test_api_health_endpoint() -> None:
    """Verifies that the health endpoint returns correct status schema."""
    # Mock tags request
    with patch("httpx.AsyncClient.get") as mock_get, patch.dict("os.environ", {"LLM_PROVIDER": "ollama"}):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"models": [{"name": "llama3.2:3b"}]}
        mock_get.return_value = mock_response
        
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["ollama"] is True
        assert "llama3.2:3b" in data["models_loaded"]


def test_cli_help() -> None:
    """Verifies that the Typer CLI shows the help message correctly."""
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "main.py", "--help"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0
    assert "Adaptive Semantic Chunking" in result.stdout
    assert "chunk" in result.stdout
    assert "index" in result.stdout
    assert "query" in result.stdout
    assert "serve" in result.stdout

