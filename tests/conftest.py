"""Common Pytest fixtures for Adaptive Semantic Chunking tests."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from langchain_core.documents import Document

# 1. sample_sentences: 15 sentences from Wikipedia-style paragraph with topic shift at sentence 8
@pytest.fixture
def sample_sentences() -> list[str]:
    return [
        "Artificial intelligence is intelligence demonstrated by machines.",
        "Early AI research focused on symbolic logic and expert heuristics.",
        "During the late 20th century, expert systems became highly popular.",
        "These systems relied on hand-coded rules to solve math problems.",
        "However, progress stalled during the periods known as AI winters.",
        "Researchers struggled to adapt expert rules to messy real-world data.",
        "This caused a lack of funding and interest in the field.",
        "Many projects were abandoned during these years of stagnation.",
        # Sentence 8: Thematic shift to molecular biology/genomics
        "Genomics is the study of the complete genetic mapping of organisms.",
        "Deoxyribonucleic acid holds the instructions for cellular blueprints.",
        "Modern genome sequencing allows researchers to decode genes in hours.",
        "Biologists use these sequences to identify disease-causing mutations.",
        "Computational tools are now widely used to model genetic arrays.",
        "This research has accelerated drug discovery and bio-engineering.",
        "Scientists can now analyze genomic structures in seconds."
    ]


# 2. sample_perplexity_scores: np.array with a clear spike at index 8
@pytest.fixture
def sample_perplexity_scores() -> np.ndarray:
    # A clear spike (55.0) at index 8, others are relatively flat
    return np.array([12.0, 11.5, 12.5, 11.0, 13.0, 12.0, 11.5, 12.0, 55.0, 13.0, 12.5, 12.0, 11.0, 12.5, 11.5], dtype=np.float32)


# 3. mock_ollama_response: factory for mocked Ollama API responses with fake logprobs
@pytest.fixture
def mock_ollama_response():
    def _factory(text: str, logprob_values: list[float] | None = None) -> dict:
        logprobs = []
        words = text.split()
        vals = logprob_values if logprob_values else [-1.0] * len(words)
        for w, v in zip(words, vals):
            logprobs.append({"token": w, "logprob": float(v)})
            
        return {
            "response": text,
            "logprobs": logprobs
        }
    return _factory


# 4. mock_perplexity_scorer: AsyncMock that returns sample_perplexity_scores
@pytest.fixture
def mock_perplexity_scorer(sample_perplexity_scores) -> MagicMock:
    scorer = MagicMock()
    # Mock score_sentences to return the sample scores
    scorer.score_sentences = AsyncMock(return_value=sample_perplexity_scores)
    # Mock the context window length parameter
    scorer.context_window = 3
    return scorer


# 5. temp_chroma_dir: tmp_path fixture for temporary ChromaDB
@pytest.fixture
def temp_chroma_dir(tmp_path) -> str:
    return str(tmp_path / "chromadb")
