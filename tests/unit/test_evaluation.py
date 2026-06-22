"""Unit tests for the evaluation metrics and benchmark."""

import pytest
import math
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.documents import Document

from asc.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    chunk_coherence_score,
    boundary_quality_score,
    chunk_size_stats
)
from asc.evaluation.benchmark import ASCBenchmark, BenchmarkResult


def test_retrieval_metrics() -> None:
    """Verifies retrieval precision, recall, and reciprocal rank calculations."""
    retrieved = ["doc2", "doc5", "doc1", "doc3", "doc4"]
    relevant = {"doc1", "doc2"} # doc2 is at idx 0, doc1 is at idx 2
    
    # Precision@1: Top-1 is ["doc2"]. Hit! P@1 = 1.0
    assert precision_at_k(retrieved, relevant, k=1) == 1.0
    # Precision@3: Top-3 is ["doc2", "doc5", "doc1"]. Hits: 2. P@3 = 2/3
    assert abs(precision_at_k(retrieved, relevant, k=3) - 2.0/3.0) < 1e-5
    
    # Recall@1: Hits: 1 out of 2. R@1 = 0.5
    assert recall_at_k(retrieved, relevant, k=1) == 0.5
    # Recall@3: Hits: 2 out of 2. R@3 = 1.0
    assert recall_at_k(retrieved, relevant, k=3) == 1.0
    
    # Reciprocal Rank: First relevant is at idx 0. RR = 1/1 = 1.0
    assert mean_reciprocal_rank(retrieved, relevant) == 1.0
    
    # Reciprocal Rank where first hit is at idx 2:
    retrieved_miss = ["doc5", "doc3", "doc1"]
    # doc1 is at index 2 -> RR = 1/3
    assert abs(mean_reciprocal_rank(retrieved_miss, relevant) - 1.0/3.0) < 1e-5


def test_chunk_size_stats() -> None:
    """Verifies chunk size statistics (tokens, std, cv)."""
    chunks = [
        Document(page_content="one two three"),
        Document(page_content="one two three four five")
    ]
    # token count will split on spaces in fallback (size 3 and 5)
    # mean = 4, std = 1, cv = 1/4 = 0.25
    stats = chunk_size_stats(chunks)
    
    assert stats["mean_tokens"] == 4.0
    assert stats["std_tokens"] == 1.0
    assert stats["min_tokens"] == 3
    assert stats["max_tokens"] == 5
    assert stats["cv"] == 0.25


def test_chunk_coherence_score() -> None:
    """Verifies chunk coherence pairwise similarity calculations."""
    mock_embedder = MagicMock()
    # Mock embeddings: perfect overlap
    mock_embedder.embed_documents.return_value = [[1.0, 0.0], [1.0, 0.0]]
    
    sentences = ["Sentence 1.", "Sentence 2."]
    score = chunk_coherence_score(sentences, mock_embedder)
    # Pairwise similarity of identical vectors = 1.0
    assert score == 1.0


def test_boundary_quality_score() -> None:
    """Verifies boundary quality distinctness calculation."""
    mock_embedder = MagicMock()
    # Mock A: [1.0, 0.0], Mock B: [0.0, 1.0] (orthogonal, cos sim = 0)
    mock_embedder.embed_documents.side_effect = [[[1.0, 0.0]], [[0.0, 1.0]]]
    
    score = boundary_quality_score(["A"], ["B"], mock_embedder)
    # Distinctness = 1.0 - cos_sim = 1.0 - 0.0 = 1.0
    assert score == 1.0


@pytest.mark.asyncio
async def test_asc_benchmark_report_generation() -> None:
    """Verifies ASCBenchmark report generation and formatting."""
    benchmark = ASCBenchmark()
    
    res1 = BenchmarkResult(
        method_name="ASC",
        precision_at_k={3: 0.9},
        recall_at_k={3: 0.8},
        mrr=0.85,
        avg_coherence=0.92,
        avg_boundary_quality=0.88,
        chunk_size_stats={"mean_tokens": 120.0, "cv": 0.1},
        total_chunks=15,
        runtime_seconds=5.2
    )
    res2 = BenchmarkResult(
        method_name="Recursive_512",
        precision_at_k={3: 0.7},
        recall_at_k={3: 0.6},
        mrr=0.65,
        avg_coherence=0.81,
        avg_boundary_quality=0.72,
        chunk_size_stats={"mean_tokens": 150.0, "cv": 0.4},
        total_chunks=12,
        runtime_seconds=0.8
    )
    
    # Pre-populate query score lists for significance report
    benchmark._query_scores["ASC"] = {"mrr": [1.0, 0.7], "precision_3": [1.0, 0.8], "recall_3": [1.0, 0.6]}
    benchmark._query_scores["Recursive_512"] = {"mrr": [0.6, 0.7], "precision_3": [0.6, 0.8], "recall_3": [0.6, 0.6]}
    
    report = benchmark.generate_report([res1, res2])
    
    # Assert ASC win is highlighted in bold
    assert "**0.9000**" in report # P@3 ASC win
    assert "**0.8500**" in report # MRR ASC win
    # Assert method names are printed
    assert "ASC" in report
    assert "Recursive_512" in report
