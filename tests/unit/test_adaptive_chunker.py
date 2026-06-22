"""Unit tests for the AdaptiveSemanticChunker."""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.documents import Document

from asc.chunker.adaptive_chunker import AdaptiveSemanticChunker, ChunkMetadata
from asc.chunker.perplexity_scorer import PerplexityScorer
from asc.chunker.boundary_detector import BoundaryDetector, BoundaryDetectorConfig


# Helper to instantiate chunker with mocked components
def get_mocked_chunker(mock_scorer) -> AdaptiveSemanticChunker:
    config = BoundaryDetectorConfig(
        z_score_threshold=1.5,
        lag_window=5,
        min_chunk_sentences=2,
        smoothing_window=5,
        smoothing_poly_order=2,
        require_local_maximum=True
    )
    detector = BoundaryDetector(config)
    
    return AdaptiveSemanticChunker(
        perplexity_scorer=mock_scorer,
        boundary_detector=detector,
        sentence_overlap=1
    )


# 1. chunk_text: returns list[Document], metadata fields present
@pytest.mark.asyncio
async def test_chunk_text_basic(sample_sentences, mock_perplexity_scorer) -> None:
    chunker = get_mocked_chunker(mock_perplexity_scorer)
    text = " ".join(sample_sentences)
    
    docs = await chunker.chunk_text(text, {"source": "wiki_doc.txt"})
    
    assert len(docs) > 1
    for d in docs:
        assert isinstance(d, Document)
        # Check metadata fields presence
        meta = d.metadata
        assert "chunk_index" in meta
        assert "total_chunks" in meta
        assert "sentence_start" in meta
        assert "sentence_end" in meta
        assert "sentence_count" in meta
        assert "avg_perplexity" in meta
        assert "max_perplexity" in meta
        assert "min_perplexity" in meta
        assert "boundary_z_score" in meta
        assert meta["source"] == "wiki_doc.txt"


# 2. sentence_overlap: overlapping sentences appear in adjacent chunks
@pytest.mark.asyncio
async def test_sentence_overlap(sample_sentences, mock_perplexity_scorer) -> None:
    chunker = get_mocked_chunker(mock_perplexity_scorer)
    chunker.sentence_overlap = 1
    
    text = " ".join(sample_sentences)
    docs = await chunker.chunk_text(text)
    
    assert len(docs) >= 2
    # Verify that the last sentence of the first chunk is prepended as the first sentence of the second chunk
    # Get NLTK tokenized sentences of each chunk to compare
    import nltk
    first_chunk_sents = nltk.sent_tokenize(docs[0].page_content)
    second_chunk_sents = nltk.sent_tokenize(docs[1].page_content)
    
    overlap_sentence = first_chunk_sents[-1]
    assert second_chunk_sents[0] == overlap_sentence


# 3. empty text: returns []
@pytest.mark.asyncio
async def test_empty_text_edge_case(mock_perplexity_scorer) -> None:
    chunker = get_mocked_chunker(mock_perplexity_scorer)
    assert await chunker.chunk_text("") == []
    assert await chunker.chunk_text("   ") == []


# 4. single sentence: returns 1 chunk
@pytest.mark.asyncio
async def test_single_sentence_edge_case(mock_perplexity_scorer) -> None:
    chunker = get_mocked_chunker(mock_perplexity_scorer)
    single_sent = "Only one sentence here."
    
    docs = await chunker.chunk_text(single_sent)
    assert len(docs) == 1
    assert docs[0].page_content == single_sent


# 5. ChunkMetadata: all fields correctly populated from mock data
@pytest.mark.asyncio
async def test_chunk_metadata_populated(sample_sentences, mock_perplexity_scorer) -> None:
    chunker = get_mocked_chunker(mock_perplexity_scorer)
    text = " ".join(sample_sentences)
    
    docs = await chunker.chunk_text(text)
    
    meta = docs[0].metadata
    # Check that metadata matches expected ranges and calculations
    assert meta["chunk_index"] == 0
    assert meta["total_chunks"] == len(docs)
    assert meta["sentence_start"] == 0
    assert meta["sentence_count"] > 0
    assert meta["avg_perplexity"] > 0.0


# 6. compare_with_fixed: returns both methods' chunks
def test_compare_with_fixed(sample_sentences, mock_perplexity_scorer) -> None:
    chunker = get_mocked_chunker(mock_perplexity_scorer)
    text = " ".join(sample_sentences)
    
    res = chunker.compare_with_fixed(text, recursive_chunk_size=100)
    
    assert "asc" in res
    assert "recursive" in res
    assert len(res["asc"]) > 0
    assert len(res["recursive"]) > 0
    
    # Assert correct chunk types mapped in metadata
    assert res["asc"][0].metadata["chunk_type"] == "semantic"
    assert res["recursive"][0].metadata["chunk_type"] == "fixed"
