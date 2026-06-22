"""Integration tests for the AdaptiveSemanticChunker pipeline."""

import pytest
from unittest.mock import AsyncMock, patch
from langchain_core.documents import Document

from asc.chunker.adaptive_chunker import AdaptiveSemanticChunker


@pytest.mark.asyncio
async def test_adaptive_chunker_document_pipeline() -> None:
    """Verifies end-to-end document transformation and metadata formatting."""
    # Instantiates chunker with customized lag/window to fit the short test document
    chunker = AdaptiveSemanticChunker(
        model="test-model",
        min_sentences_per_chunk=2,
        z_threshold=1.0,
        rolling_lag=2,
        savgol_window=3,
        savgol_polyorder=1,
        sentence_overlap=0,
    )
    
    # Mock scorer return values for 5 sentences
    # Spike at index 2, with graded slope to survive smoothing
    mock_ppls = [1.1, 1.5, 4.5, 1.5, 1.1]
    
    with patch.object(chunker, "_score_sentences", AsyncMock(return_value=mock_ppls)):
        text = (
            "This is sentence one. "
            "This is sentence two. "
            "This is surprise sentence three. "
            "This is sentence four. "
            "This is sentence five."
        )
        doc = Document(page_content=text, metadata={"source": "test_doc.txt"})
        
        # Split documents
        chunked_docs = await chunker.transform_documents_async([doc])
        
        # Verify chunks created
        assert len(chunked_docs) > 1
        
        # Verify metadata schemas
        for i, chunk in enumerate(chunked_docs):
            meta = chunk.metadata
            assert meta["source"] == "test_doc.txt"
            assert meta["source_doc_index"] == 0
            assert meta["chunk_index"] == i
            assert meta["is_chunked"] is True
            assert "sentence_count" in meta
            assert "mean_perplexity" in meta
            
        # Re-join page content and verify original information is preserved
        combined_text = " ".join([c.page_content for c in chunked_docs])
        # NLTK joining might change spaces slightly, clean up whitespace to compare
        assert "".join(combined_text.split()) == "".join(text.split())
