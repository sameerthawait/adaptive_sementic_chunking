"""Unit tests for ASCVectorStore and AdaptiveSemanticRetriever."""

import pytest
import math
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from langchain_core.documents import Document

from asc.vectorstore.chroma_store import ASCVectorStore
from asc.retrieval.retriever import AdaptiveSemanticRetriever


@pytest.fixture
def mock_chromadb():
    """Mocks chromadb client and collection."""
    with patch("chromadb.PersistentClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client
        yield mock_client, mock_collection


@pytest.mark.asyncio
async def test_vectorstore_add_documents(mock_chromadb) -> None:
    """Verifies that add_documents embeds texts and upserts into Chroma."""
    _, mock_collection = mock_chromadb
    
    store = ASCVectorStore(
        collection_name="test_col",
        persist_directory="./test_dir"
    )
    
    # Mock embedder calls
    store.embedder.aembed_documents = AsyncMock(return_value=[[0.1]*768, [0.2]*768])
    
    docs = [
        Document(page_content="Content 1", metadata={"source": "doc1.txt", "chunk_index": 0}),
        Document(page_content="Content 2", metadata={"source": "doc1.txt", "chunk_index": 1})
    ]
    
    ids = await store.add_documents(docs, batch_size=2)
    
    assert len(ids) == 2
    # Verify deterministic ID format (md5 of source + '_' + chunk_index)
    assert "_" in ids[0]
    assert mock_collection.upsert.called


@pytest.mark.asyncio
async def test_vectorstore_similarity_search(mock_chromadb) -> None:
    """Verifies similarity search query and distance translation to cosine similarity."""
    _, mock_collection = mock_chromadb
    
    store = ASCVectorStore(
        collection_name="test_col",
        persist_directory="./test_dir"
    )
    
    store.embedder.aembed_query = AsyncMock(return_value=[0.1]*768)
    
    # Mock collection query return value
    # Chroma returns lists of lists
    mock_collection.query.return_value = {
        "ids": [["id1", "id2"]],
        "distances": [[0.2, 0.8]], # 1 - similarity
        "metadatas": [[{"source": "doc1.txt"}, {"source": "doc1.txt"}]],
        "documents": [["text1", "text2"]]
    }
    
    res = await store.similarity_search("test query", k=2)
    
    assert len(res) == 2
    # doc 1: distance 0.2 -> similarity 0.8
    assert abs(res[0][1] - 0.8) < 1e-5
    # doc 2: distance 0.8 -> similarity 0.2
    assert abs(res[1][1] - 0.2) < 1e-5
    
    assert res[0][0].page_content == "text1"


def test_vectorstore_get_adjacent_chunks(mock_chromadb) -> None:
    """Verifies parsing of adjacent chunk IDs."""
    _, mock_collection = mock_chromadb
    
    store = ASCVectorStore(
        collection_name="test_col",
        persist_directory="./test_dir"
    )
    
    mock_collection.get.return_value = {
        "ids": ["sourcehash_1", "sourcehash_3"],
        "documents": ["Adjacent Prev", "Adjacent Next"],
        "metadatas": [{"chunk_index": 1}, {"chunk_index": 3}]
    }
    
    adjacent = store.get_adjacent_chunks("sourcehash_2", window=1)
    
    assert len(adjacent) == 2
    # Must query target IDs: sourcehash_1 and sourcehash_3
    mock_collection.get.assert_called_once_with(ids=["sourcehash_1", "sourcehash_3"])


def test_coherence_score() -> None:
    """Verifies formula of the coherence score."""
    retriever = AdaptiveSemanticRetriever(vector_store=MagicMock())
    
    q_emb = [1.0, 0.0]
    d_emb = [1.0, 0.0] # perfect cosine similarity = 1.0
    
    # Cos_sim * (1 / log(1 + avg_perplexity))
    # For perplexity 1.1, divisor is log(2.1) = 0.74193. Score = 1.0 / 0.74193 = 1.3478
    score = retriever._coherence_score(q_emb, d_emb, avg_perplexity=1.1)
    assert abs(score - (1.0 / math.log(2.1))) < 1e-5


@pytest.mark.asyncio
async def test_retriever_pipeline(mock_chromadb) -> None:
    """Verifies the retriever pipeline: over-retrieve, re-rank, expand."""
    _, mock_collection = mock_chromadb
    
    store = ASCVectorStore(
        collection_name="test_col",
        persist_directory="./test_dir"
    )
    
    # Over-retrieved search results: 4 items (k=2 * 2)
    doc1 = Document(page_content="Text 1", metadata={"id": "src_1", "avg_perplexity": 2.0, "boundary_z_score": 1.0, "chunk_index": 1})
    doc2 = Document(page_content="Text 2", metadata={"id": "src_2", "avg_perplexity": 10.0, "boundary_z_score": 3.0, "chunk_index": 2}) # high z-score -> trigger expand!
    doc3 = Document(page_content="Text 3", metadata={"id": "src_3", "avg_perplexity": 50.0, "boundary_z_score": 0.5, "chunk_index": 3})
    doc4 = Document(page_content="Text 4", metadata={"id": "src_4", "avg_perplexity": 100.0, "boundary_z_score": 0.2, "chunk_index": 4})
    
    store.similarity_search = AsyncMock(return_value=[
        (doc1, 0.9), (doc2, 0.85), (doc3, 0.8), (doc4, 0.7)
    ])
    
    store.embedder.aembed_query = AsyncMock(return_value=[1.0, 0.0])
    
    retriever = AdaptiveSemanticRetriever(
        vector_store=store,
        k=2,
        coherence_rerank=True,
        boundary_expand=True,
        expand_threshold=2.5
    )
    
    # Mock document embeddings retrieval from db
    def mock_get_embedding(doc):
        if doc.metadata["id"] == "src_1":
            return [1.0, 0.0] # perfect cos sim = 1.0
        return [0.9, 0.1]
        
    retriever._get_document_embedding = MagicMock(side_effect=mock_get_embedding)
    
    # Mock adjacent chunks retrieval for doc2 (high z-score = 3.0 > 2.5)
    doc2_adjacent = Document(page_content="Text 2 Adjacent", metadata={"id": "src_3", "chunk_index": 3})
    store.get_adjacent_chunks = MagicMock(return_value=[doc2_adjacent])
    
    retrieved_docs = await retriever._aget_relevant_documents("query")
    
    # Top k is 2. Should keep top 2 re-ranked.
    # Doc1 score: 1.0 * (1/log(3.0)) = 1.0 / 1.098 = 0.91
    # Doc2 score: 0.9 * (1/log(11.0)) = 0.9 / 2.397 = 0.37
    # Doc3 score: 0.9 * (1/log(51.0)) = 0.9 / 3.93 = 0.22
    # So top 2 are indeed Doc1 and Doc2.
    # Since Doc2 has boundary_z_score 3.0 > 2.5, it expands to doc2_adjacent ("src_3").
    # The final set should include Doc1, Doc2, and doc2_adjacent.
    assert len(retrieved_docs) == 3
    assert retrieved_docs[0].metadata["id"] == "src_1"
    assert retrieved_docs[1].metadata["id"] == "src_2"
    assert retrieved_docs[2].metadata["id"] == "src_3"
