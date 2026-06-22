"""Unit tests for the LangGraph RAG pipeline."""

import pytest
from unittest.mock import MagicMock, AsyncMock
from langchain_core.documents import Document

from asc.retrieval.rag_pipeline import (
    RagState,
    retrieve_node,
    grade_node,
    generate_node,
    verify_node,
    rewrite_node,
    build_rag_graph,
    run_rag
)


@pytest.mark.asyncio
async def test_retrieve_node() -> None:
    """Verifies that retrieve_node fetches relevant documents and updates both state schemas."""
    state = {
        "query": "test query",
        "question": "test query",
    }
    
    mock_retriever = MagicMock()
    mock_retriever._aget_relevant_documents = AsyncMock(return_value=[
        Document(page_content="Context passage", metadata={"source": "test.txt"})
    ])
    
    new_state = await retrieve_node(state, mock_retriever)
    
    assert "retrieved_docs" in new_state
    assert len(new_state["retrieved_docs"]) == 1
    # Verify backward compatibility key mapping
    assert "documents" in new_state
    assert len(new_state["documents"]) == 1
    assert new_state["retrieved_docs"][0].page_content == "Context passage"
    mock_retriever._aget_relevant_documents.assert_called_once_with("test query")


@pytest.mark.asyncio
async def test_grade_node_relevance() -> None:
    """Verifies that grade_node parses integer score and handles rewrite trigger."""
    state = {
        "query": "query text",
        "retrieved_docs": [Document(page_content="Doc content 1"), Document(page_content="Doc content 2")]
    }
    
    mock_llm = MagicMock()
    # Let first document be grade 8/10, second be grade 2/10
    # Mean grade = (0.8 + 0.2) / 2 = 0.5 >= 0.5 (needs_rewrite = False)
    mock_llm.ainvoke = AsyncMock(side_effect=["8", "2"])
    
    new_state = await grade_node(state, mock_llm)
    
    assert new_state["doc_grades"] == [0.8, 0.2]
    assert new_state["needs_rewrite"] is False


@pytest.mark.asyncio
async def test_grade_node_low_relevance_rewrite() -> None:
    """Verifies rewrite is flagged when mean relevance falls below 0.5."""
    state = {
        "query": "query text",
        "retrieved_docs": [Document(page_content="Doc content")]
    }
    
    mock_llm = MagicMock()
    # Grade 3/10 -> normalized 0.3 < 0.5 -> needs_rewrite = True
    mock_llm.ainvoke = AsyncMock(return_value="3")
    
    new_state = await grade_node(state, mock_llm)
    assert new_state["needs_rewrite"] is True


@pytest.mark.asyncio
async def test_generate_node_threshold() -> None:
    """Verifies generate_node filters context using 0.4 grade threshold."""
    state = {
        "query": "query text",
        "retrieved_docs": [
            Document(page_content="High Relevance Doc"), 
            Document(page_content="Low Relevance Doc")
        ],
        "doc_grades": [0.8, 0.3] # second doc under 0.4, should be excluded
    }
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value="Generative Answer.")
    
    new_state = await generate_node(state, mock_llm)
    
    assert new_state["answer"] == "Generative Answer."
    assert new_state["generation"] == "Generative Answer."
    
    # Assert query to llm only included top docs
    call_prompt = mock_llm.ainvoke.call_args[0][0]
    assert "High Relevance Doc" in call_prompt
    assert "Low Relevance Doc" not in call_prompt


@pytest.mark.asyncio
async def test_verify_node_entailment() -> None:
    """Verifies verify_node entailment checks and iteration limit triggers final state."""
    state1 = {
        "retrieved_docs": [Document(page_content="Context")],
        "answer": "Answer",
        "iteration": 1
    }
    
    mock_llm = MagicMock()
    # Entailment score: 0.85 >= 0.7 -> final is True
    mock_llm.ainvoke = AsyncMock(return_value="0.85")
    
    new_state1 = await verify_node(state1, mock_llm)
    assert new_state1["entailment_score"] == 0.85
    assert new_state1["final"] is True
    
    # Check iteration limit trigger: iteration = 3 -> final is True even if score is low
    state2 = {
        "retrieved_docs": [Document(page_content="Context")],
        "answer": "Answer",
        "iteration": 3
    }
    mock_llm.ainvoke = AsyncMock(return_value="0.2")
    new_state2 = await verify_node(state2, mock_llm)
    assert new_state2["final"] is True


@pytest.mark.asyncio
async def test_rewrite_node() -> None:
    """Verifies rewrite_node preserves iteration count and updates query."""
    state = {
        "query": "old query",
        "iteration": 2
    }
    
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value="better query")
    
    new_state = await rewrite_node(state, mock_llm)
    
    assert new_state["rewritten_query"] == "better query"
    assert new_state["question"] == "better query"
    assert new_state["iteration"] == 2


def test_build_rag_graph() -> None:
    """Verifies that build_rag_graph returns a compiled LangGraph workflow."""
    mock_retriever = MagicMock()
    mock_llm = MagicMock()
    
    graph = build_rag_graph(mock_retriever, mock_llm)
    assert graph is not None
