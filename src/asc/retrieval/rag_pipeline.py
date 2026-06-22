"""LangGraph Agentic Self-Correcting RAG pipeline.

Defines the state graph for retrieving, grading, generating, and validating answers.
"""

import json
import logging
import re
import math
import asyncio
import numpy as np
from typing import Any, Literal, Dict, List, Union
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledGraph

from asc.retrieval.retriever import AdaptiveSemanticRetriever

logger = logging.getLogger(__name__)


class RagState(TypedDict, total=False):
    """Represents the internal state of the agentic RAG graph."""
    # New state keys requested
    query: str
    rewritten_query: str | None
    retrieved_docs: list[Document]
    doc_grades: list[float]          # relevance score 0-1 per doc
    answer: str
    entailment_score: float          # 0-1, does answer follow from docs?
    needs_rewrite: bool
    iteration: int
    final: bool

    # Old state keys for backward compatibility
    question: str
    original_question: str
    documents: list[Document]
    generation: str
    steps: list[str]
    retry_count: int
    ollama_url: str
    model: str


# Nodes
async def retrieve_node(state: RagState, retriever: Any) -> RagState:
    """Uses rewritten_query if available, else original query."""
    query = state.get("rewritten_query") or state.get("query") or state.get("question")
    logger.info(f"--- RETRIEVE NODE --- (Query: '{query}')")
    
    # Retrieve documents using the async retriever call
    docs = await retriever._aget_relevant_documents(query)
    
    # Synchronize old and new state fields
    new_state = dict(state)
    new_state["retrieved_docs"] = docs
    new_state["documents"] = docs
    
    steps = list(state.get("steps", []))
    steps.append("retrieve")
    new_state["steps"] = steps
    
    # Increment loop iteration count on every retrieval
    iteration = state.get("iteration") or 0
    new_state["iteration"] = iteration + 1
    
    return new_state


async def grade_node(state: RagState, llm: Any) -> RagState:
    """
    For each retrieved doc, prompt LLM:
    "On a scale 0-10, how relevant is this passage to the query: {query}
     Passage: {doc.page_content}
     Return only a single integer."
    Normalize to 0-1. Store in doc_grades.
    Set needs_rewrite = (mean(doc_grades) < 0.5)
    """
    query = state.get("rewritten_query") or state.get("query") or state.get("question")
    docs = state.get("retrieved_docs") or []
    
    doc_grades = []
    logger.info(f"--- GRADE NODE --- (Grading {len(docs)} documents)")
    
    for doc in docs:
        prompt = (
            f"On a scale 0-10, how relevant is this passage to the query: {query}\n"
            f"Passage: {doc.page_content}\n"
            f"Return only a single integer."
        )
        try:
            raw_response = await llm.ainvoke(prompt)
            if hasattr(raw_response, "content"):
                raw_response = raw_response.content
            raw_response = str(raw_response).strip()
            
            # Find the first digits
            match = re.search(r'\d+', raw_response)
            grade_int = int(match.group()) if match else 0
        except Exception as e:
            logger.warning(f"Error grading document: {e}. Falling back to 5.")
            grade_int = 5
            
        grade = max(0.0, min(1.0, grade_int / 10.0))
        doc_grades.append(grade)
        logger.info(f"  Document relevance: {grade_int}/10 -> normalized score: {grade}")
        
    mean_grade = float(np.mean(doc_grades)) if doc_grades else 0.0
    needs_rewrite = mean_grade < 0.5
    
    logger.info(f"  Mean Grade: {mean_grade:.2f}. Needs rewrite: {needs_rewrite}")
    
    new_state = dict(state)
    new_state["doc_grades"] = doc_grades
    new_state["needs_rewrite"] = needs_rewrite
    
    steps = list(state.get("steps", []))
    steps.append("grade_documents")
    new_state["steps"] = steps
    
    return new_state


async def generate_node(state: RagState, llm: Any) -> RagState:
    """
    Prompt: "Using ONLY the following context, answer the question.
    If the context doesn't contain the answer, say 'I don't know.'
    Context: {top_docs}
    Question: {query}
    Answer:"
    Only use docs where doc_grades[i] >= 0.4.
    """
    query = state.get("rewritten_query") or state.get("query") or state.get("question")
    docs = state.get("retrieved_docs") or []
    grades = state.get("doc_grades") or [1.0] * len(docs)
    
    logger.info("--- GENERATE NODE ---")
    
    # Filter documents based on grade threshold >= 0.4
    filtered_docs = []
    for doc, grade in zip(docs, grades):
        if grade >= 0.4:
            filtered_docs.append(doc)
            
    top_docs = "\n\n".join([d.page_content for d in filtered_docs]) if filtered_docs else "No relevant context."
    
    prompt = (
        f"Using ONLY the following context, answer the question.\n"
        f"If the context doesn't contain the answer, say 'I don't know.'\n"
        f"Context: {top_docs}\n"
        f"Question: {query}\n"
        f"Answer:"
    )
    
    try:
        response = await llm.ainvoke(prompt)
        if hasattr(response, "content"):
            response = response.content
        answer = str(response).strip()
    except Exception as e:
        logger.error(f"Error in generate_node: {e}")
        answer = "I don't know."
        
    logger.info(f"  Generated Answer: '{answer[:60]}...'")
    
    new_state = dict(state)
    new_state["answer"] = answer
    new_state["generation"] = answer
    
    steps = list(state.get("steps", []))
    steps.append("generate")
    new_state["steps"] = steps
    
    return new_state


async def verify_node(state: RagState, llm: Any) -> RagState:
    """
    Entailment check prompt:
    "Does the following answer follow logically from the context?
    Context: {docs}
    Answer: {answer}
    Reply with only a float between 0.0 and 1.0."
    Set state.final = (entailment_score >= 0.7) OR (iteration >= 3)
    """
    docs = state.get("retrieved_docs") or []
    answer = state.get("answer") or ""
    iteration = state.get("iteration") or 0
    
    logger.info("--- VERIFY NODE ---")
    
    context = "\n\n".join([d.page_content for d in docs]) if docs else "No context."
    
    prompt = (
        f"Does the following answer follow logically from the context?\n"
        f"Context: {context}\n"
        f"Answer: {answer}\n"
        f"Reply with only a float between 0.0 and 1.0."
    )
    
    try:
        response = await llm.ainvoke(prompt)
        if hasattr(response, "content"):
            response = response.content
        raw_val = str(response).strip()
        
        # Parse float
        match = re.search(r'\d+\.\d+', raw_val)
        if match:
            entailment_score = float(match.group())
        else:
            match_int = re.search(r'\d+', raw_val)
            entailment_score = float(match_int.group()) if match_int else 0.5
    except Exception as e:
        logger.warning(f"Error parsing entailment score: {e}. Defaulting to 0.5")
        entailment_score = 0.5
        
    entailment_score = max(0.0, min(1.0, entailment_score))
    final = (entailment_score >= 0.7) or (iteration >= 3)
    
    logger.info(f"  Entailment Score: {entailment_score:.2f}. Final: {final} (Iteration: {iteration})")
    
    new_state = dict(state)
    new_state["entailment_score"] = entailment_score
    new_state["final"] = final
    
    steps = list(state.get("steps", []))
    steps.append("verify")
    new_state["steps"] = steps
    
    return new_state


async def rewrite_node(state: RagState, llm: Any) -> RagState:
    """
    Prompt: "The query '{query}' returned poor results.
    Rewrite it to be more specific and retrieval-friendly.
    Return only the rewritten query."
    Increment iteration.
    """
    query = state.get("query") or state.get("question")
    iteration = state.get("iteration") or 0
    
    logger.info(f"--- REWRITE NODE --- (Iteration {iteration})")
    
    prompt = (
        f"The query '{query}' returned poor results.\n"
        f"Rewrite it to be more specific and retrieval-friendly.\n"
        f"Return only the rewritten query."
    )
    
    try:
        response = await llm.ainvoke(prompt)
        if hasattr(response, "content"):
            response = response.content
        rewritten = str(response).strip()
    except Exception as e:
        logger.error(f"Error rewriting query: {e}")
        rewritten = query
        
    logger.info(f"  Rewritten Query: '{rewritten}'")
    
    new_state = dict(state)
    new_state["rewritten_query"] = rewritten
    new_state["question"] = rewritten
    new_state["iteration"] = iteration
    new_state["retry_count"] = iteration
    
    steps = list(state.get("steps", []))
    steps.append("rewrite_query")
    new_state["steps"] = steps
    
    return new_state


# Edge routing functions
def route_grade(state: RagState) -> str:
    iteration = state.get("iteration") or 0
    if iteration >= 3:
        return "generate"
    if state.get("needs_rewrite", False):
        return "rewrite"
    return "generate"


def route_verify(state: RagState) -> str:
    if state.get("final", False):
        return "end"
    return "retrieve"


# Graph Builder
def build_rag_graph(
    retriever: AdaptiveSemanticRetriever,
    llm: Any = None,
    checkpointer: Any = None
) -> CompiledGraph:
    """
    Builds the state graph.
    
    Edges:
    START → retrieve → grade
    grade → generate (if not needs_rewrite) OR rewrite (if needs_rewrite)
    generate → verify
    verify → END (if final) OR retrieve (if not final, loop)
    rewrite → retrieve
    """
    if llm is None:
        from asc.utils.model_factory import get_llm_from_env
        llm = get_llm_from_env()

    # Initialize checkpointer if None for interrupt capability
    if checkpointer is None:
        try:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        except ImportError:
            try:
                from langgraph.checkpoint import MemorySaver
                checkpointer = MemorySaver()
            except ImportError:
                checkpointer = None

    async def retrieve_wrapper(state: RagState) -> RagState:
        return await retrieve_node(state, retriever)

    async def grade_wrapper(state: RagState) -> RagState:
        return await grade_node(state, llm)

    async def generate_wrapper(state: RagState) -> RagState:
        return await generate_node(state, llm)

    async def verify_wrapper(state: RagState) -> RagState:
        return await verify_node(state, llm)

    async def rewrite_wrapper(state: RagState) -> RagState:
        return await rewrite_node(state, llm)

    workflow = StateGraph(RagState)
    
    # Bind dependencies to node functions
    workflow.add_node("retrieve", retrieve_wrapper)
    workflow.add_node("grade", grade_wrapper)
    workflow.add_node("generate", generate_wrapper)
    workflow.add_node("verify", verify_wrapper)
    workflow.add_node("rewrite", rewrite_wrapper)
    
    workflow.set_entry_point("retrieve")
    
    workflow.add_edge("retrieve", "grade")
    
    workflow.add_conditional_edges(
        "grade",
        route_grade,
        {
            "generate": "generate",
            "rewrite": "rewrite",
        }
    )
    
    workflow.add_edge("generate", "verify")
    workflow.add_edge("rewrite", "retrieve")
    
    workflow.add_conditional_edges(
        "verify",
        route_verify,
        {
            "end": END,
            "retrieve": "retrieve",
        }
    )
    
    return workflow.compile(
        checkpointer=checkpointer,
        interrupt_before=["generate"]
    )


# Convenience run function
async def run_rag(
    question: str,
    retriever: AdaptiveSemanticRetriever,
    llm: Any,
    stream: bool = False
) -> Union[dict, Any]:
    """
    Returns: {answer, entailment_score, iterations, sources, query_rewrites}
    If stream=True, yields state updates as they happen.
    """
    graph = build_rag_graph(retriever, llm)
    
    initial_state = {
        "query": question,
        "rewritten_query": None,
        "retrieved_docs": [],
        "doc_grades": [],
        "answer": "",
        "entailment_score": 0.0,
        "needs_rewrite": False,
        "iteration": 0,
        "final": False,
        "question": question,
        "original_question": question,
        "documents": [],
        "generation": "",
        "steps": [],
        "retry_count": 0,
    }
    
    config = {"configurable": {"thread_id": "rag_thread"}}
    
    if stream:
        async def _stream_generator():
            # First execution segment (stops at retrieve/grade, or generate if reached)
            async for event in graph.astream(initial_state, config=config):
                yield event
                
            # If hit generate interrupt, resume automatically
            while True:
                curr_state = await graph.aget_state(config)
                if not curr_state.next:
                    break
                async for event in graph.astream(None, config=config):
                    yield event
        return _stream_generator()
    else:
        # Run synchronous-like loop resuming past interrupts
        await graph.ainvoke(initial_state, config=config)
        while True:
            curr_state = await graph.aget_state(config)
            if not curr_state.next:
                final_val = curr_state.values
                break
            await graph.ainvoke(None, config=config)
            
        answer = final_val.get("answer", "")
        entailment = final_val.get("entailment_score", 0.0)
        iterations = final_val.get("iteration", 0)
        
        docs = final_val.get("retrieved_docs", [])
        sources = list(set([doc.metadata.get("source", "") for doc in docs if doc.metadata]))
        sources = [s for s in sources if s]
        
        query_rewrites = []
        if final_val.get("rewritten_query"):
            query_rewrites.append(final_val.get("rewritten_query"))
            
        return {
            "answer": answer,
            "entailment_score": entailment,
            "iterations": iterations,
            "sources": sources,
            "query_rewrites": query_rewrites,
        }


if __name__ == "__main__":
    from unittest.mock import MagicMock, AsyncMock
    
    async def main():
        print("Starting RAG Pipeline standalone mock test...")
        
        # Mock vector store
        mock_vs = MagicMock()
        mock_vs.similarity_search = AsyncMock(return_value=[])
        
        # Instantiate real Retriever with mocked vector store
        retriever = AdaptiveSemanticRetriever(vector_store=mock_vs)
        # Mock retrieval documents returning
        retriever._aget_relevant_documents = AsyncMock(return_value=[
            Document(page_content="Adaptive semantic chunking outperforms static windows in RAG.", metadata={"source": "asc_paper.pdf", "boundary_z_score": 1.2, "avg_perplexity": 2.4}),
            Document(page_content="Neural perplexity measures sentence surprisal.", metadata={"source": "surprise.txt", "boundary_z_score": 0.8, "avg_perplexity": 1.5})
        ])
        
        # Mock LLM calls
        class MockLLM:
            async def ainvoke(self, prompt: str) -> str:
                print(f"[MockLLM] Prompt: {prompt[:60]}...")
                if "scale 0-10" in prompt:
                    return "8" # Doc grade (relevant)
                elif "follow logically" in prompt:
                    return "0.9" # Entailment score (high)
                elif "poor results" in prompt:
                    return "What is neural surprise?"
                else:
                    return "Mocked Answer: Neural perplexity is an effective proxy for semantic shifts."

        llm = MockLLM()
        
        print("\nInvoking run_rag...")
        result = await run_rag(
            question="What is perplexity?",
            retriever=retriever,
            llm=llm,
            stream=False
        )
        print("\nRAG Run completed:")
        print(result)

    asyncio.run(main())
