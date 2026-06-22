"""Adaptive Semantic Chunking (ASC) package.

This package provides a research-grade document chunking method that uses
LLM perplexity scores to dynamically detect semantic boundaries.
"""

from dotenv import load_dotenv
load_dotenv()

from asc.chunker.adaptive_chunker import AdaptiveSemanticChunker, ChunkMetadata
from asc.retrieval.retriever import AdaptiveSemanticRetriever
from asc.vectorstore.chroma_store import ASCVectorStore
from asc.evaluation.benchmark import ASCBenchmark

__version__ = "0.1.0"
__author__ = "ASC Research"

__all__ = [
    "AdaptiveSemanticChunker",
    "AdaptiveSemanticRetriever",
    "ASCVectorStore",
    "ASCBenchmark",
    "ChunkMetadata",
]
