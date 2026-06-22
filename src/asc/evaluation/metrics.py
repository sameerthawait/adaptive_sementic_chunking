"""Metrics module for Adaptive Semantic Chunking evaluation.

Computes stats on chunk sizes, semantic coherence, and retrieval quality.
"""

import numpy as np
from typing import Sequence, Any
from langchain_core.documents import Document

# New metrics requested
def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k retrieved that are relevant."""
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_retrieved = [rid for rid in top_k if rid in relevant_ids]
    return len(relevant_retrieved) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant that appear in top-k retrieved."""
    if not retrieved_ids or not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_retrieved = [rid for rid in top_k if rid in relevant_ids]
    return len(relevant_retrieved) / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    """Computes reciprocal rank for a single query (averaged externally for MRR)."""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    for idx, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (idx + 1)
    return 0.0


def chunk_coherence_score(
    chunk_sentences: list[str],
    embedder
) -> float:
    """
    Mean pairwise cosine similarity of sentence embeddings within a chunk.
    Higher = more coherent chunk. This is a KEY metric showing ASC superiority.
    """
    if len(chunk_sentences) <= 1:
        return 1.0
        
    embeddings = embedder.embed_documents(chunk_sentences)
    embeddings = [np.array(e) for e in embeddings]
    
    sims = []
    n = len(embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            norm_i = np.linalg.norm(embeddings[i])
            norm_j = np.linalg.norm(embeddings[j])
            if norm_i == 0.0 or norm_j == 0.0:
                sims.append(0.0)
            else:
                sim = float(np.dot(embeddings[i], embeddings[j]) / (norm_i * norm_j))
                sims.append(sim)
                
    return float(np.mean(sims)) if sims else 1.0


def boundary_quality_score(
    chunk_a_sentences: list[str],
    chunk_b_sentences: list[str],
    embedder
) -> float:
    """
    1 - cosine_similarity(mean_embed(A), mean_embed(B)).
    Higher = more distinct adjacent chunks = better boundary placement.
    """
    if not chunk_a_sentences or not chunk_b_sentences:
        return 1.0
        
    embed_a = embedder.embed_documents(chunk_a_sentences)
    embed_b = embedder.embed_documents(chunk_b_sentences)
    
    mean_a = np.mean(embed_a, axis=0)
    mean_b = np.mean(embed_b, axis=0)
    
    norm_a = np.linalg.norm(mean_a)
    norm_b = np.linalg.norm(mean_b)
    
    if norm_a == 0.0 or norm_b == 0.0:
        cos_sim = 0.0
    else:
        cos_sim = float(np.dot(mean_a, mean_b) / (norm_a * norm_b))
        
    return 1.0 - cos_sim


def chunk_size_stats(chunks: list[Document]) -> dict:
    """Returns: mean_tokens, std_tokens, min_tokens, max_tokens, cv (coeff of variation)."""
    import tiktoken
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        lengths = [len(encoding.encode(doc.page_content)) for doc in chunks]
    except Exception:
        lengths = [len(doc.page_content.split()) for doc in chunks]
        
    if not lengths:
        return {
            "mean_tokens": 0.0,
            "std_tokens": 0.0,
            "min_tokens": 0,
            "max_tokens": 0,
            "cv": 0.0
        }
        
    mean_val = float(np.mean(lengths))
    std_val = float(np.std(lengths))
    cv_val = std_val / mean_val if mean_val > 0.0 else 0.0
    
    return {
        "mean_tokens": mean_val,
        "std_tokens": std_val,
        "min_tokens": int(np.min(lengths)),
        "max_tokens": int(np.max(lengths)),
        "cv": cv_val
    }


# Original functions preserved for backward compatibility:
def calculate_chunk_statistics(chunks: Sequence[Document] | list[str]) -> dict[str, Any]:
    """Calculates distribution statistics for a set of chunks (character length)."""
    lengths = []
    for c in chunks:
        if hasattr(c, "page_content"):
            lengths.append(len(c.page_content))
        else:
            lengths.append(len(str(c)))

    if not lengths:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "total_chars": 0,
        }

    return {
        "count": len(lengths),
        "mean": float(np.mean(lengths)),
        "std": float(np.std(lengths)),
        "min": int(np.min(lengths)),
        "max": int(np.max(lengths)),
        "total_chars": int(np.sum(lengths)),
    }


def compute_retrieval_overlap(
    retrieved_docs: list[Document],
    ground_truth_text: str,
) -> float:
    """Computes character-level overlap ratio between retrieved docs and ground truth text."""
    if not retrieved_docs or not ground_truth_text.strip():
        return 0.0

    retrieved_combined = " ".join([d.page_content for d in retrieved_docs])
    gt_words = set(ground_truth_text.lower().split())
    ret_words = set(retrieved_combined.lower().split())
    
    if not gt_words:
        return 0.0
        
    overlap = gt_words.intersection(ret_words)
    return len(overlap) / len(gt_words)
