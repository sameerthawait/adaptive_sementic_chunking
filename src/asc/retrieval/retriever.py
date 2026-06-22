"""Custom retriever module for Adaptive Semantic Chunking.

Implements a custom LangChain BaseRetriever with Coherence Re-Ranking and
Boundary-Aware Expansion.
"""

import logging
import math
import asyncio
import numpy as np
from typing import Any, List
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun

from asc.vectorstore.chroma_store import ASCVectorStore

logger = logging.getLogger(__name__)


class AdaptiveSemanticRetriever(BaseRetriever):
    """
    LangChain retriever with two novel enhancements:
    
    1. COHERENCE RE-RANKING: After initial retrieval, re-ranks docs by 
       computing cosine similarity between query embedding and the 
       avg_perplexity-weighted centroid of chunk embeddings.
       
    2. BOUNDARY-AWARE EXPANSION: If a retrieved chunk's boundary_z_score
       exceeds expand_threshold, also fetches adjacent chunks — because
       high z-score means the chunk boundary was "forced" and the full
       semantic unit may span multiple chunks.
    """
    
    vectorstore: Any
    vector_store: Any = None
    k: int = 4
    coherence_rerank: bool = True
    boundary_expand: bool = True
    expand_threshold: float = 2.5
    use_hybrid_search: bool = True
    mmr_lambda: float = 0.5          # 0=diverse, 1=relevant
    vector_weight: float = 0.6       # weight for vector scores in hybrid
    bm25_weight: float = 0.4         # weight for BM25 scores in hybrid
    active_filters: Any = None
    
    class Config:
        arbitrary_types_allowed = True
        
    def __init__(self, **data: Any) -> None:
        # Pydantic initialization compatibility: sync vector_store and vectorstore
        if "vector_store" in data and "vectorstore" not in data:
            data["vectorstore"] = data["vector_store"]
        super().__init__(**data)
        if self.vector_store is None:
            self.vector_store = self.vectorstore

    def _coherence_score(
        self,
        query_embedding: list[float],
        doc_embedding: list[float],
        avg_perplexity: float
    ) -> float:
        """
        score = cosine_similarity(query_emb, doc_emb) * (1 / log(1 + avg_perplexity))
        Perplexity penalty: very high-perplexity chunks are downweighted
        even if semantically similar (they may be incoherent).
        """
        q = np.array(query_embedding)
        d = np.array(doc_embedding)
        norm_q = np.linalg.norm(q)
        norm_d = np.linalg.norm(d)
        
        if norm_q == 0.0 or norm_d == 0.0:
            cos_sim = 0.0
        else:
            cos_sim = float(np.dot(q, d) / (norm_q * norm_d))
            
        # Log scaling divisor (ensure divisor > 0)
        denom = math.log(1.0 + max(1.1, avg_perplexity))
        return cos_sim * (1.0 / denom)

    def _get_document_embedding(self, doc: Document) -> list[float]:
        """Retrieve stored embedding from ChromaDB for this document's ID."""
        doc_id = doc.metadata.get("id")
        fallback_dim = 1024 if "nvidia" in str(getattr(self.vectorstore, "embedder", None) and self.vectorstore.embedder.provider) else 768
        if not doc_id:
            return [0.0] * fallback_dim
            
        res = self.vectorstore.collection.get(ids=[doc_id], include=["embeddings"])
        if res and "embeddings" in res and res["embeddings"] is not None and len(res["embeddings"]) > 0:
            emb = res["embeddings"][0]
            if hasattr(emb, "tolist"):
                return emb.tolist()
            return list(emb)
            
        return [0.0] * fallback_dim

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Synchronously retrieves relevant documents for a query."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self._aget_relevant_documents(query))
            else:
                return loop.run_until_complete(self._aget_relevant_documents(query))
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self._aget_relevant_documents(query))
            finally:
                new_loop.close()

    def mmr_rerank(
        self,
        query_embedding: list[float],
        candidates: list[tuple[Document, float]],
        k: int = 4,
        lambda_mult: float = 0.5
    ) -> list[Document]:
        """Maximal Marginal Relevance reranking for diversity.

        Formula for each selection step:
          score(d) = lambda * sim(d, query) - (1 - lambda) * max(sim(d, selected))
          where sim = cosine similarity

        Args:
          query_embedding: The query embedding vector.
          candidates: List of (Document, score) tuples.
          k: Number of documents to select.
          lambda_mult: 0.0 = maximum diversity, 1.0 = maximum relevance, 0.5 = balanced

        Returns:
          List of selected Documents balancing relevance and diversity.
        """
        if not candidates:
            return []

        q = np.array(query_embedding)
        norm_q = np.linalg.norm(q)
        if norm_q == 0.0:
            return [doc for doc, _ in candidates[:k]]

        # Map doc ID to its embedding to avoid repeated DB fetches
        candidate_embeddings = {}
        for doc, _ in candidates:
            doc_id = doc.metadata.get("id")
            if doc_id:
                emb = self._get_document_embedding(doc)
                candidate_embeddings[doc_id] = np.array(emb)

        selected_docs = []

        def cos_sim(emb1, emb2):
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0.0 or norm2 == 0.0:
                return 0.0
            return float(np.dot(emb1, emb2) / (norm1 * norm2))

        remaining = list(candidates)

        while len(selected_docs) < k and remaining:
            best_score = -float("inf")
            best_idx = -1

            for idx, (doc, rel_score) in enumerate(remaining):
                doc_id = doc.metadata.get("id")
                if not doc_id:
                    continue
                emb = candidate_embeddings.get(doc_id)
                if emb is None:
                    continue

                # Query similarity: if coherence_rerank is enabled, use coherence score
                if self.coherence_rerank:
                    avg_ppl = float(doc.metadata.get("avg_perplexity", 15.0))
                    sim_to_query = self._coherence_score(query_embedding, emb.tolist(), avg_ppl)
                else:
                    sim_to_query = cos_sim(emb, q)

                # Max similarity to already selected docs
                if not selected_docs:
                    max_sim_to_selected = 0.0
                else:
                    max_sim_to_selected = max(
                        [cos_sim(emb, candidate_embeddings[sel_doc.metadata.get("id")])
                         for sel_doc in selected_docs]
                    )

                score = lambda_mult * sim_to_query - (1.0 - lambda_mult) * max_sim_to_selected

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == -1:
                selected_docs.append(remaining.pop(0)[0])
            else:
                selected_docs.append(remaining.pop(best_idx)[0])

        return selected_docs

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun | None = None
    ) -> list[Document]:
        """Retrieves relevant documents based on hybrid search and MMR reranking.

        Pipeline:
        1. hybrid_search(query, k=k*3, filter=active_filters)   ← over-retrieve
        2. mmr_rerank(query_embedding, candidates, k, lambda_mult)
        3. boundary_expand(results)                              ← existing feature
        4. return final k documents
        """
        # Resolve filters
        chroma_filter = None
        if self.active_filters:
            if hasattr(self.active_filters, "to_chroma_where"):
                chroma_filter = self.active_filters.to_chroma_where()
            else:
                chroma_filter = self.active_filters

        # 1. Retrieve candidates
        if self.use_hybrid_search:
            candidates = await self.vector_store.hybrid_search(
                query=query,
                k=self.k * 3,
                vector_weight=self.vector_weight,
                bm25_weight=self.bm25_weight,
                filter=chroma_filter
            )
        else:
            candidates = await self.vector_store.similarity_search(
                query=query,
                k=self.k * 3,
                filter=chroma_filter
            )

        if not candidates:
            return []

        # 2. MMR Rerank
        query_embedding = await self.vector_store.embedder.aembed_query(query)
        top_k_docs = self.mmr_rerank(
            query_embedding=query_embedding,
            candidates=candidates,
            k=self.k,
            lambda_mult=self.mmr_lambda
        )

        # 3. Boundary Expansion
        expanded_docs = []
        for doc in top_k_docs:
            expanded_docs.append(doc)

            if self.boundary_expand:
                z_score = float(doc.metadata.get("boundary_z_score", 0.0))
                if z_score >= self.expand_threshold:
                    doc_id = doc.metadata.get("id")
                    if doc_id:
                        adjacent = self.vector_store.get_adjacent_chunks(doc_id, window=1)
                        expanded_docs.extend(adjacent)

        seen_ids = set()
        final_docs = []
        for doc in expanded_docs:
            doc_id = doc.metadata.get("id") or f"{doc.metadata.get('source')}_{doc.metadata.get('chunk_index')}"
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                final_docs.append(doc)

        return final_docs
