"""ChromaStore module for Adaptive Semantic Chunking.

Manages persistent Chroma vector store creation, indexing, and similarity search
using raw Chroma client to enable deep metadata control and stats reporting.
"""

import os
import shutil
import logging
import hashlib
import asyncio
import numpy as np
from typing import Any
from langchain_core.documents import Document

from asc.embedding.embedder import ASCEmbedder
from asc.retrieval.bm25_index import BM25Index

logger = logging.getLogger(__name__)


class ASCVectorStore:
    """
    ChromaDB wrapper that stores ASC chunks with full metadata.
    Supports: add, query, get_by_ids, get_adjacent_chunks, delete, reset.
    """
    
    def __init__(
        self,
        collection_name: str,
        persist_directory: str,
        embedding_model: str | None = None,
        ollama_base_url: str | None = None
    ) -> None:
        """Initializes the vector store and persistent client."""
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedder = ASCEmbedder(
            model=embedding_model,
            base_url=ollama_base_url
        )
        
        import chromadb
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # hnsw:space cosine ensures we use cosine distance for vector matching
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize and build/load BM25 index
        self.bm25_path = os.path.join(self.persist_directory, f"bm25_{self.collection_name}.pkl")
        self.bm25_index = BM25Index()
        if os.path.exists(self.bm25_path):
            try:
                self.bm25_index.load(self.bm25_path)
            except Exception as e:
                logger.warning(f"Failed to load BM25 index from {self.bm25_path}: {e}. Rebuilding...")
                self._rebuild_bm25_from_chroma()
        else:
            self._rebuild_bm25_from_chroma()

    def _rebuild_bm25_from_chroma(self) -> None:
        """Rebuilds the BM25 index from documents currently in ChromaDB."""
        res = self.collection.get()
        documents = []
        if res and "ids" in res and res["ids"]:
            for doc_id, text, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                doc = Document(page_content=text, metadata=meta)
                doc.metadata["id"] = doc_id
                documents.append(doc)
        self.bm25_index.build(documents)
        try:
            self.bm25_index.save(self.bm25_path)
        except Exception as e:
            logger.error(f"Failed to save BM25 index to {self.bm25_path}: {e}")
    
    async def add_documents(
        self,
        documents: list[Document],
        batch_size: int = 50
    ) -> list[str]:
        """
        Embeds and stores documents in batches.
        Generates deterministic IDs from source + chunk_index.
        Skips duplicates (upsert behavior).
        Returns list of stored IDs.
        """
        stored_ids = []
        if not documents:
            return stored_ids

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch]
            
            # Asynchronously compute embeddings using OllamaEmbedder
            embeddings = await self.embedder.aembed_documents(texts)
            
            ids = []
            metadatas = []
            for doc in batch:
                source = doc.metadata.get("source", "") or "default_source"
                source_hash = hashlib.md5(source.encode("utf-8")).hexdigest()
                chunk_index = doc.metadata.get("chunk_index", 0)
                
                # Deterministic ID combining source hash and chunk index
                doc_id = f"{source_hash}_{chunk_index}"
                ids.append(doc_id)
                stored_ids.append(doc_id)
                doc.metadata["id"] = doc_id
                
                # Ensure all metadata values are primitive types
                clean_meta = {}
                for k, v in doc.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)
                metadatas.append(clean_meta)
                
            # Upsert behavior (replaces existing if keys collide)
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=texts
            )

        # De-duplicate existing BM25 documents and update the index
        new_ids = set(stored_ids)
        filtered_existing = [doc for doc in self.bm25_index.documents if doc.metadata.get("id") not in new_ids]
        self.bm25_index.build(filtered_existing + documents)
        try:
            self.bm25_index.save(self.bm25_path)
        except Exception as e:
            logger.error(f"Failed to save BM25 index to {self.bm25_path}: {e}")
            
        return stored_ids

    async def add(self, documents: list[Document], batch_size: int = 50) -> list[str]:
        """Alias for add_documents."""
        return await self.add_documents(documents, batch_size=batch_size)
    
    async def embed_query(self, query: str) -> list[float]:
        """Asynchronously embeds a query."""
        return await self.embedder.aembed_query(query)

    
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None
    ) -> list[tuple[Document, float]]:
        """Returns (document, score) tuples, score is cosine similarity 0-1."""
        if not query.strip():
            return []

        query_vector = await self.embedder.aembed_query(query)
        
        res = self.collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=filter
        )
        
        results = []
        if not res or "ids" not in res or not res["ids"] or not res["ids"][0]:
            return results
            
        ids = res["ids"][0]
        distances = res["distances"][0] if "distances" in res and res["distances"] else [0.0] * len(ids)
        metadatas = res["metadatas"][0] if "metadatas" in res and res["metadatas"] else [{}] * len(ids)
        documents = res["documents"][0] if "documents" in res and res["documents"] else [""] * len(ids)
        
        for doc_id, dist, meta, text in zip(ids, distances, metadatas, documents):
            # Chroma Cosine Distance = 1.0 - Cosine Similarity
            # Cosine Similarity = 1.0 - Distance
            similarity = max(0.0, min(1.0, 1.0 - float(dist)))
            
            doc = Document(page_content=text, metadata=meta)
            doc.metadata["id"] = doc_id
            results.append((doc, similarity))
            
        return results

    async def query(self, query: str, k: int = 4, filter: dict | None = None) -> list[tuple[Document, float]]:
        """Alias for similarity_search."""
        return await self.similarity_search(query, k=k, filter=filter)

    def get_by_ids(self, ids: list[str]) -> list[Document]:
        """Fetches document chunks matching the list of unique IDs."""
        if not ids:
            return []
        res = self.collection.get(ids=ids)
        docs = []
        if res and "ids" in res and res["ids"]:
            for doc_id, text, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                doc = Document(page_content=text, metadata=meta)
                doc.metadata["id"] = doc_id
                docs.append(doc)
        return docs
    
    def get_adjacent_chunks(
        self,
        chunk_id: str,
        window: int = 1
    ) -> list[Document]:
        """
        Fetches chunks with chunk_index ± window from the same source.
        Used by boundary-aware expansion in retriever.
        """
        parts = chunk_id.split("_")
        if len(parts) != 2:
            return []
            
        source_hash, chunk_index_str = parts[0], parts[1]
        try:
            chunk_index = int(chunk_index_str)
        except ValueError:
            return []
            
        target_ids = []
        for i in range(chunk_index - window, chunk_index + window + 1):
            if i == chunk_index:
                continue
            target_ids.append(f"{source_hash}_{i}")
            
        res = self.collection.get(ids=target_ids)
        docs = []
        if res and "ids" in res and res["ids"]:
            for doc_id, text, meta in zip(res["ids"], res["documents"], res["metadatas"]):
                doc = Document(page_content=text, metadata=meta)
                doc.metadata["id"] = doc_id
                docs.append(doc)
                
        # Sort sequentially to keep semantic context ordered
        docs.sort(key=lambda d: d.metadata.get("chunk_index", 0))
        return docs
    
    def get_collection_stats(self) -> dict:
        """Returns: total_chunks, sources, avg_chunk_size, chunk_type_counts."""
        res = self.collection.get()
        if not res or "ids" not in res or not res["ids"]:
            return {
                "total_chunks": 0,
                "sources": [],
                "avg_chunk_size": 0.0,
                "chunk_type_counts": {},
            }
            
        ids = res["ids"]
        metadatas = res["metadatas"]
        documents = res["documents"]
        
        total_chunks = len(ids)
        sources = list(set([meta.get("source", "") for meta in metadatas if meta]))
        sources = [s for s in sources if s]
        
        sizes = [len(doc) for doc in documents if doc]
        avg_chunk_size = float(np.mean(sizes)) if sizes else 0.0
        
        chunk_type_counts = {}
        for meta in metadatas:
            if meta:
                ctype = meta.get("chunk_type", "unknown")
                chunk_type_counts[ctype] = chunk_type_counts.get(ctype, 0) + 1
                
        return {
            "total_chunks": total_chunks,
            "sources": sources,
            "avg_chunk_size": avg_chunk_size,
            "chunk_type_counts": chunk_type_counts,
        }

    def delete(self, ids: list[str]) -> None:
        """Deletes chunks by unique IDs."""
        if ids:
            self.collection.delete(ids=ids)
            self._rebuild_bm25_from_chroma()

    def reset(self) -> None:
        """Resets the vector store collection."""
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25_index = BM25Index()
        if os.path.exists(self.bm25_path):
            try:
                os.remove(self.bm25_path)
            except Exception:
                pass

    async def hybrid_search(
        self,
        query: str,
        k: int = 4,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        filter: dict | None = None,
        rrf_k: int = 60
    ) -> list[tuple[Document, float]]:
        """Combines ChromaDB vector search with BM25 keyword search using Reciprocal Rank Fusion.

        Pipeline:
        1. Run vector search → get top (k * 3) candidates
        2. Run BM25 search → get top (k * 3) candidates
        3. For each unique document, compute:
           rrf_score = 1/(rrf_k + rank_vector) + 1/(rrf_k + rank_bm25)
           If document only appears in one list: use rank = k*3 + 1 for the missing list
        4. Sort by rrf_score descending, return top k
        5. Apply metadata filter BEFORE vector search using ChromaDB where clause

        Returns:
            List of (Document, rrf_score) tuples.
        """
        # 1. Run vector search with filter
        vector_results = await self.similarity_search(query, k=k * 3, filter=filter)

        # 2. Run BM25 search
        bm25_all = self.bm25_index.search(query, k=len(self.bm25_index.documents))

        # Filter BM25 results by allowed metadata IDs if filter is provided
        if filter:
            res = self.collection.get(where=filter)
            allowed_ids = set(res["ids"]) if res and "ids" in res else set()
            bm25_all = [item for item in bm25_all if item[0].metadata.get("id") in allowed_ids]

        bm25_results = bm25_all[:k * 3]

        # 3. Reciprocal Rank Fusion
        fallback_rank = k * 3 + 1

        vector_ranks = {}
        for idx, (doc, _) in enumerate(vector_results):
            doc_id = doc.metadata.get("id")
            if doc_id:
                vector_ranks[doc_id] = (doc, idx + 1)

        bm25_ranks = {}
        for idx, (doc, _) in enumerate(bm25_results):
            doc_id = doc.metadata.get("id")
            if doc_id:
                bm25_ranks[doc_id] = (doc, idx + 1)

        all_doc_ids = set(vector_ranks.keys()).union(set(bm25_ranks.keys()))

        rrf_results = []
        for doc_id in all_doc_ids:
            doc = None
            rank_v = fallback_rank
            rank_b = fallback_rank

            if doc_id in vector_ranks:
                doc, rank_v = vector_ranks[doc_id]
            if doc_id in bm25_ranks:
                doc, rank_b = bm25_ranks[doc_id]

            # Weighted RRF score
            rrf_score = vector_weight * (1.0 / (rrf_k + rank_v)) + bm25_weight * (1.0 / (rrf_k + rank_b))
            rrf_results.append((doc, rrf_score))

        # Sort descending by rrf_score
        rrf_results.sort(key=lambda x: x[1], reverse=True)
        return rrf_results[:k]


# Backward compatibility wrapper:
class ChromaStore:
    """Manages index operations for ChromaDB vector store (wrapper)."""

    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "asc_collection",
        embedder: Any | None = None,
    ) -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vector_store = ASCVectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory
        )

    def add_documents(self, documents: list[Document]) -> None:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                loop.run_until_complete(self.vector_store.add_documents(documents))
            else:
                loop.run_until_complete(self.vector_store.add_documents(documents))
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                new_loop.run_until_complete(self.vector_store.add_documents(documents))
            finally:
                new_loop.close()

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                tuples = loop.run_until_complete(self.vector_store.similarity_search(query, k=k))
            else:
                tuples = loop.run_until_complete(self.vector_store.similarity_search(query, k=k))
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                tuples = new_loop.run_until_complete(self.vector_store.similarity_search(query, k=k))
            finally:
                new_loop.close()
        return [doc for doc, _ in tuples]

    def as_retriever(self, search_kwargs: dict[str, Any] | None = None) -> Any:
        from asc.retrieval.retriever import AdaptiveSemanticRetriever
        kwargs = search_kwargs or {}
        k = kwargs.get("k", 4)
        return AdaptiveSemanticRetriever(vector_store=self.vector_store, k=k)

    def clear(self) -> None:
        self.vector_store.reset()
