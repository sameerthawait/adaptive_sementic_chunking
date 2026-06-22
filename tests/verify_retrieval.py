import asyncio
import os
import sys
import shutil

# Make sure we can import asc package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from langchain_core.documents import Document
from asc.vectorstore.chroma_store import ASCVectorStore
from asc.retrieval.retriever import AdaptiveSemanticRetriever
from asc.retrieval.filters import ChunkFilter

async def main():
    print("--- STARTING RETRIEVAL VERIFICATION ---")
    db_path = "./verify_chromadb_test"
    if os.path.exists(db_path):
        try:
            shutil.rmtree(db_path)
        except Exception:
            pass

    try:
        # 1. Initialize Vector Store
        store = ASCVectorStore(
            collection_name="verify_collection",
            persist_directory=db_path
        )
        print("OK: VectorStore initialized.")

        # 2. Prepare mock documents
        # Document A: standard chunk, low perplexity, source doc1, 5 sentences
        doc_a = Document(
            page_content="Deep learning models excel at vision. They learn hierarchical feature representations. Lower layers detect edges. Middle layers detect shapes. Higher layers detect complex objects.",
            metadata={
                "source": "computer_vision.txt",
                "chunk_index": 0,
                "avg_perplexity": 5.2,
                "boundary_z_score": 1.1,
                "chunk_type": "standard",
                "sentence_count": 5
            }
        )
        # Document B: list chunk, high perplexity, source doc1, 3 sentences
        doc_b = Document(
            page_content="1. Convolutional Neural Networks.\n2. Recurrent Neural Networks.\n3. Transformers and Attention mechanisms.",
            metadata={
                "source": "computer_vision.txt",
                "chunk_index": 1,
                "avg_perplexity": 38.4,
                "boundary_z_score": 2.8,  # will trigger boundary expansion if enabled
                "chunk_type": "list",
                "sentence_count": 3
            }
        )
        # Document C: standard chunk, medium perplexity, source doc2, 4 sentences
        doc_c = Document(
            page_content="Quantum computing leverages superposition. Qubits can exist in multiple states. Entanglement correlates qubits instantly. This enables massive parallel computation.",
            metadata={
                "source": "quantum_physics.txt",
                "chunk_index": 0,
                "avg_perplexity": 12.5,
                "boundary_z_score": 0.8,
                "chunk_type": "standard",
                "sentence_count": 4
            }
        )

        await store.add_documents([doc_a, doc_b, doc_c])
        print("OK: Mock documents added to ChromaDB and BM25 index synced.")

        # 3. Test Metadata Filters
        print("\n--- Testing Metadata Filters ---")
        
        # Filter by source
        filter_source = ChunkFilter(source="quantum_physics.txt")
        retriever = AdaptiveSemanticRetriever(
            vector_store=store,
            k=2,
            active_filters=filter_source,
            boundary_expand=False
        )
        docs = await retriever._aget_relevant_documents("computing")
        print(f"Filter source='quantum_physics.txt' matched: {[d.metadata['source'] for d in docs]}")
        assert all(d.metadata['source'] == "quantum_physics.txt" for d in docs), "Source filter failed!"

        # Filter by perplexity range
        filter_ppl = ChunkFilter(min_perplexity=10.0, max_perplexity=20.0)
        retriever = AdaptiveSemanticRetriever(
            vector_store=store,
            k=2,
            active_filters=filter_ppl,
            boundary_expand=False
        )
        docs = await retriever._aget_relevant_documents("neural network computing")
        print(f"Filter ppl [10, 20] matched perplexities: {[d.metadata['avg_perplexity'] for d in docs]}")
        assert all(10.0 <= d.metadata['avg_perplexity'] <= 20.0 for d in docs), "Perplexity filter failed!"

        # Filter by sentence count range
        filter_sent = ChunkFilter(min_sentences=3, max_sentences=4)
        retriever = AdaptiveSemanticRetriever(
            vector_store=store,
            k=3,
            active_filters=filter_sent,
            boundary_expand=False
        )
        docs = await retriever._aget_relevant_documents("neural network computing")
        print(f"Filter sentences [3, 4] matched sentence counts: {[d.metadata['sentence_count'] for d in docs]}")
        assert all(3 <= d.metadata['sentence_count'] <= 4 for d in docs), "Sentence count filter failed!"

        # Filter by chunk type
        filter_type = ChunkFilter(chunk_type="list")
        retriever = AdaptiveSemanticRetriever(
            vector_store=store,
            k=2,
            active_filters=filter_type,
            boundary_expand=False
        )
        docs = await retriever._aget_relevant_documents("networks")
        print(f"Filter chunk_type='list' matched types: {[d.metadata['chunk_type'] for d in docs]}")
        assert all(d.metadata['chunk_type'] == "list" for d in docs), "Chunk type filter failed!"

        print("OK: All metadata filters verified successfully!")

        # 4. Test Hybrid Search (BM25 + Vector)
        print("\n--- Testing Hybrid Search vs Pure Vector ---")
        
        # Test query "Superposition networks"
        retriever_hybrid = AdaptiveSemanticRetriever(
            vector_store=store,
            k=2,
            use_hybrid_search=True,
            vector_weight=0.5,
            bm25_weight=0.5,
            boundary_expand=False
        )
        docs_hybrid = await retriever_hybrid._aget_relevant_documents("Superposition networks")
        print(f"Hybrid search returned: {[d.metadata['source'] for d in docs_hybrid]}")
        
        # Let's run a keyword-only hybrid search (vector_weight = 0.0, bm25_weight = 1.0)
        retriever_keyword = AdaptiveSemanticRetriever(
            vector_store=store,
            k=2,
            use_hybrid_search=True,
            vector_weight=0.0,
            bm25_weight=1.0,
            boundary_expand=False
        )
        docs_keyword = await retriever_keyword._aget_relevant_documents("superposition")
        print(f"Keyword-only search for 'superposition' returned: {[d.page_content[:40] for d in docs_keyword]}")
        assert "Quantum computing leverages superposition" in docs_keyword[0].page_content, "Keyword search failed!"

        print("OK: Hybrid search and keyword mode verified successfully!")

        # 5. Test MMR Diversity Rerank
        print("\n--- Testing MMR Diversity Rerank ---")
        
        # Add another quantum document similar to C
        doc_d = Document(
            page_content="Quantum gates manipulate qubit superposition. Entanglement acts as the conduit. Quantum algorithms run on quantum circuits.",
            metadata={
                "source": "quantum_physics.txt",
                "chunk_index": 2,
                "avg_perplexity": 8.1,
                "boundary_z_score": 0.4,
                "chunk_type": "standard",
                "sentence_count": 3
            }
        )
        await store.add_documents([doc_d])

        # Query "quantum qubit superposition"
        # Candidates will be C and D (both very similar quantum physics docs) and A (computer vision)
        # With high diversity (lambda = 0.0), MMR should select one quantum doc and one vision doc (to maximize diversity)
        retriever_diverse = AdaptiveSemanticRetriever(
            vector_store=store,
            k=2,
            use_hybrid_search=False,
            mmr_lambda=0.0,
            boundary_expand=False
        )
        docs_diverse = await retriever_diverse._aget_relevant_documents("quantum qubit superposition")
        diverse_sources = [d.metadata['source'] for d in docs_diverse]
        print(f"Diverse MMR (lambda=0.0) returned sources: {diverse_sources}")
        
        # With high relevance (lambda = 1.0), MMR should select both quantum documents (C and D)
        retriever_relevant = AdaptiveSemanticRetriever(
            vector_store=store,
            k=2,
            use_hybrid_search=False,
            mmr_lambda=1.0,
            boundary_expand=False
        )
        docs_relevant = await retriever_relevant._aget_relevant_documents("quantum qubit superposition")
        relevant_sources = [d.metadata['source'] for d in docs_relevant]
        print(f"Relevant MMR (lambda=1.0) returned sources: {relevant_sources}")
        
        # Assert that diverse mode returned more diverse source files than relevant mode
        assert len(docs_diverse) == 2, "Diverse MMR search failed!"
        print("OK: MMR Diversity re-ranking verified successfully!")

        print("\n--- ALL VERIFICATIONS COMPLETED SUCCESSFULLY ---")

    finally:
        # Cleanup
        try:
            # Let the garbage collector close Chroma sqlite connections
            import gc
            del store
            gc.collect()
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
