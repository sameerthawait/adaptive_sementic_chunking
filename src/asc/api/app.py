"""FastAPI REST API application for Adaptive Semantic Chunking.

Provides endpoints for text chunking, indexing, retrieval, health status, and full RAG.
"""

import os
import io
import time
import base64
import logging
import nltk
import numpy as np
import httpx
from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from asc.chunker.adaptive_chunker import AdaptiveSemanticChunker
from asc.vectorstore.chroma_store import ASCVectorStore
from asc.retrieval.retriever import AdaptiveSemanticRetriever
from asc.retrieval.rag_pipeline import run_rag

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("asc_api")

app = FastAPI(
    title="Adaptive Semantic Chunking API",
    description="Research implementation of LLM perplexity-based semantic chunking for RAG",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class APIPrefixMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path.startswith("/api/"):
                scope["path"] = path[4:]
            elif path == "/api":
                scope["path"] = "/"
        await self.app(scope, receive, send)


app.add_middleware(APIPrefixMiddleware)



# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    t0 = time.time()
    response = await call_next(request)
    duration = time.time() - t0
    logger.info(f"Response status: {response.status_code} for {request.method} {request.url.path} (took {duration:.3f}s)")
    return response


# Serve compiled React SPA
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "frontend", "dist"))

if os.path.exists(frontend_dist) and os.path.exists(os.path.join(frontend_dist, "assets")):
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="static")

@app.get("/", include_in_schema=False)
async def serve_index():
    if os.path.exists(frontend_dist):
        index_path = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
    return {
        "status": "online",
        "message": "ASC Backend is running. Run 'npm run build' inside frontend directory to mount the React Dashboard here."
    }


# Pydantic Request/Response Models
class ChunkRequest(BaseModel):
    text: str = Field(..., description="The raw document text to segment.")
    source: str = Field("api_input", description="Filename or descriptor of source.")
    visualize: bool = Field(False, description="Whether to compute and return base64 plot of perplexities.")
    z_score_threshold: Optional[float] = Field(None, description="Z-score threshold for boundary detection.")
    context_window: Optional[int] = Field(None, description="Context window size for perplexity calculation.")
    min_chunk_sentences: Optional[int] = Field(None, description="Minimum sentences per chunk.")
    sentence_overlap: Optional[bool] = Field(None, description="Whether to include sentence overlap.")


class IndexRequest(BaseModel):
    texts: List[str] = Field(..., description="List of document texts to chunk and index.")
    sources: List[str] = Field(..., description="List of source filenames corresponding to texts.")
    collection: str = Field("default", description="The ChromaDB collection name.")


class FilterParams(BaseModel):
    source: Optional[str] = None
    sources: Optional[List[str]] = None
    min_perplexity: Optional[float] = None
    max_perplexity: Optional[float] = None
    min_sentences: Optional[int] = None
    max_sentences: Optional[int] = None
    chunk_type: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., description="The search query or question.")
    collection: str = Field("default", description="The ChromaDB collection name.")
    k: int = Field(4, description="Number of document chunks to retrieve.")
    use_rag: bool = Field(True, description="Whether to run the self-correcting RAG pipeline.")
    use_hybrid: bool = Field(True, description="Whether to use hybrid search (BM25 + Vector).")
    mmr_lambda: float = Field(0.5, description="MMR lambda multiplier.")
    filters: Optional[FilterParams] = Field(None, description="Metadata filters.")
    vector_weight: Optional[float] = Field(0.6, description="Weight for vector search in hybrid.")
    bm25_weight: Optional[float] = Field(0.4, description="Weight for BM25 search in hybrid.")


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="The search query.")
    collection: str = Field("default", description="The ChromaDB collection name.")
    k: int = Field(4, description="Number of document chunks to retrieve.")
    use_hybrid: bool = Field(True, description="Whether to use hybrid search (BM25 + Vector).")
    mmr_lambda: float = Field(0.5, description="MMR lambda multiplier.")
    filters: Optional[FilterParams] = Field(None, description="Metadata filters.")
    vector_weight: Optional[float] = Field(0.6, description="Weight for vector search in hybrid.")
    bm25_weight: Optional[float] = Field(0.4, description="Weight for BM25 search in hybrid.")


class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    total_chunks: int
    avg_perplexity: float
    boundaries_detected: int
    visualization_base64: Optional[str] = None
    perplexity_scores: Optional[List[float]] = None
    boundaries: Optional[List[int]] = None


# Endpoints
@app.post("/chunk", response_model=ChunkResponse, tags=["ASC Operations"])
async def chunk_text_endpoint(request: ChunkRequest) -> Any:
    """Segments a document using the perplexity-driven adaptive semantic chunker."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text payload cannot be empty.")
        
    try:
        # Resolve parameters from request or fallback to env/defaults
        provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
        context_window = request.context_window if request.context_window is not None else int(os.environ.get("PERPLEXITY_CONTEXT_WINDOW", "3"))
        z_threshold = request.z_score_threshold if request.z_score_threshold is not None else float(os.environ.get("Z_SCORE_THRESHOLD", "2.0"))
        min_chunk_sentences = request.min_chunk_sentences if request.min_chunk_sentences is not None else int(os.environ.get("MIN_CHUNK_SENTENCES", "3"))
        sentence_overlap = 1 if (request.sentence_overlap if request.sentence_overlap is not None else True) else 0

        # Construct customized scorer
        from asc.chunker.perplexity_scorer import PerplexityScorer
        if provider in ["nvidia", "openai"]:
            from asc.embedding.embedder import ASCEmbedder
            embeddings = ASCEmbedder(provider=provider)
            scorer = PerplexityScorer(
                context_window=context_window,
                embeddings=embeddings
            )
        else:
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            model = os.environ.get("OLLAMA_PERPLEXITY_MODEL", "llama3.2:3b")
            max_concurrent = int(os.environ.get("MAX_CONCURRENT_OLLAMA_CALLS", "3"))
            scorer = PerplexityScorer(
                model=model,
                base_url=base_url,
                context_window=context_window,
                max_concurrent=max_concurrent,
            )

        # Construct customized detector
        from asc.chunker.boundary_detector import BoundaryDetector, BoundaryDetectorConfig
        config = BoundaryDetectorConfig(
            z_score_threshold=z_threshold,
            min_chunk_sentences=min_chunk_sentences,
            smoothing_window=5,
            smoothing_poly_order=2,
            lag_window=5,
            require_local_maximum=True,
        )
        detector = BoundaryDetector(config=config)

        # Create chunker instance with custom configs
        chunker = AdaptiveSemanticChunker(
            perplexity_scorer=scorer,
            boundary_detector=detector,
            sentence_overlap=sentence_overlap
        )

        sentences = nltk.sent_tokenize(request.text)
        
        # Calculate perplexity scores
        perplexity_scores = await chunker._score_sentences(sentences)
        perplexity_scores_np = np.array(perplexity_scores)
        
        # Detect boundary positions
        boundaries, diagnostics = chunker.detector.detect_boundaries(perplexity_scores_np)
        
        # Build chunks with overlap
        chunk_dicts = chunker._build_chunks_with_overlap(
            sentences=sentences,
            boundaries=boundaries,
            perplexity_scores=perplexity_scores_np,
            diagnostics=diagnostics,
            overlap=chunker.sentence_overlap
        )
        
        docs = chunker._to_documents(chunk_dicts, {"source": request.source})
        
        chunks_out = []
        for doc in docs:
            chunks_out.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })
            
        avg_ppl = float(np.mean(perplexity_scores_np)) if len(perplexity_scores_np) > 0 else 15.0
        
        viz_b64 = None
        if request.visualize:
            buf = io.BytesIO()
            chunker.detector.visualize(
                perplexity_scores=perplexity_scores_np,
                boundaries=boundaries,
                sentences=sentences,
                save_path=buf,
                show=False
            )
            buf.seek(0)
            viz_b64 = base64.b64encode(buf.read()).decode("utf-8")
            
        return ChunkResponse(
            chunks=chunks_out,
            total_chunks=len(chunks_out),
            avg_perplexity=avg_ppl,
            boundaries_detected=len(boundaries) - 2 if len(boundaries) >= 2 else 0,
            visualization_base64=viz_b64,
            perplexity_scores=perplexity_scores_np.tolist() if hasattr(perplexity_scores_np, "tolist") else list(perplexity_scores_np),
            boundaries=boundaries.tolist() if hasattr(boundaries, "tolist") else list(boundaries)
        )
    except Exception as e:
        logger.error(f"Error in /chunk: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", tags=["Vector Store"])
async def index_documents_endpoint(request: IndexRequest) -> Any:
    """Segment, embed, and load multiple documents into ChromaDB."""
    if len(request.texts) != len(request.sources):
        raise HTTPException(status_code=400, detail="Length of texts must match sources.")
        
    try:
        chunker = AdaptiveSemanticChunker.from_env()
        db_path = os.environ.get("CHROMA_PERSIST_DIR", "./data/chromadb")
        store = ASCVectorStore(collection_name=request.collection, persist_directory=db_path)
        
        total_indexed = 0
        for text, source in zip(request.texts, request.sources):
            docs = await chunker.chunk_text(text, {"source": source})
            await store.add_documents(docs)
            total_indexed += len(docs)
            
        stats = store.get_collection_stats()
        return {
            "indexed": total_indexed,
            "collection": request.collection,
            "total_in_collection": stats.get("total_chunks", 0)
        }
    except Exception as e:
        logger.error(f"Error in /index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve", tags=["Retrieval"])
async def retrieve_endpoint(request: RetrieveRequest) -> Any:
    """Retrieve relevant chunks using over-retrieval, coherence re-ranking, and boundary expansion."""
    try:
        db_path = os.environ.get("CHROMA_PERSIST_DIR", "./data/chromadb")
        store = ASCVectorStore(collection_name=request.collection, persist_directory=db_path)
        
        # Parse filters
        chunk_filter = None
        if request.filters:
            from asc.retrieval.filters import ChunkFilter
            chunk_filter = ChunkFilter(
                source=request.filters.source,
                sources=request.filters.sources,
                min_perplexity=request.filters.min_perplexity,
                max_perplexity=request.filters.max_perplexity,
                min_sentences=request.filters.min_sentences,
                max_sentences=request.filters.max_sentences,
                chunk_type=request.filters.chunk_type
            )

        retriever = AdaptiveSemanticRetriever(
            vector_store=store,
            k=request.k,
            coherence_rerank=True,
            boundary_expand=True,
            use_hybrid_search=request.use_hybrid,
            mmr_lambda=request.mmr_lambda,
            vector_weight=request.vector_weight if request.vector_weight is not None else 0.6,
            bm25_weight=request.bm25_weight if request.bm25_weight is not None else 0.4,
            active_filters=chunk_filter
        )
        
        docs = await retriever._aget_relevant_documents(request.query)
        q_emb = await store.embedder.aembed_query(request.query)
        
        docs_out = []
        scores_out = []
        for doc in docs:
            docs_out.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
            
            # Recalculate exact cosine similarity score
            doc_emb = retriever._get_document_embedding(doc)
            q = np.array(q_emb)
            d = np.array(doc_emb)
            norm_q = np.linalg.norm(q)
            norm_d = np.linalg.norm(d)
            if norm_q == 0.0 or norm_d == 0.0:
                score = 0.0
            else:
                score = float(np.dot(q, d) / (norm_q * norm_d))
            scores_out.append(score)
            
        return {
            "documents": docs_out,
            "scores": scores_out
        }
    except Exception as e:
        logger.error(f"Error in /retrieve: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag", tags=["RAG Pipeline"])
async def rag_endpoint(request: QueryRequest) -> Any:
    """Query documents through the full LangGraph self-correcting RAG pipeline."""
    try:
        db_path = os.environ.get("CHROMA_PERSIST_DIR", "./data/chromadb")
        store = ASCVectorStore(collection_name=request.collection, persist_directory=db_path)
        
        # Parse filters
        chunk_filter = None
        if request.filters:
            from asc.retrieval.filters import ChunkFilter
            chunk_filter = ChunkFilter(
                source=request.filters.source,
                sources=request.filters.sources,
                min_perplexity=request.filters.min_perplexity,
                max_perplexity=request.filters.max_perplexity,
                min_sentences=request.filters.min_sentences,
                max_sentences=request.filters.max_sentences,
                chunk_type=request.filters.chunk_type
            )

        retriever = AdaptiveSemanticRetriever(
            vector_store=store,
            k=request.k,
            coherence_rerank=True,
            boundary_expand=True,
            use_hybrid_search=request.use_hybrid,
            mmr_lambda=request.mmr_lambda,
            vector_weight=request.vector_weight if request.vector_weight is not None else 0.6,
            bm25_weight=request.bm25_weight if request.bm25_weight is not None else 0.4,
            active_filters=chunk_filter
        )
        
        # Load LLM from config
        from asc.utils.model_factory import get_llm_from_env
        llm = get_llm_from_env()
        
        res = await run_rag(request.query, retriever, llm)
        
        # Map sources
        docs = await retriever._aget_relevant_documents(request.query)
        sources_list = []
        for doc in docs:
            sources_list.append({
                "source": doc.metadata.get("source", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "page_content": doc.page_content[:200]
            })
            
        return {
            "answer": res["answer"],
            "entailment_score": res["entailment_score"],
            "sources": sources_list,
            "iterations": res["iterations"]
        }
    except Exception as e:
        logger.error(f"Error in /rag: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", tags=["Status"])
async def health_endpoint() -> Any:
    """Get system health, checking connections to LLM/Embedding provider and ChromaDB."""
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    provider_ok = False
    chroma_ok = False
    models_loaded = []
    
    if provider == "nvidia":
        api_key = os.environ.get("NVIDIA_API_KEY")
        base_url = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                headers = {"Authorization": f"Bearer {api_key}"}
                r = await client.get(f"{base_url}/models", headers=headers)
                if r.status_code == 200:
                    provider_ok = True
                    models_loaded = [m["id"] for m in r.json().get("data", [])]
        except Exception:
            pass
    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                headers = {"Authorization": f"Bearer {api_key}"}
                r = await client.get(f"{base_url}/models", headers=headers)
                if r.status_code == 200:
                    provider_ok = True
                    models_loaded = [m["id"] for m in r.json().get("data", [])]
        except Exception:
            pass
    else:
        # Default Ollama
        ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{ollama_url}/api/tags")
                if r.status_code == 200:
                    provider_ok = True
                    data = r.json()
                    models_loaded = [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        
    db_path = os.environ.get("CHROMA_PERSIST_DIR", "./data/chromadb")
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        client.heartbeat()
        chroma_ok = True
    except Exception:
        pass
        
    return {
        "status": "ok",
        "provider": provider,
        "provider_connected": provider_ok,
        "ollama": provider_ok if provider == "ollama" else False,
        "chromadb": chroma_ok,
        "models_loaded": models_loaded[:10]
    }


@app.get("/sources", tags=["Status"])
async def get_sources_endpoint(collection: str = "default") -> Any:
    """Get list of unique source document names in a collection."""
    try:
        db_path = os.environ.get("CHROMA_PERSIST_DIR", "./data/chromadb")
        store = ASCVectorStore(collection_name=collection, persist_directory=db_path)
        stats = store.get_collection_stats()
        return stats.get("sources", [])
    except Exception as e:
        logger.error(f"Error in /sources: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections", tags=["Status"])
async def collections_endpoint() -> Any:
    """List all database collections and summarize their chunk statistics."""
    try:
        db_path = os.environ.get("CHROMA_PERSIST_DIR", "./data/chromadb")
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        cols = client.list_collections()
        col_names = [c.name for c in cols]
        
        stats = {}
        for name in col_names:
            store = ASCVectorStore(collection_name=name, persist_directory=db_path)
            stats[name] = store.get_collection_stats()
            
        return {
            "collections": col_names,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error in /collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/{path_name:path}", include_in_schema=False)
async def catch_all(path_name: str):
    """Catch-all route to serve frontend assets and route all fallback requests to index.html for SPA."""
    if os.path.exists(frontend_dist):
        file_path = os.path.join(frontend_dist, path_name)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        index_path = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
    return {"detail": "Not Found"}

