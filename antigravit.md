# CLAUDE.md — Adaptive Semantic Chunking (ASC)

## Project Identity

ASC is a research-grade RAG research tool that chunks documents using **LLM perplexity-based boundary detection** — a novel alternative to fixed-size chunking. Instead of splitting at arbitrary token counts, it measures how "surprising" each sentence is to a local LLM. High perplexity = semantic discontinuity = chunk boundary.

Two parts: `backend/` (Python FastAPI + LangChain + LangGraph) and `frontend/` (React + Vite).

---

## Mandatory Rules

- **Never modify** `chunker/boundary_detector.py` or `chunker/perplexity_scorer.py` without also updating `docs/algorithm.md` — this file documents the core research contribution.
- **Always async** — use `httpx`, never `requests`. All Python I/O is async.
- **Type hints on every function** — PEP 604 union syntax (`str | None`, not `Optional[str]`).
- **Google-style docstrings** on every class and public method.
- **No hardcoded hex colors** in frontend — use CSS custom properties (`--signal`, `--boundary`, etc.).
- **No neon colors** — accent is `--signal: #2C5F8A` (muted blue), boundary is `--boundary: #B45309` (warm amber).
- **All frontend API calls** go through `src/api/client.js` only. Never call `fetch` or `axios` directly in components.
- **All config via env vars** — never hardcode model names, ports, or paths in code.
- **Lucide React only** for icons. No other icon libraries.
- **Ollama for all LLM calls** — never call OpenAI or Anthropic directly unless explicitly asked.

---

## Tech Stack

**Backend**
- Python 3.11, FastAPI 0.115, uvicorn
- LangChain 0.3, LangGraph 0.2
- ChromaDB 0.5 (persistent vector store)
- Ollama (local LLMs: `llama3.2:3b` for perplexity, `nomic-embed-text` for embeddings)
- httpx (async HTTP), scipy + numpy (signal processing), nltk (sentence tokenization)
- pytest + pytest-asyncio

**Frontend**
- React 18 + Vite 5
- Tailwind CSS 3 (custom tokens in `tailwind.config.js`)
- Recharts 2 (perplexity signal chart — the hero visual)
- React Router v6, Axios, @tanstack/react-query v5
- Framer Motion, Lucide React, clsx

---

## Commands

```bash
# First-time setup (backend)
cd backend && bash setup.sh

# Backend dev server
source backend/venv/bin/activate
uvicorn src.asc.api.app:app --reload --port 8000

# Frontend dev server
cd frontend && npm run dev          # runs on http://localhost:3000

# Unit tests (no Ollama needed)
pytest tests/unit/ -v --tb=short

# Integration tests (requires Ollama running)
pytest tests/integration/ -v -m integration

# CLI usage
python main.py chunk input.txt --output chunks.json
python main.py index ./docs/ --collection my_project
python main.py query "What is ASC?" --collection my_project
python main.py benchmark --n-articles 10
python main.py serve

# Health check
curl http://localhost:8000/health
```

---

## Architecture

```
backend/src/asc/
├── chunker/
│   ├── perplexity_scorer.py   ← CORE ALGORITHM: Ollama log-prob extraction
│   ├── boundary_detector.py   ← CORE ALGORITHM: Savitzky-Golay + z-score spikes
│   └── adaptive_chunker.py    ← Orchestrator → LangChain Documents
├── vectorstore/chroma_store.py
├── retrieval/
│   ├── retriever.py           ← LangChain BaseRetriever + coherence re-ranking
│   ├── bm25_index.py          ← Keyword-based search and reciprocal rank fusion
│   ├── filters.py             ← Metadata filter query parser and translation
│   └── rag_pipeline.py        ← LangGraph: retrieve→grade→generate→verify loop
├── evaluation/
│   ├── benchmark.py           ← ASC vs fixed-size comparison
│   └── metrics.py             ← precision@k, coherence, boundary quality
└── api/app.py                 ← FastAPI, 6 endpoints

frontend/src/
├── api/client.js              ← Axios instance, all API functions
├── hooks/                     ← useChunk, useRag, useHealth
├── components/
│   ├── layout/                ← Sidebar (220px dark), TopBar, Layout
│   ├── ui/                    ← Button, Badge, Card, Spinner, Toast, Empty
│   ├── chunker/               ← PerplexityChart (hero), HeatmapStrip, ChunkCard
│   ├── query/                 ← SearchBar, AnswerCard, SourceChunk
│   ├── index/                 ← DropZone, DocumentTable
│   └── benchmark/             ← MetricsTable, ComparisonCharts
└── pages/                     ← Dashboard, Chunker, Query, Index, Benchmark
```

---

## Key Environment Variables

```bash
# backend/.env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_PERPLEXITY_MODEL=llama3.2:3b
OLLAMA_EMBED_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./data/chromadb
Z_SCORE_THRESHOLD=2.0
MIN_CHUNK_SENTENCES=3
PERPLEXITY_CONTEXT_WINDOW=3

# frontend/.env
VITE_API_URL=http://localhost:8000
```

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/chunk` | Chunk text → perplexity scores + boundaries + chunks |
| POST | `/index` | Chunk + embed + store in ChromaDB |
| POST | `/rag` | Full RAG pipeline → answer + entailment score |
| GET | `/health` | Ollama + ChromaDB status |
| GET | `/collections` | List ChromaDB collections |

---

## Linked Documentation

- `docs/algorithm.md` — algorithm documentation (Savitzky-Golay, z-score, log-prob extraction)
- `docs/benchmark_results.json` — latest benchmark output (auto-generated)