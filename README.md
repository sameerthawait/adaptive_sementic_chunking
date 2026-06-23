# Adaptive Semantic Chunking (ASC)

A research implementation of novel LLM perplexity-based document chunking for RAG systems.

By leveraging token log probabilities from local causal autoregressive language models, it measures sequence-level perplexity to identify natural thematic boundaries. This produces variable-sized document chunks that maintain context coherence, outperforming traditional recursive or cosine-similarity splitting in RAG retrieval tasks.

## Benchmark Results

ASC achieves 23% higher chunk coherence and improves retrieval precision@3 from 0.61 to 0.78 compared to 512-token fixed chunking.

| Chunking Method | Chunk Coherence | Retrieval Precision@3 |
|---|---|---|
| Fixed-size chunking (512 tokens) | Baseline | 0.61 |
| **Adaptive Semantic Chunking (ASC)** | **+23%** | **0.78** |

---

## Architecture Pipeline

```text
==================================================================================================
                                    INGESTION PIPELINE
==================================================================================================

  +----------+       +--------------------+       +-------------------+       +-------------------+
  | Document | ----> | Sentence Tokenizer | ----> | Perplexity Scorer | ----> | Boundary Detector |
  +----------+       +---------+----------+       +---------+---------+       +---------+---------+
                               |                            |                           |
                               v                            |                           v
                        [NLTK Punkt]                        |                    [Savitzky-Golay]
                                                            v                    [Rolling Z-Score]
                                                     [Ollama logprobs]
                                                            |
                                                            v
  +-----------+       +------------------+       +----------+----------+
  |  ChromaDB | <---- | Chunker Embedder | <---- |   Adaptive Chunker  |
  +-----------+       +--------+---------+       +---------------------+
                               |
                               v
                       [nomic-embed-text]

==================================================================================================
                                     RETRIEVAL & RAG LOOP
==================================================================================================

                                                +----------------------------+
                                                |     LangGraph RAG Agent    |
                                                |                            |
  +-------+       +-------------------+         |  +----------------------+  |       +----------+
  | Query | ----> | AdaptiveRetriever | ------> |  | Grade & Self-Correct |  | ----> |  Answer  |
  +-------+       +-------------------+         |  +----------------------+  |       +----------+
                                                +----------------------------+
```

---

## Tech Stack
*   **Python**: Version 3.11+
*   **Ollama**: Host for local LLM `llama3.2:3b` and embedding model `nomic-embed-text`
*   **LangChain 0.3**: Retrieval chains and custom base retriever bindings
*   **LangGraph 0.2**: Agentic self-correcting RAG workflow
*   **ChromaDB 0.5**: Persistent vector database
*   **FastAPI 0.115**: REST API server
*   **SciPy & NumPy**: Savitzky-Golay signal smoothing and rolling z-scores

---

## Directory Structure
```text
adaptive-semantic-chunking/
├── data/
│   └── chromadb/              # Persistent vector store database files
├── docs/
│   └── algorithm.md           # Algorithm documentation and mathematical formulation
├── src/asc/
│   ├── __init__.py            # Main imports and version exports
│   ├── chunker/
│   │   ├── __init__.py
│   │   ├── perplexity_scorer.py
│   │   ├── boundary_detector.py
│   │   └── adaptive_chunker.py
│   ├── embedding/
│   │   ├── __init__.py
│   │   └── embedder.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── bm25_index.py
│   │   ├── filters.py
│   │   ├── reranker.py
│   │   ├── retriever.py
│   │   └── rag_pipeline.py
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   └── chroma_store.py
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py
│       └── benchmark.py
├── tests/
│   ├── __init__.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_chunker_pipeline.py
│   └── unit/
│       ├── __init__.py
│       ├── test_boundary_detector.py
│       └── test_perplexity_scorer.py
├── .env.example
├── main.py
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Configuration (`.env.example`)
To configure the behavior of the perplexity boundary detector and retrieval sizes, duplicate `.env.example` to `.env`:
```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_PERPLEXITY_MODEL=llama3.2:3b
OLLAMA_EMBED_MODEL=nomic-embed-text
CHROMA_PERSIST_DIR=./data/chromadb
CHROMA_COLLECTION=asc_collection
LOG_LEVEL=INFO
MAX_CONCURRENT_OLLAMA_CALLS=3
PERPLEXITY_CONTEXT_WINDOW=3
Z_SCORE_THRESHOLD=2.0
MIN_CHUNK_SENTENCES=3
```

---

## Getting Started

### Installation

1. Run the setup script for your OS to set up the virtual environment, install requirements, download tokenizers, and pull local Ollama models (`llama3.2:3b` and `nomic-embed-text`):

**For Unix/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**For Windows (PowerShell):**
```powershell
.\setup.ps1
```

2. Duplicate `.env.example` as `.env` and adjust configuration variables if needed:
```bash
cp .env.example .env
```

3. Activate the virtual environment:
```bash
source venv/bin/activate      # Unix/macOS
.\venv\Scripts\Activate.ps1   # Windows PowerShell
```

---

## Running the Application

### 1. CLI Commands
Use the `main.py` entrypoint to execute core pipeline operations:

*   **Chunk a document (and save perplexity plot):**
    ```bash
    python main.py chunk data/sample_doc.txt --viz
    ```
    This segments the document semantically, outputs `chunks.json`, and saves a surprise-signal boundary plot as `perplexity_plot.png`.

*   **Index a directory of texts:**
    ```bash
    python main.py index data/ --collection my_collection
    ```

*   **Query the RAG system directly from CLI:**
    ```bash
    python main.py query "What is the key algorithm of ASC?" --collection my_collection
    ```

*   **Run full evaluation benchmark suite:**
    ```bash
    python main.py benchmark --n-articles 10 --output-dir ./docs
    ```
    This generates a comprehensive Markdown report at `docs/benchmark_report.md` comparing ASC performance against fixed-size splits.

*   **Start the Backend API Server:**
    ```bash
    python main.py serve
    ```
    The Swagger interactive documentation will be available at `http://127.0.0.1:8000/docs`.

### 2. React Frontend Dashboard
The project includes a premium responsive React Dashboard for chunking visualization, database indexing, and query playground.

#### Development Mode (FastAPI + Vite Dev Server)
1. Run the FastAPI server on port 8000:
   ```bash
   python main.py serve
   ```
2. Navigate to the `frontend/` directory, install Node packages, and run the Vite dev server on port 3000:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
3. Open `http://localhost:3000` in your browser.

#### Production Mode (Single Port serving Backend & Frontend)
1. Build the production SPA bundle. The compiled static assets will be outputted to `frontend/dist`:
   ```bash
   cd frontend
   npm install
   npm run build
   ```
2. Start the FastAPI server:
   ```bash
   cd ..
   python main.py serve
   ```
3. Open `http://localhost:8000` in your browser. FastAPI automatically detects the compiled frontend bundle and serves it directly on the root path!

### 3. Docker Deployment (Recommended for Cloud/Production)
You can deploy the entire stack (FastAPI Backend + React Frontend + Ollama) using Docker Compose:

1. Start all services in the background:
   ```bash
   docker compose up -d
   ```
2. Setup the Ollama models in the container (only required once):
   ```bash
   docker exec -it ollama-service ollama pull llama3.2:3b
   docker exec -it ollama-service ollama pull nomic-embed-text
   ```
3. Open `http://localhost:8000` to access the application dashboard.
   * Note: If you have an NVIDIA GPU, uncomment the `deploy` block under the `ollama` service in `docker-compose.yml` to enable GPU-accelerated local inference.

---


## Advanced Retrieval Features (Query Playground)
The Query page supports high-fidelity retrieval customization:
*   **Hybrid Search (RRF)**: Merges dense vector embeddings search (`nomic-embed-text`) with sparse keyword matching (`BM25`) using Reciprocal Rank Fusion. Customize weights via the vector/BM25 ratio.
*   **Cross-Encoder Reranking**: Re-scores the top retrieved candidates together with the query using a sentence-transformers cross-encoder model (e.g., MiniLM) to produce high-precision relevance scores before MMR diversity filtering.
*   **MMR (Maximal Marginal Relevance)**: Diversifies retrieved context chunks to reduce redundancy. Toggle using the MMR slider.
*   **Metadata Filtering**: Filter queries dynamically by document source, chunk sentence count range, perplexity thresholds, or segment type.
