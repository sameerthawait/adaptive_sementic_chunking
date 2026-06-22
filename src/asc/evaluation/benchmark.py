


"""Benchmark runner for Adaptive Semantic Chunking.

Compares ASC against RecursiveCharacterTextSplitter at multiple fixed sizes,
computes precision, recall, MRR, coherence, boundary distinctness, and saves plots.
"""

import os
import re
import json
import time
import logging
import asyncio
import shutil
import nltk
import numpy as np
from dataclasses import dataclass, asdict
from typing import Any, List, Dict, Set, Optional

import matplotlib
matplotlib.use('Agg') # Headless execution check to prevent GUI crashes
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from asc.chunker.adaptive_chunker import AdaptiveSemanticChunker
from asc.chunker.boundary_detector import BoundaryDetector, BoundaryDetectorConfig
from asc.chunker.perplexity_scorer import PerplexityScorer
from asc.vectorstore.chroma_store import ASCVectorStore
from asc.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    chunk_coherence_score,
    boundary_quality_score,
    chunk_size_stats,
    calculate_chunk_statistics
)

logger = logging.getLogger(__name__)

# Preloaded Wikipedia articles for offline fallback
MOCK_ARTICLES = [
    {
        "title": "Artificial Intelligence",
        "text": "Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions that maximize its chance of achieving its goals. The term artificial intelligence had previously been used to describe machines that mimic and display human cognitive skills. However, this definition has since been rejected by major AI researchers who now describe AI in terms of rationality and acting rationally."
    },
    {
        "title": "Deep Learning",
        "text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional neural networks and transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, and climate science."
    },
    {
        "title": "Natural Language Processing",
        "text": "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. Goal of NLP is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents as well as categorize and organize the style itself."
    },
    {
        "title": "Vector Database",
        "text": "A vector database is a database that stores data as high-dimensional vectors, which are mathematical representations of features or attributes. Each vector has a certain number of dimensions, which can range from tens to thousands, depending on the complexity of the data. The data stored in a vector database can be indexed and queried using vector search algorithms, which locate vectors based on their similarity rather than exact matches. This allows for similarity searches that find similar items rather than exact keyword matches."
    },
    {
        "title": "Retrieval-Augmented Generation",
        "text": "Retrieval-Augmented Generation (RAG) is a technique for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge. By retrieving relevant passages of text from a document collection or database before generating a response, the LLM can generate more accurate, factual, and up-to-date answers. RAG addresses the limitations of standard language models, which can hallucinate information not present in their training parameters, by forcing them to use retrieved facts."
    },
    {
        "title": "Signal Processing",
        "text": "Signal processing is an electrical engineering subfield that focuses on analyzing, modifying, and synthesizing signals, such as sound, images, and scientific measurements. Signal processing techniques are used to improve transmission, storage efficiency, and subjective quality, and to emphasize or detect components of interest in a measured signal. Historically, signal processing began with analog systems, but has since shifted almost entirely to digital computers using discrete signal formulas."
    },
    {
        "title": "Cosine Similarity",
        "text": "Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space. It defined as the cosine of the angle between them, which is also the inner product of the vectors normalized to both have length 1. Cosine similarity is particularly used in positive-space high-dimensional positive spaces, such as information retrieval and text mining, where each term is assigned a unique dimension and documents are represented as vectors of term frequencies."
    },
    {
        "title": "Information Retrieval",
        "text": "Information retrieval (IR) is the software science of searching for information in a document, searching for documents themselves, and also searching for metadata that describes data, and for databases of texts, images or sounds. Automated information retrieval systems are used to reduce information overload. An IR process begins when a user enters a query into the system. Queries are formal statements of information needs, for example search strings in web search engines. In information retrieval, a query does not uniquely identify a single object."
    },
    {
        "title": "Machine Learning",
        "text": "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Recently, generative artificial intelligence networks have surpassed traditional expert systems due to representations learned from massive datasets. Machine learning algorithms are used in a wide variety of applications, such as email filtering, computer vision, and speech recognition."
    },
    {
        "title": "Neural Networks",
        "text": "Artificial neural networks (ANNs), usually simply called neural networks (NNs), are computing systems inspired by the biological neural networks that constitute animal brains. An ANN is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain. Each connection, like the synapses in a biological brain, can transmit a signal to other neurons. An artificial neuron receives signals, processes them, and can signal neurons connected to it."
    }
]

DEFAULT_SAMPLE_TEXT = (
    "Artificial intelligence has underwent rapid evolution since the mid-20th century. "
    "Early pioneers like Alan Turing and John McCarthy laid the theoretical foundations of computation and symbolic systems. "
    "During the early decades, rules-based logic and expert systems dominated the academic discourse. "
    "These systems relied on hand-coded heuristics to solve specific, highly constrained mathematical problems. "
    "However, progress stalled during the periods known as 'AI winters,' when funding dried up due to over-inflated expectations.\n\n"
    "In the 2010s, a dramatic paradigm shift occurred with the rise of deep learning and neural network architectures. "
    "This revolution was fueled by two primary drivers: massive datasets and powerful parallel GPU computation. "
    "Instead of hand-crafting rules, engineers trained large models to learn representations directly from data. "
    "ImageNet classifications and natural language tasks began falling to deep architectures with unprecedented accuracy. "
    "Convolutional networks and recurrent models became the standard backbones of commercial software.\n\n"
    "Today, large language models (LLMs) represent the frontier of generative artificial intelligence. "
    "Models like GPT-4 and Llama are trained on trillions of tokens, displaying emergent reasoning and contextual adaptation. "
    "These systems leverage self-attention mechanisms to process long sequences of tokens simultaneously. "
    "However, LLMs suffer from well-documented flaws, including hallucination and high compute overhead. "
    "Consequently, architectures like Retrieval-Augmented Generation (RAG) are used to ground outputs in external databases.\n\n"
    "Separately, molecular biology is undergoing a parallel computational revolution. "
    "Proteins are the workhorses of the cell, folded into complex three-dimensional structures. "
    "For fifty years, predicting a protein's structure from its raw amino acid sequence was an unsolved grand challenge. "
    "Then, deep learning architectures like AlphaFold solved the folding problem with near-experimental accuracy. "
    "This breakthrough has accelerated drug discovery, agricultural bio-engineering, and basic cellular research worldwide. "
    "Biologists can now model protein-protein interactions in seconds, transforming molecular sciences."
)


@dataclass
class BenchmarkResult:
    method_name: str
    precision_at_k: dict[int, float]    # k: 1, 3, 5
    recall_at_k: dict[int, float]
    mrr: float
    avg_coherence: float
    avg_boundary_quality: float
    chunk_size_stats: dict
    total_chunks: int
    runtime_seconds: float


class ASCBenchmark:
    """
    Compares AdaptiveSemanticChunker vs RecursiveCharacterTextSplitter
    at multiple fixed chunk sizes.
    
    Uses WikipediaLoader for reproducible test data.
    """
    
    def __init__(
        self,
        model: str | None = None,
        ollama_url: str | None = None
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        
        # Instantiate LLM wrapper based on active provider
        from asc.utils.model_factory import get_llm_from_env
        try:
            self.llm = get_llm_from_env()
        except Exception:
            self.llm = None
                
        # Internal scores store for t-test
        self._query_scores = {}

    async def run(
        self,
        n_wikipedia_articles: int = 10,
        queries: list[dict] | None = None,   # [{"query": str, "relevant_chunks": list[str]}]
        k_values: list[int] = [1, 3, 5],
        fixed_chunk_sizes: list[int] = [256, 512, 1024]
    ) -> list[BenchmarkResult]:
        """
        If queries is None, auto-generates 20 test queries from article titles
        using the LLM.
        """
        logger.info("Setting up Wikipedia articles...")
        articles = []
        
        # Attempt WikipediaLoader
        try:
            from langchain_community.document_loaders import WikipediaLoader
            topics = [
                "Artificial intelligence", "Deep learning", "Natural language processing",
                "Vector database", "Retrieval-augmented generation", "Signal processing",
                "Cosine similarity", "Information retrieval", "Machine learning", "Neural network"
            ]
            for topic in topics[:n_wikipedia_articles]:
                try:
                    loader = WikipediaLoader(query=topic, load_max_docs=1)
                    # Load in thread to keep loop free
                    loop = asyncio.get_running_loop()
                    docs = await loop.run_in_executor(None, loader.load)
                    if docs:
                        articles.extend(docs)
                except Exception as e:
                    logger.warning(f"Could not load article '{topic}': {e}")
        except ImportError:
            logger.info("WikipediaLoader not available. Falling back to default mock articles.")
            
        # Fallback if loader failed or was missing
        if len(articles) < n_wikipedia_articles:
            needed = n_wikipedia_articles - len(articles)
            for i in range(min(needed, len(MOCK_ARTICLES))):
                item = MOCK_ARTICLES[i]
                articles.append(Document(page_content=item["text"], metadata={"title": item["title"], "source": f"{item['title']}.txt"}))
                
        logger.info(f"Loaded {len(articles)} documents for benchmarking.")
        
        # 2. Auto-generate queries if not provided
        if queries is None:
            queries = []
            logger.info("Generating search questions via LLM...")
            for idx, art in enumerate(articles):
                title = art.metadata.get("title", f"Article {idx}")
                summary = art.page_content[:400]
                prompt = (
                    f"Given this Wikipedia article title: '{title}' and summary:\n"
                    f"'{summary}'\n\n"
                    f"Generate exactly two realistic, short search queries that can be answered by this text. "
                    f"Return only the queries, one per line. Do not number them."
                )
                try:
                    if self.llm:
                        res = await self.llm.ainvoke(prompt)
                        if hasattr(res, "content"):
                            res = res.content
                        lines = [l.strip() for l in str(res).split("\n") if l.strip()]
                        for line in lines[:2]:
                            # Strip numbering or prefixes
                            line = re.sub(r'^\d+[\.\)]\s*', '', line)
                            queries.append({
                                "query": line,
                                "relevant_chunks": [art.page_content[:400]]
                            })
                    else:
                        queries.append({
                            "query": f"Explain key details about {title}",
                            "relevant_chunks": [art.page_content[:400]]
                        })
                except Exception as e:
                    logger.warning(f"LLM query generation failed: {e}")
                    queries.append({
                        "query": f"Explain key details about {title}",
                        "relevant_chunks": [art.page_content[:400]]
                    })
            # Trim to max 20 queries
            queries = queries[:20]

        logger.info(f"Using {len(queries)} query evaluations.")
        
        results = []
        # We reuse one persistent Chroma DB location
        db_path = "./chroma_benchmark_db"
        
        # Loop through Methods: ASC and Fixed baseline splits
        methods = ["ASC"] + [f"Recursive_{size}" for size in fixed_chunk_sizes]
        
        for method in methods:
            t0 = time.time()
            chunks = []
            
            # Setup splitters
            if method == "ASC":
                import os
                provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
                if provider in ["nvidia", "openai"]:
                    chunker = AdaptiveSemanticChunker.from_env()
                else:
                    chunker = AdaptiveSemanticChunker(model=self.model or "llama3.2:3b", ollama_url=self.ollama_url or "http://localhost:11434")
                # Chunk documents asynchronously
                chunks = await chunker.chunk_documents(articles, show_progress=False)
            else:
                size = int(method.split("_")[1])
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=size,
                    chunk_overlap=int(size * 0.15),
                    length_function=len
                )
                # Run sync split in loop executor
                loop = asyncio.get_running_loop()
                chunks = await loop.run_in_executor(None, lambda: splitter.create_documents([a.page_content for a in articles], [a.metadata for a in articles]))
                # Format chunk index metadata
                for idx, c in enumerate(chunks):
                    c.metadata["chunk_index"] = idx
                    c.metadata["chunk_type"] = "fixed"
                    
            runtime = time.time() - t0
            
            # Measure stats
            size_stats = chunk_size_stats(chunks)
            total_chunks = len(chunks)
            
            # Index chunks in temporary ASCVectorStore to evaluate retrieval
            store = ASCVectorStore(
                collection_name=f"col_{method.lower()}",
                persist_directory=db_path
            )
            store.reset()
            # Index documents async
            await store.add_documents(chunks)
            
            # Compute Coherence & Boundary quality
            coherence_scores = []
            boundary_scores = []
            
            for c in chunks:
                sentences = nltk.sent_tokenize(c.page_content)
                coherence_scores.append(chunk_coherence_score(sentences, store.embedder))
                
            # Boundary Quality score between adjacent chunks of same source
            # Map chunks by source
            source_chunks = {}
            for c in chunks:
                src = c.metadata.get("source", "unknown")
                source_chunks.setdefault(src, []).append(c)
                
            for src, src_list in source_chunks.items():
                src_list.sort(key=lambda d: d.metadata.get("chunk_index", 0))
                for idx in range(len(src_list) - 1):
                    sents_a = nltk.sent_tokenize(src_list[idx].page_content)
                    sents_b = nltk.sent_tokenize(src_list[idx + 1].page_content)
                    boundary_scores.append(boundary_quality_score(sents_a, sents_b, store.embedder))
                    
            avg_coherence = float(np.mean(coherence_scores)) if coherence_scores else 1.0
            avg_boundary_quality = float(np.mean(boundary_scores)) if boundary_scores else 1.0
            
            # Evaluation Retrieval loop
            p_at_k_totals = {k: [] for k in k_values}
            r_at_k_totals = {k: [] for k in k_values}
            mrr_list = []
            
            # Helper logic to match document hit
            def is_hit(ret_text: str, rel_texts: list[str]) -> bool:
                for rel in rel_texts:
                    if rel.lower() in ret_text.lower() or ret_text.lower() in rel.lower():
                        return True
                    w1 = set(ret_text.lower().split())
                    w2 = set(rel.lower().split())
                    if w1 and w2:
                        intersection = w1.intersection(w2)
                        overlap = len(intersection) / min(len(w1), len(w2))
                        if overlap >= 0.5:
                            return True
                return False

            for q_item in queries:
                q_text = q_item["query"]
                rel_chunks = q_item["relevant_chunks"]
                
                # Fetch matching documents (over-retrieve up to max k)
                max_k = max(k_values)
                ret_results = await store.similarity_search(q_text, k=max_k)
                
                # Get list of Boolean indicators (hit = True, miss = False)
                retrieved_indicators = []
                for doc, sim in ret_results:
                    retrieved_indicators.append("HIT" if is_hit(doc.page_content, rel_chunks) else "MISS")
                    
                # We treat relevant target as set {"HIT"} and compute indices
                # retrieved_ids list: e.g. ["HIT", "MISS", "HIT"]
                # relevant_ids set: {"HIT"}
                retrieved_ids = retrieved_indicators
                relevant_ids = {"HIT"}
                
                for k in k_values:
                    p_at_k_totals[k].append(precision_at_k(retrieved_ids, relevant_ids, k))
                    r_at_k_totals[k].append(recall_at_k(retrieved_ids, relevant_ids, k))
                    
                mrr_list.append(mean_reciprocal_rank(retrieved_ids, relevant_ids))
                
            # Aggregate stats
            p_final = {k: float(np.mean(p_at_k_totals[k])) for k in k_values}
            r_final = {k: float(np.mean(r_at_k_totals[k])) for k in k_values}
            mrr_final = float(np.mean(mrr_list))
            
            # Store in private query score lists for statistical significance (t-test)
            self._query_scores[method] = {
                "mrr": mrr_list,
                "precision_3": p_at_k_totals.get(3, [0.0]),
                "recall_3": r_at_k_totals.get(3, [0.0])
            }
            
            res = BenchmarkResult(
                method_name=method,
                precision_at_k=p_final,
                recall_at_k=r_final,
                mrr=mrr_final,
                avg_coherence=avg_coherence,
                avg_boundary_quality=avg_boundary_quality,
                chunk_size_stats=size_stats,
                total_chunks=total_chunks,
                runtime_seconds=runtime
            )
            results.append(res)
            logger.info(f"Completed benchmark for: {method} (MRR: {mrr_final:.4f}, Coherence: {avg_coherence:.4f})")
            
            # Clean collection
            store.reset()
            
        # Clean local DB directory
        try:
            shutil.rmtree(db_path)
        except Exception:
            pass
            
        return results
        
    def generate_report(self, results: list[BenchmarkResult]) -> str:
        """
        Generates markdown table:
        | Method | P@3 | R@3 | MRR | Coherence | Boundary Quality |
        Highlights ASC wins in bold. Includes statistical significance (t-test p-values).
        """
        # Find which result is ASC
        asc_res = None
        for r in results:
            if r.method_name == "ASC":
                asc_res = r
                break
                
        # Compare ASC metrics against baseline averages to check wins
        baselines = [r for r in results if r.method_name != "ASC"]
        
        def is_win(metric_name: str, asc_val: float) -> bool:
            if not baselines:
                return False
            if metric_name == "p_3":
                base_vals = [b.precision_at_k.get(3, 0.0) for b in baselines]
            elif metric_name == "r_3":
                base_vals = [b.recall_at_k.get(3, 0.0) for b in baselines]
            elif metric_name == "mrr":
                base_vals = [b.mrr for b in baselines]
            elif metric_name == "coherence":
                base_vals = [b.avg_coherence for b in baselines]
            elif metric_name == "boundary":
                base_vals = [b.avg_boundary_quality for b in baselines]
            else:
                return False
            return asc_val > np.max(base_vals)
            
        # Calculate t-test p-values comparing ASC vs baseline averages
        p_mrr = p_p3 = p_r3 = 1.0
        if asc_res and baselines:
            asc_scores = self._query_scores.get("ASC", {})
            baseline_mrr = []
            baseline_p3 = []
            baseline_r3 = []
            for b in baselines:
                scores = self._query_scores.get(b.method_name, {})
                baseline_mrr.extend(scores.get("mrr", []))
                baseline_p3.extend(scores.get("precision_3", []))
                baseline_r3.extend(scores.get("recall_3", []))
                
            if asc_scores and baseline_mrr:
                p_mrr = float(ttest_ind(asc_scores.get("mrr", []), baseline_mrr, equal_var=False).pvalue)
                p_p3 = float(ttest_ind(asc_scores.get("precision_3", []), baseline_p3, equal_var=False).pvalue)
                p_r3 = float(ttest_ind(asc_scores.get("recall_3", []), baseline_r3, equal_var=False).pvalue)
                
        report = []
        report.append("# Adaptive Semantic Chunking (ASC) Benchmark Report")
        report.append("\nThis report compares the research-grade ASC method against traditional fixed-size recursive character splitters.\n")
        
        # Table Header
        report.append("| Method | P@3 | R@3 | MRR | Coherence | Boundary Quality | Runtime (s) | Chunks |")
        report.append("|---|---|---|---|---|---|---|---|")
        
        for r in results:
            m_name = r.method_name
            p3 = r.precision_at_k.get(3, 0.0)
            r3 = r.recall_at_k.get(3, 0.0)
            mrr = r.mrr
            coh = r.avg_coherence
            bq = r.avg_boundary_quality
            rtime = r.runtime_seconds
            tchunks = r.total_chunks
            
            # Format and apply bold highlighting for wins
            cell_p3 = f"**{p3:.4f}**" if m_name == "ASC" and is_win("p_3", p3) else f"{p3:.4f}"
            cell_r3 = f"**{r3:.4f}**" if m_name == "ASC" and is_win("r_3", r3) else f"{r3:.4f}"
            cell_mrr = f"**{mrr:.4f}**" if m_name == "ASC" and is_win("mrr", mrr) else f"{mrr:.4f}"
            cell_coh = f"**{coh:.4f}**" if m_name == "ASC" and is_win("coherence", coh) else f"{coh:.4f}"
            cell_bq = f"**{bq:.4f}**" if m_name == "ASC" and is_win("boundary", bq) else f"{bq:.4f}"
            
            report.append(f"| {m_name} | {cell_p3} | {cell_r3} | {cell_mrr} | {cell_coh} | {cell_bq} | {rtime:.2f} | {tchunks} |")
            
        report.append("\n### Statistical Significance (t-test vs Baselines)")
        report.append(f"- **MRR p-value**: {p_mrr:.4e} " + ("(Statistically Significant, p < 0.05)" if p_mrr < 0.05 else "(Not Significant)"))
        report.append(f"- **P@3 p-value**: {p_p3:.4e} " + ("(Statistically Significant, p < 0.05)" if p_p3 < 0.05 else "(Not Significant)"))
        report.append(f"- **R@3 p-value**: {p_r3:.4e} " + ("(Statistically Significant, p < 0.05)" if p_r3 < 0.05 else "(Not Significant)"))
        
        return "\n".join(report)
        
    def plot_comparison(
        self,
        results: list[BenchmarkResult],
        save_dir: str = "./docs/benchmark_charts"
    ) -> None:
        """
        4 side-by-side bar charts using seaborn:
        1. Precision@K comparison
        2. Coherence scores
        3. Chunk size distribution (violin plot)
        4. Boundary quality scores
        Color ASC bars in electric blue, fixed chunking in gray.
        Save as PNG + SVG for algorithm documentation.
        """
        os.makedirs(save_dir, exist_ok=True)
        sns.set_theme(style="whitegrid")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Color palette: electric blue for ASC, gray for recursive baselines
        def get_colors(m_names):
            return ["#0052FF" if "ASC" in name else "#9E9E9E" for name in m_names]
            
        methods = [r.method_name for r in results]
        colors = get_colors(methods)
        
        # Chart 1: Precision@3 & Recall@3 & MRR Comparison
        x = np.arange(len(methods))
        width = 0.25
        
        axes[0, 0].bar(x - width, [r.precision_at_k.get(3, 0.0) for r in results], width, label="P@3", color="#29B6F6")
        axes[0, 0].bar(x, [r.recall_at_k.get(3, 0.0) for r in results], width, label="R@3", color="#AB47BC")
        axes[0, 0].bar(x + width, [r.mrr for r in results], width, label="MRR", color="#0052FF")
        
        axes[0, 0].set_title("1. Retrieval Quality Metrics", fontsize=14, weight="bold")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(methods, rotation=15)
        axes[0, 0].legend()
        
        # Chart 2: Coherence scores
        axes[0, 1].bar(methods, [r.avg_coherence for r in results], color=colors, edgecolor="black", width=0.5)
        axes[0, 1].set_title("2. Semantic Coherence Scores (Higher is Better)", fontsize=14, weight="bold")
        axes[0, 1].set_ylabel("Coherence (Cosine Similarity)")
        axes[0, 1].tick_params(axis='x', rotation=15)
        
        # Chart 3: Boundary quality scores
        axes[1, 0].bar(methods, [r.avg_boundary_quality for r in results], color=colors, edgecolor="black", width=0.5)
        axes[1, 0].set_title("3. Boundary Quality Scores (Higher is Better)", fontsize=14, weight="bold")
        axes[1, 0].set_ylabel("Distinctness (1 - Cosine Similarity)")
        axes[1, 0].tick_params(axis='x', rotation=15)
        
        # Chart 4: Coefficient of Variation of chunk sizes
        cvs = [r.chunk_size_stats.get("cv", 0.0) for r in results]
        axes[1, 1].bar(methods, cvs, color=colors, edgecolor="black", width=0.5)
        axes[1, 1].set_title("4. Chunk Size Variation (CV)", fontsize=14, weight="bold")
        axes[1, 1].set_ylabel("Coefficient of Variation")
        axes[1, 1].tick_params(axis='x', rotation=15)
        
        plt.tight_layout()
        
        png_path = os.path.join(save_dir, "benchmark_comparison.png")
        svg_path = os.path.join(save_dir, "benchmark_comparison.svg")
        
        plt.savefig(png_path, dpi=150)
        plt.savefig(svg_path, format="svg")
        logger.info(f"Saved charts to '{png_path}' and '{svg_path}'")
        plt.close()


# Backward compatible run_benchmark function for main.py:
async def run_benchmark(
    text: str | None = None,
    output_image_path: str = "evaluation_results.png",
    model: str | None = None,
    ollama_url: str | None = None,
) -> dict[str, Any]:
    """Runs a single document benchmark and outputs standard visual charts (backward-compat wrapper)."""
    text = text or DEFAULT_SAMPLE_TEXT
    
    import os
    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
    if provider in ["nvidia", "openai"]:
        asc_chunker = AdaptiveSemanticChunker.from_env()
    else:
        asc_chunker = AdaptiveSemanticChunker(
            model=model or "llama3.2:3b",
            ollama_url=ollama_url or "http://localhost:11434",
            window_size=3,
            savgol_window=5,
            savgol_polyorder=2,
            z_threshold=1.1,
            rolling_lag=4,
            min_sentences_per_chunk=2,
        )
    
    sentences = nltk.sent_tokenize(text)
    n_sentences = len(sentences)
    
    # Score perplexities
    perplexities = await asc_chunker._score_sentences(sentences)
    boundaries, diagnostics = asc_chunker.detector.detect_boundaries(np.array(perplexities))
    smoothed = diagnostics["smoothed_scores"]
    z_scores = diagnostics["z_scores"]
    
    # Compile ASC Chunks
    asc_chunks = []
    for idx in range(len(boundaries) - 1):
        start_idx = boundaries[idx]
        end_idx = boundaries[idx + 1]
        chunk_text = " ".join(sentences[start_idx:end_idx])
        asc_chunks.append(Document(page_content=chunk_text, metadata={"chunk_index": idx}))
        
    asc_stats = calculate_chunk_statistics(asc_chunks)
    target_chunk_size = max(200, int(asc_stats["mean"]))
    
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_chunk_size,
        chunk_overlap=int(target_chunk_size * 0.15),
        length_function=len,
    )
    recursive_chunks = recursive_splitter.create_documents([text])
    rec_stats = calculate_chunk_statistics(recursive_chunks)
    
    # Save standard plot comparison
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    x_indices = np.arange(n_sentences)
    
    axes[0].plot(x_indices, perplexities, 'o--', color='lightgray', label='Raw Perplexity')
    axes[0].plot(x_indices, smoothed, 'r-', linewidth=2, label='Savitzky-Golay Smoothed')
    for idx, b in enumerate(boundaries):
        if b < n_sentences:
            axes[0].axvline(x=b, color='darkblue', linestyle='--', alpha=0.7)
            if b > 0:
                axes[0].text(b + 0.1, max(perplexities) * 0.95, f"B{idx}", color='darkblue', weight='bold')
    axes[0].set_title("Perplexity Boundaries & Signal Smoothing", fontsize=14, weight='bold')
    axes[0].set_ylabel("Perplexity")
    axes[0].legend()
    
    axes[1].plot(x_indices, z_scores, 'g-', linewidth=2, label='Rolling Z-Score')
    axes[1].axhline(y=asc_chunker.detector.config.z_score_threshold, color='orange', linestyle=':', label='Threshold')
    for b in boundaries:
        if 0 < b < n_sentences:
            axes[1].plot(b, z_scores[b], 'x', color='darkblue', markersize=10, weight='bold')
    axes[1].set_title("Rolling Z-Score Surprise Spikes", fontsize=14, weight='bold')
    axes[1].set_ylabel("Z-Score")
    axes[1].legend()
    
    asc_lens = [len(c.page_content) for c in asc_chunks]
    rec_lens = [len(c.page_content) for c in recursive_chunks]
    if len(asc_lens) > 1 and len(rec_lens) > 1:
        sns.kdeplot(asc_lens, fill=True, label="ASC (Variable)", ax=axes[2], color="blue")
        sns.kdeplot(rec_lens, fill=True, label="Recursive (Fixed)", ax=axes[2], color="orange")
    else:
        axes[2].bar(["ASC", "Recursive"], [np.mean(asc_lens), np.mean(rec_lens)], color=["blue", "orange"])
    axes[2].set_title("Chunk Character Size Distributions", fontsize=14, weight='bold')
    axes[2].set_xlabel("Chunk Character Length")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=150)
    plt.close()
    
    return {
        "asc": asc_stats,
        "recursive": rec_stats,
        "boundaries": boundaries,
        "perplexities": perplexities,
    }


# Auto-execution check when run standalone
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    async def run_standalone_benchmark():
        logger.info("Starting ASC Benchmark standalone execution...")
        benchmark = ASCBenchmark()
        
        # Run comparison with 3 documents
        results = await benchmark.run(n_wikipedia_articles=3)
        
        # Save results JSON
        docs_dir = "./docs"
        os.makedirs(docs_dir, exist_ok=True)
        results_json_path = os.path.join(docs_dir, "benchmark_results.json")
        
        serializable_results = [asdict(r) for r in results]
        with open(results_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=4)
        logger.info(f"Saved benchmark results JSON to: '{results_json_path}'")
        
        # Generate and save report
        report = benchmark.generate_report(results)
        report_md_path = os.path.join(docs_dir, "benchmark_report.md")
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Saved benchmark markdown report to: '{report_md_path}'")
        
        # Plot and save charts
        benchmark.plot_comparison(results)
        
        print("\n" + "="*80)
        print("          ASC BENCHMARK STANDALONE SUMMARY")
        print("="*80)
        print(report)
        print("="*80 + "\n")

    # Run
    try:
        asyncio.run(run_standalone_benchmark())
    except KeyboardInterrupt:
        print("Cancelled.")
