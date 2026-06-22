"""Typer CLI entrypoint for the Adaptive Semantic Chunking research system.

Provides commands for chunking, indexing, querying, benchmarking, and serving the REST API.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import nltk
import typer
import uvicorn
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

from asc.chunker.adaptive_chunker import AdaptiveSemanticChunker
from asc.vectorstore.chroma_store import ASCVectorStore
from asc.retrieval.retriever import AdaptiveSemanticRetriever
from asc.retrieval.rag_pipeline import run_rag
from asc.evaluation.benchmark import ASCBenchmark

console = Console()
app = typer.Typer(help="Adaptive Semantic Chunking — Research-grade RAG system")


def _run_async(coro):
    """Runs async functions synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()


@app.command()
def chunk(
    input_file: Path = typer.Argument(..., help="Path to input text document."),
    output: Path = typer.Option("chunks.json", help="Path to output JSON file."),
    visualize: bool = typer.Option(False, "--viz", help="Whether to save the diagnostic perplexity plot.")
):
    """Chunk a document using adaptive semantic chunking."""
    if not input_file.exists():
        console.print(f"[bold red]Error:[/bold red] Input file '{input_file}' does not exist.")
        raise typer.Exit(1)
        
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    console.print(f"[bold green]ASC Chunker[/bold green] - Segmenting '{input_file.name}'...")
    
    chunker = AdaptiveSemanticChunker.from_env()
    
    async def _action():
        # Score and chunk
        sentences = nltk.sent_tokenize(text)
        ppls = await chunker._score_sentences(sentences)
        ppls_np = np.array(ppls)
        
        boundaries, diagnostics = chunker.detector.detect_boundaries(ppls_np)
        
        chunk_dicts = chunker._build_chunks_with_overlap(
            sentences=sentences,
            boundaries=boundaries,
            perplexity_scores=ppls_np,
            diagnostics=diagnostics,
            overlap=chunker.sentence_overlap
        )
        
        docs = chunker._to_documents(chunk_dicts, {"source": input_file.name})
        
        # Save output JSON
        serializable = []
        for doc in docs:
            serializable.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })
            
        with open(output, "w", encoding="utf-8") as out_f:
            json.dump(serializable, out_f, indent=4)
            
        if visualize:
            plot_path = "perplexity_plot.png"
            chunker.detector.visualize(
                perplexity_scores=ppls_np,
                boundaries=boundaries,
                sentences=sentences,
                save_path=plot_path,
                show=False
            )
            console.print(f"Saved perplexity signal plot to '{plot_path}'")
            
        return len(docs), float(np.mean(ppls_np))

    num_chunks, avg_perplexity = _run_async(_action())
    
    console.print(f"\n[bold green]Success![/bold green] Generated [bold cyan]{num_chunks}[/bold cyan] semantic chunks.")
    console.print(f"Average Document Perplexity: [bold yellow]{avg_perplexity:.2f}[/bold yellow]")
    console.print(f"Results saved to '[underline]{output}[/underline]'")


@app.command()
def index(
    input_dir: Path = typer.Argument(..., help="Directory containing documents to index."),
    collection: str = typer.Option("default", help="ChromaDB collection name."),
    pattern: str = typer.Option("*.txt", help="Glob pattern to filter input documents.")
):
    """Index all documents in a directory into ChromaDB."""
    if not input_dir.exists() or not input_dir.is_dir():
        console.print(f"[bold red]Error:[/bold red] Input directory '{input_dir}' is invalid.")
        raise typer.Exit(1)
        
    files = list(input_dir.glob(pattern))
    if not files:
        console.print(f"[bold yellow]Warning:[/bold yellow] No files matching '{pattern}' in '{input_dir}'.")
        return
        
    console.print(f"[bold green]ASC Vector Store[/bold green] - Indexing {len(files)} files into collection '{collection}'...")
    
    chunker = AdaptiveSemanticChunker.from_env()
    db_path = os.environ.get("CHROMA_PERSIST_DIR", "./data/chromadb")
    store = ASCVectorStore(collection_name=collection, persist_directory=db_path)
    
    async def _action():
        total_chunks = 0
        indexed_table = Table(title="Indexed Documents Summary")
        indexed_table.add_column("Filename", style="cyan")
        indexed_table.add_column("Size (chars)", style="magenta")
        indexed_table.add_column("Chunks", style="green")
        
        for f in files:
            with open(f, "r", encoding="utf-8") as open_f:
                content = open_f.read()
                
            docs = await chunker.chunk_text(content, {"source": f.name})
            await store.add_documents(docs)
            
            indexed_table.add_row(f.name, str(len(content)), str(len(docs)))
            total_chunks += len(docs)
            
        console.print(indexed_table)
        return total_chunks

    total = _run_async(_action())
    console.print(f"\n[bold green]Indexing completed successfully![/bold green] Added [bold]{total}[/bold] total chunks.")


@app.command()
def query(
    question: str = typer.Argument(..., help="Search query or question for the RAG pipeline."),
    collection: str = typer.Option("default", help="ChromaDB collection name."),
    k: int = typer.Option(4, help="Number of document chunks to retrieve."),
    rag: bool = typer.Option(True, help="Whether to run the self-correcting RAG pipeline (True) or just retrieve (False).")
):
    """Query the indexed documents."""
    db_path = os.environ.get("CHROMA_PERSIST_DIR", "./data/chromadb")
    store = ASCVectorStore(collection_name=collection, persist_directory=db_path)
    
    # Check if collection is empty
    stats = store.get_collection_stats()
    if stats.get("total_chunks", 0) == 0:
        console.print(f"[bold red]Error:[/bold red] The collection '{collection}' is empty. Run 'index' first.")
        raise typer.Exit(1)
        
    retriever = AdaptiveSemanticRetriever(
        vector_store=store,
        k=k,
        coherence_rerank=True,
        boundary_expand=True
    )
    
    if rag:
        console.print(f"[bold green]Running Self-Correcting RAG Pipeline...[/bold green]\n")
        
        # Load LLM
        # pyrefly: ignore [missing-import]
        from asc.utils.model_factory import get_llm_from_env
        llm = get_llm_from_env()
        
        async def _action():
            return await run_rag(question, retriever, llm)
            
        res = _run_async(_action())
        
        console.print("[bold yellow]Question:[/bold yellow]")
        console.print(question)
        console.print("\n[bold green]Answer:[/bold green]")
        console.print(res["answer"])
        console.print(f"\nEntailment Score: [bold cyan]{res['entailment_score']:.2f}[/bold cyan] | Loops: [bold cyan]{res['iterations']}[/bold cyan]")
        console.print(f"Sources: [magenta]{', '.join(res['sources'])}[/magenta]")
    else:
        console.print(f"[bold green]Performing semantic search (k={k}) on collection '{collection}'...[/bold green]\n")
        
        async def _action():
            return await retriever._aget_relevant_documents(question)
            
        docs = _run_async(_action())
        
        table = Table(title=f"Top {len(docs)} retrieved chunks")
        table.add_column("Index", style="cyan", justify="center")
        table.add_column("Source File", style="magenta")
        table.add_column("Snippet Preview", style="green")
        table.add_column("Perplexity", style="yellow")
        
        for idx, doc in enumerate(docs):
            snippet = doc.page_content[:120].strip() + "..."
            table.add_row(
                str(idx + 1),
                doc.metadata.get("source", "unknown"),
                snippet,
                f"{doc.metadata.get('avg_perplexity', 0.0):.2f}"
            )
        console.print(table)


@app.command()
def benchmark(
    n_articles: int = typer.Option(10, help="Number of Wikipedia articles to load for evaluation."),
    output_dir: Path = typer.Option("./docs", help="Path to write the report and charts.")
):
    """Run full benchmark: ASC vs fixed chunking."""
    console.print(f"[bold green]Starting evaluation benchmark suite (articles={n_articles})...[/bold green]")
    os.makedirs(output_dir, exist_ok=True)
    
    benchmark_runner = ASCBenchmark()
    
    async def _action():
        results = await benchmark_runner.run(n_wikipedia_articles=n_articles)
        
        # Save JSON results
        results_json = os.path.join(output_dir, "benchmark_results.json")
        from dataclasses import asdict
        serializable = [asdict(r) for r in results]
        with open(results_json, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=4)
            
        # Generate report
        report = benchmark_runner.generate_report(results)
        report_md = os.path.join(output_dir, "benchmark_report.md")
        with open(report_md, "w", encoding="utf-8") as f:
            f.write(report)
            
        # Plot and save comparison charts
        charts_dir = os.path.join(output_dir, "benchmark_charts")
        benchmark_runner.plot_comparison(results, save_dir=charts_dir)
        
        return report

    report_markdown = _run_async(_action())
    
    console.print("\n" + "="*80)
    console.print(Markdown(report_markdown))
    console.print("="*80)
    console.print(f"\n[bold green]Success![/bold green] Benchmark files saved to '{output_dir}'.")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="FastAPI host address."),
    port: int = typer.Option(8000, help="FastAPI port number."),
    reload: bool = typer.Option(False, help="Enable hot reload (for development).")
):
    """Start the FastAPI server."""
    console.print(f"[bold green]Starting FastAPI REST API server at http://{host}:{port}...[/bold green]")
    uvicorn.run("src.asc.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
