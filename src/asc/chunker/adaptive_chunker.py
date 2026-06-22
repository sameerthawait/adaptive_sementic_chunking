"""Adaptive Semantic Chunker module.

Orchestrates the perplexity scoring and boundary detection to split raw text
into semantic chunks prepended with overlap context.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Sequence
import nltk
import numpy as np
from langchain_core.documents import BaseDocumentTransformer, Document

from asc.chunker.perplexity_scorer import PerplexityScorer
from asc.chunker.boundary_detector import BoundaryDetector, BoundaryDetectorConfig

logger = logging.getLogger(__name__)


def _ensure_nltk_resources() -> None:
    """Ensures NLTK sentence tokenization resources are downloaded."""
    for package in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{package}")
        except LookupError:
            try:
                nltk.download(package, quiet=True)
                logger.debug(f"Downloaded NLTK resource: {package}")
            except Exception as e:
                logger.warning(f"Could not download NLTK resource '{package}': {e}")


@dataclass
class ChunkMetadata:
    chunk_index: int
    total_chunks: int
    sentence_start: int
    sentence_end: int
    sentence_count: int
    avg_perplexity: float
    max_perplexity: float
    min_perplexity: float
    boundary_z_score: float         # z-score that triggered boundary
    chunk_type: str = "semantic"    # "semantic" vs "fixed" for comparison
    source: str = ""


class AdaptiveSemanticChunker(BaseDocumentTransformer):
    """
    Orchestrates: sentences → perplexity scores → boundaries → LangChain Documents.
    
    Produces Documents with ChunkMetadata embedded in .metadata dict,
    enabling downstream retrieval enhancements and algorithm performance demonstration.
    """

    def __init__(
        self,
        perplexity_scorer: PerplexityScorer | None = None,
        boundary_detector: BoundaryDetector | None = None,
        sentence_overlap: int = 1,    # shared sentences between adjacent chunks
        # Backward compatibility parameters:
        model: str = "llama3.2:3b",
        ollama_url: str = "http://localhost:11434",
        window_size: int = 3,
        savgol_window: int = 5,
        savgol_polyorder: int = 2,
        z_threshold: float = 1.2,
        rolling_lag: int = 5,
        min_sentences_per_chunk: int = 2,
        concurrency_limit: int = 1,
    ) -> None:
        """Initializes the adaptive semantic chunker."""
        _ensure_nltk_resources()
        
        if perplexity_scorer is None:
            perplexity_scorer = PerplexityScorer(
                model=model,
                base_url=ollama_url,
                context_window=window_size,
                max_concurrent=concurrency_limit,
            )
            
        if boundary_detector is None:
            config = BoundaryDetectorConfig(
                z_score_threshold=z_threshold,
                lag_window=rolling_lag,
                min_chunk_sentences=min_sentences_per_chunk,
                smoothing_window=savgol_window,
                smoothing_poly_order=savgol_polyorder,
                require_local_maximum=True,
            )
            boundary_detector = BoundaryDetector(config=config)

        self.scorer = perplexity_scorer
        self.detector = boundary_detector
        self.sentence_overlap = sentence_overlap

    async def _score_sentences(self, sentences: list[str]) -> list[float]:
        """Scores each sentence's perplexity in a sliding context window.

        Args:
            sentences: The list of sentences in the document.

        Returns:
            A list of perplexity scores, one for each sentence.
        """
        if not sentences:
            return []
        scores = await self.scorer.score_sentences(sentences)
        return scores.tolist()

    async def chunk_text(
        self,
        text: str,
        source_metadata: dict | None = None
    ) -> list[Document]:
        """
        Full pipeline:
        1. NLTK sentence tokenize
        2. Score sentences (async perplexity)
        3. Detect boundaries
        4. Build Documents with overlap and ChunkMetadata
        5. Return list[Document]
        
        Handles edge cases:
        - Text with < min_chunk_sentences sentences: return as single chunk
        - Empty text: return []
        - Sentences with special chars: preserve faithfully
        """
        if not text.strip():
            return []

        sentences = nltk.sent_tokenize(text)
        min_sentences = self.detector.config.min_chunk_sentences

        if len(sentences) <= min_sentences:
            # return as single chunk
            chunk_dict = {
                "text": text,
                "sentences": sentences,
                "sentence_start": 0,
                "sentence_end": len(sentences),
                "sentence_count": len(sentences),
                "avg_perplexity": 15.0,
                "max_perplexity": 15.0,
                "min_perplexity": 15.0,
                "boundary_z_score": 0.0,
            }
            return self._to_documents([chunk_dict], source_metadata or {})

        # 2. Score sentences
        perplexity_scores = await self._score_sentences(sentences)
        perplexity_scores_np = np.array(perplexity_scores)

        # 3. Detect boundaries
        boundaries, diagnostics = self.detector.detect_boundaries(perplexity_scores_np)

        # 4. Build chunks with overlap
        chunk_dicts = self._build_chunks_with_overlap(
            sentences=sentences,
            boundaries=boundaries,
            perplexity_scores=perplexity_scores_np,
            diagnostics=diagnostics,
            overlap=self.sentence_overlap
        )

        # 5. Convert to Documents
        return self._to_documents(chunk_dicts, source_metadata or {})

    async def chunk_documents(
        self,
        documents: list[Document],
        show_progress: bool = True
    ) -> list[Document]:
        """
        Processes list of LangChain Documents.
        Preserves and merges source .metadata into each output chunk.
        Uses tqdm progress bar if show_progress=True.
        """
        all_chunks = []
        if not documents:
            return []

        if show_progress:
            from tqdm.asyncio import tqdm
            for idx, doc in enumerate(tqdm(documents, desc="Chunking Documents")):
                meta = {**doc.metadata, "source_doc_index": idx}
                chunks = await self.chunk_text(doc.page_content, meta)
                all_chunks.extend(chunks)
        else:
            for idx, doc in enumerate(documents):
                meta = {**doc.metadata, "source_doc_index": idx}
                chunks = await self.chunk_text(doc.page_content, meta)
                all_chunks.extend(chunks)

        return all_chunks

    def _build_chunks_with_overlap(
        self,
        sentences: list[str],
        boundaries: list[int],
        perplexity_scores: np.ndarray,
        diagnostics: dict,
        overlap: int
    ) -> list[dict]:
        """
        Given boundary positions, construct chunk dicts.
        Each chunk includes `overlap` sentences from the previous chunk
        prepended (for retrieval context continuity).
        
        Returns list of dicts: {text, sentences, sentence_start, sentence_end,
        avg_perplexity, max_perplexity, min_perplexity, boundary_z_score}
        """
        chunk_dicts = []
        num_chunks = len(boundaries) - 1

        for idx in range(num_chunks):
            start_idx = boundaries[idx]
            end_idx = boundaries[idx + 1]

            # Determine overlapping sentences to prepend from previous chunk
            overlap_start = max(0, start_idx - overlap)
            chunk_sentences = sentences[overlap_start:end_idx]
            chunk_text = " ".join(chunk_sentences)

            # Perplexities of the semantic part (excluding prepended context)
            semantic_ppls = perplexity_scores[start_idx:end_idx]

            if len(semantic_ppls) > 0:
                avg_ppl = float(np.mean(semantic_ppls))
                max_ppl = float(np.max(semantic_ppls))
                min_ppl = float(np.min(semantic_ppls))
            else:
                avg_ppl = max_ppl = min_ppl = 15.0

            # boundary_z_score is the z-score at the end boundary that split this chunk
            z_scores = diagnostics.get("z_scores", [])
            boundary_z = 0.0
            if end_idx < len(z_scores):
                boundary_z = float(z_scores[end_idx])

            chunk_dicts.append({
                "text": chunk_text,
                "sentences": chunk_sentences,
                "sentence_start": start_idx,
                "sentence_end": end_idx,
                "sentence_count": len(chunk_sentences),
                "avg_perplexity": avg_ppl,
                "max_perplexity": max_ppl,
                "min_perplexity": min_ppl,
                "boundary_z_score": boundary_z,
            })

        return chunk_dicts

    def _to_documents(
        self,
        chunk_dicts: list[dict],
        source_metadata: dict
    ) -> list[Document]:
        """Converts chunk dicts to LangChain Documents with full ChunkMetadata."""
        docs = []
        total_chunks = len(chunk_dicts)
        for idx, c in enumerate(chunk_dicts):
            meta = ChunkMetadata(
                chunk_index=idx,
                total_chunks=total_chunks,
                sentence_start=c["sentence_start"],
                sentence_end=c["sentence_end"],
                sentence_count=c["sentence_count"],
                avg_perplexity=c["avg_perplexity"],
                max_perplexity=c["max_perplexity"],
                min_perplexity=c["min_perplexity"],
                boundary_z_score=c["boundary_z_score"],
                chunk_type="semantic",
                source=source_metadata.get("source", ""),
            )
            meta_dict = {
                **source_metadata,
                "chunk_index": meta.chunk_index,
                "total_chunks": meta.total_chunks,
                "sentence_start": meta.sentence_start,
                "sentence_end": meta.sentence_end,
                "sentence_count": meta.sentence_count,
                "avg_perplexity": meta.avg_perplexity,
                "mean_perplexity": meta.avg_perplexity,
                "max_perplexity": meta.max_perplexity,
                "min_perplexity": meta.min_perplexity,
                "boundary_z_score": meta.boundary_z_score,
                "chunk_type": meta.chunk_type,
                "source": meta.source,
                "is_chunked": True,
            }
            docs.append(Document(page_content=c["text"], metadata=meta_dict))
        return docs

    @classmethod
    def from_env(cls) -> "AdaptiveSemanticChunker":
        """
        Builds instance from environment variables.
        Reads: LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_PERPLEXITY_MODEL, 
               PERPLEXITY_CONTEXT_WINDOW, Z_SCORE_THRESHOLD, MIN_CHUNK_SENTENCES
        """
        import os
        from dotenv import load_dotenv
        load_dotenv()

        provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
        context_window = int(os.environ.get("PERPLEXITY_CONTEXT_WINDOW", "3"))
        z_threshold = float(os.environ.get("Z_SCORE_THRESHOLD", "2.0"))
        min_chunk_sentences = int(os.environ.get("MIN_CHUNK_SENTENCES", "3"))

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

        config = BoundaryDetectorConfig(
            z_score_threshold=z_threshold,
            min_chunk_sentences=min_chunk_sentences,
            smoothing_window=5,
            smoothing_poly_order=2,
            lag_window=5,
            require_local_maximum=True,
        )

        detector = BoundaryDetector(config=config)
        return cls(perplexity_scorer=scorer, boundary_detector=detector)

    def compare_with_fixed(
        self,
        text: str,
        recursive_chunk_size: int | None = None,
        recursive_chunk_overlap: int | None = None,
    ) -> dict[str, list[Document]]:
        """
        Compares ASC chunks with LangChain RecursiveCharacterTextSplitter chunks.
        
        Returns:
            dict with keys:
            - 'asc': list[Document] from AdaptiveSemanticChunker
            - 'recursive': list[Document] from RecursiveCharacterTextSplitter
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        asc_chunks = self.split_text(text)
        
        if recursive_chunk_size is None:
            if asc_chunks:
                avg_size = int(np.mean([len(c) for c in asc_chunks]))
                recursive_chunk_size = max(200, avg_size)
            else:
                recursive_chunk_size = 500

        if recursive_chunk_overlap is None:
            recursive_chunk_overlap = int(recursive_chunk_size * 0.15)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=recursive_chunk_size,
            chunk_overlap=recursive_chunk_overlap,
            length_function=len,
        )

        recursive_docs = splitter.create_documents([text])
        for doc in recursive_docs:
            doc.metadata["chunk_type"] = "fixed"

        # Obtain ASC full docs with metadata synchronously using event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                asc_full_docs = loop.run_until_complete(self.chunk_text(text))
            else:
                asc_full_docs = loop.run_until_complete(self.chunk_text(text))
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                asc_full_docs = new_loop.run_until_complete(self.chunk_text(text))
            finally:
                new_loop.close()

        return {
            "asc": asc_full_docs,
            "recursive": recursive_docs,
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    # Backward compatibility wrappers:
    async def transform_documents_async(
        self,
        documents: Sequence[Document],
    ) -> list[Document]:
        """Backward compatibility for transform_documents_async."""
        return await self.chunk_documents(list(documents), show_progress=False)

    def transform_documents(
        self,
        documents: Sequence[Document],
        **kwargs: Any,
    ) -> Sequence[Document]:
        """Backward compatibility for transform_documents."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.transform_documents_async(documents))
            else:
                return loop.run_until_complete(self.transform_documents_async(documents))
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.transform_documents_async(documents))
            finally:
                new_loop.close()

    async def split_text_async(self, text: str) -> list[str]:
        """Backward compatibility for split_text_async."""
        docs = await self.chunk_text(text)
        return [doc.page_content for doc in docs]

    def split_text(self, text: str) -> list[str]:
        """Backward compatibility for split_text."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
                return loop.run_until_complete(self.split_text_async(text))
            else:
                return loop.run_until_complete(self.split_text_async(text))
        except RuntimeError:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(self.split_text_async(text))
            finally:
                new_loop.close()
