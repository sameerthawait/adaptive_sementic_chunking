"""
Cross-Encoder Reranker for ASC retrieval pipeline.

Unlike bi-encoder (vector) search which encodes query and document
separately, a cross-encoder reads them TOGETHER — giving much more
accurate relevance scores at the cost of speed.

Position in pipeline:
  Hybrid Search (fast, 20 candidates)
       ↓
  Cross-Encoder Reranker  ← HERE (accurate, scores all 20 pairs)
       ↓
  MMR (diversity filter, returns final k)
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


AVAILABLE_MODELS = {
    "minilm": {
        "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "size": "80MB",
        "speed": "fast",
        "description": "Best for quick inference, good quality"
    },
    "bge-base": {
        "name": "BAAI/bge-reranker-base",
        "size": "280MB",
        "speed": "medium",
        "description": "Strong multilingual reranker"
    },
    "bge-v2": {
        "name": "BAAI/bge-reranker-v2-m3",
        "size": "570MB",
        "speed": "slower",
        "description": "Best quality, recommended for production"
    },
}


@dataclass
class RerankerConfig:
    model_key: str = "minilm"          # key from AVAILABLE_MODELS
    top_k_before: int = 20            # candidates to score (from hybrid search)
    top_k_after: int = 8              # results to pass to MMR after reranking
    max_length: int = 512             # max token length for cross-encoder
    batch_size: int = 16              # pairs to score per batch
    show_scores: bool = True          # include reranker_score in metadata


class CrossEncoderReranker:
    """
    Reranks retrieved documents using a cross-encoder model.

    Cross-encoders are more accurate than bi-encoders because they attend
    to both the query and document simultaneously rather than encoding them
    separately. This catches semantic relevance that vector similarity misses.

    Example:
      Query: "How does perplexity detect boundaries?"
      Doc A: "Perplexity measures token surprise." — high vector sim, truly relevant
      Doc B: "Perplexity is used in language modelling." — high vector sim, less relevant
      Cross-encoder correctly ranks Doc A higher than Doc B.
    """

    def __init__(self, config: RerankerConfig | None = None):
        self.config = config or RerankerConfig()
        self._model: CrossEncoder | None = None
        self._model_name: str = ""

    def _load_model(self, model_key: str) -> CrossEncoder:
        """Lazy load — only downloads model on first use."""
        model_info = AVAILABLE_MODELS.get(model_key)
        if not model_info:
            raise ValueError(
                f"Unknown model key '{model_key}'. "
                f"Available: {list(AVAILABLE_MODELS.keys())}"
            )
        model_name = model_info["name"]
        if self._model is None or self._model_name != model_name:
            logger.info(f"Loading cross-encoder: {model_name}")
            self._model = CrossEncoder(
                model_name,
                max_length=self.config.max_length
            )
            self._model_name = model_name
            logger.info("Cross-encoder loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[tuple[Document, float]],
        model_key: str | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Reranks candidate documents using cross-encoder scores.

        Args:
            query: The user query string.
            candidates: List of (Document, score) from hybrid search.
            model_key: Override the config model. Uses config default if None.

        Returns:
            Reranked list of (Document, cross_encoder_score) tuples,
            limited to config.top_k_after results, sorted descending.
        """
        if not candidates:
            return []

        key = model_key or self.config.model_key
        model = self._load_model(key)

        # Build (query, document_text) pairs for cross-encoder
        pairs = [(query, doc.page_content) for doc, _ in candidates]

        logger.info(f"Reranking {len(pairs)} candidates with {key}...")
        scores = model.predict(
            pairs,
            batch_size=self.config.batch_size,
            show_progress_bar=False
        )

        # Combine documents with cross-encoder scores
        reranked = []
        for (doc, _), score in zip(candidates, scores):
            if self.config.show_scores:
                doc.metadata["reranker_score"] = float(round(score, 4))
                doc.metadata["reranker_model"] = key
            reranked.append((doc, float(score)))

        # Sort by cross-encoder score descending
        reranked.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"Reranking complete. "
            f"Top score: {reranked[0][1]:.3f}, "
            f"Bottom score: {reranked[-1][1]:.3f}"
        )

        return reranked[: self.config.top_k_after]

    @staticmethod
    def list_models() -> list[dict]:
        """Returns available model info for API and frontend."""
        return [
            {"key": k, **v}
            for k, v in AVAILABLE_MODELS.items()
        ]
