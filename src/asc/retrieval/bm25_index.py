"""BM25 keyword index for hybrid search.

Runs alongside ChromaDB vector search and scores are combined
using Reciprocal Rank Fusion (RRF).
"""

import logging
import pickle
from typing import Sequence
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """Maintains a BM25 keyword index over all indexed chunks.

    Rebuilt from ChromaDB on startup. Updated incrementally on new documents.

    Algorithm:
      BM25 scores each chunk by term frequency + inverse document frequency.
      Combined with vector scores via RRF:
        rrf_score = 1/(k + rank_bm25) + 1/(k + rank_vector)  where k=60
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initializes the BM25 index parameters.

        Args:
            k1: term frequency saturation (1.2–2.0 typical)
            b: length normalization (0=no normalization, 1=full)
        """
        self.k1 = k1
        self.b = b
        self.documents: list[Document] = []
        self.bm25: BM25Okapi | None = None

    def build(self, documents: Sequence[Document]) -> None:
        """Build index from a list of LangChain Documents.

        Tokenizes on whitespace + lowercase.

        Args:
            documents: List of documents to build the index from.
        """
        self.documents = list(documents)
        if not self.documents:
            self.bm25 = None
            return

        # Simple tokenization by converting to lowercase and splitting on whitespace
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def add_documents(self, documents: Sequence[Document]) -> None:
        """Incrementally add new documents to existing index.

        Args:
            documents: New documents to index.
        """
        self.documents.extend(list(documents))
        self.build(self.documents)

    def search(self, query: str, k: int = 20) -> list[tuple[Document, float]]:
        """Returns (document, normalized_bm25_score) tuples sorted descending.

        Scores normalized to 0-1 range by dividing by max score in results.

        Args:
            query: The text query.
            k: Maximum number of results to return.

        Returns:
            List of (Document, score) tuples.
        """
        if not self.bm25 or not self.documents:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Zip documents with scores
        doc_scores = list(zip(self.documents, scores))

        # Sort descending by score
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        top_k = doc_scores[:k]
        if not top_k:
            return []

        # Find maximum score in the results
        max_score = max([score for _, score in top_k])

        if max_score == 0.0:
            return [(doc, 0.0) for doc, _ in top_k]

        return [(doc, float(score / max_score)) for doc, score in top_k]

    def save(self, path: str) -> None:
        """Persist index to disk using pickle.

        Args:
            path: Target file path to write to.
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "k1": self.k1,
                    "b": self.b,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load index from disk.

        Args:
            path: Source file path to load from.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.documents = data.get("documents", [])
            self.k1 = data.get("k1", 1.5)
            self.b = data.get("b", 0.75)
            self.build(self.documents)
