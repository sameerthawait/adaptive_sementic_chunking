"""Metadata filter builder for ASC chunk retrieval.

Translates user-friendly filter specs into ChromaDB where clauses.
"""

from dataclasses import dataclass


@dataclass
class ChunkFilter:
    """Represents a metadata filter for chunk retrieval.

    Available filter fields (all optional):
      source: str           — filter by source document name
      sources: list[str]    — filter by multiple source documents (OR)
      min_perplexity: float — only chunks with avg_perplexity >= this
      max_perplexity: float — only chunks with avg_perplexity <= this
      min_sentences: int    — only chunks with sentence_count >= this
      max_sentences: int    — only chunks with sentence_count <= this
      chunk_type: str       — "semantic" or "fixed"
    """

    source: str | None = None
    sources: list[str] | None = None
    min_perplexity: float | None = None
    max_perplexity: float | None = None
    min_sentences: int | None = None
    max_sentences: int | None = None
    chunk_type: str | None = None

    def to_chroma_where(self) -> dict | None:
        """Converts to ChromaDB where clause format.

        Returns None if no filters set.

        Examples:
          {source: "doc.txt"} -> {"source": {"$eq": "doc.txt"}}
          {sources: ["a.txt", "b.txt"]} -> {"source": {"$in": ["a.txt", "b.txt"]}}
          {min_perplexity: 40, max_perplexity: 80} ->
            {"$and": [{"avg_perplexity": {"$gte": 40}}, {"avg_perplexity": {"$lte": 80}}]}
        """
        conditions = []

        if self.source:
            conditions.append({"source": {"$eq": self.source}})

        if self.sources:
            if len(self.sources) == 1:
                conditions.append({"source": {"$eq": self.sources[0]}})
            elif len(self.sources) > 1:
                conditions.append({"source": {"$in": self.sources}})

        if self.min_perplexity is not None:
            conditions.append({"avg_perplexity": {"$gte": float(self.min_perplexity)}})

        if self.max_perplexity is not None:
            conditions.append({"avg_perplexity": {"$lte": float(self.max_perplexity)}})

        if self.min_sentences is not None:
            conditions.append({"sentence_count": {"$gte": int(self.min_sentences)}})

        if self.max_sentences is not None:
            conditions.append({"sentence_count": {"$lte": int(self.max_sentences)}})

        if self.chunk_type:
            conditions.append({"chunk_type": {"$eq": self.chunk_type}})

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def is_empty(self) -> bool:
        """Returns True if no filters are set."""
        return not any([
            self.source,
            self.sources,
            self.min_perplexity is not None,
            self.max_perplexity is not None,
            self.min_sentences is not None,
            self.max_sentences is not None,
            self.chunk_type,
        ])
