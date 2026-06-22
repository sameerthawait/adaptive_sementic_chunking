"""Perplexity scorer module for Adaptive Semantic Chunking.

Uses a local Ollama LLM to calculate the conditional perplexity of sentences
using token log probabilities and a context window.
"""

import asyncio
import logging
import math
import time
from typing import Any
import httpx
import numpy as np
import tiktoken

logger = logging.getLogger(__name__)


class PerplexityScorer:
    """Computes per-sentence perplexity using a local Ollama causal language model.

    Algorithm: Uses token log-probabilities from a causal LM to measure
    how surprising each sentence is given its preceding context window.
    High perplexity signals semantic discontinuity — a potential chunk boundary.
    This approach differs from embedding cosine distance methods by measuring
    causal surprise rather than semantic similarity.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        context_window: int = 3,
        max_concurrent: int = 3,
        timeout: float = 120.0,
        embeddings: Any | None = None,
    ) -> None:
        """Initializes the perplexity scorer.

        Args:
            model: The name of the local Ollama model to use (e.g. 'llama3.2:3b').
            base_url: The base URL of the Ollama server.
            context_window: Sliding context window size (number of prior sentences).
            max_concurrent: Semaphore size for concurrent Ollama API calls.
            timeout: Timeout in seconds for HTTP requests.
            embeddings: Optional external embedding client to use for semantic similarity distance fallback.
        """
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.context_window = context_window
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.embeddings = embeddings

    def _build_context(self, sentences: list[str], idx: int) -> str:
        """Returns joined context sentences for position idx.

        Args:
            sentences: List of sentences.
            idx: The current sentence index.

        Returns:
            The merged context text.
        """
        start_idx = max(0, idx - self.context_window)
        context_sents = sentences[start_idx:idx]
        return " ".join(context_sents).strip()

    async def _call_ollama(self, prompt: str) -> dict:
        """Async httpx call with retry (3 attempts, exponential backoff 1s/2s/4s).

        Args:
            prompt: The full prompt text to send to Ollama.

        Returns:
            The parsed JSON response dict.

        Raises:
            httpx.HTTPError: If the API call fails after all retries.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "logprobs": True,
                "temperature": 0,
            },
        }

        # Keep a separate client config for execution timeouts
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        
        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(limits=limits, timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                    return response.json()
            except (httpx.HTTPError, Exception) as e:
                backoff = attempt * 2 if attempt > 1 else 1.0
                logger.warning(
                    f"Ollama API attempt {attempt} failed for model '{self.model}': {e}. "
                    f"Retrying in {backoff}s..."
                )
                if attempt == 3:
                    raise e
                await asyncio.sleep(backoff)

        return {}

    async def _score_single(
        self,
        sentence: str,
        context_sentences: list[str],
    ) -> float:
        """Calls Ollama API, extracts log-probs for sentence tokens only.

        Token counting: use tiktoken cl100k_base to count context tokens,
        then slice response logprobs from context_token_count onward.
        Return exp(-mean(sentence_log_probs)).
        Handle empty sentence or API error: return float('nan').

        Args:
            sentence: Target sentence string.
            context_sentences: List of prior context sentences.

        Returns:
            The conditional perplexity of the sentence.
        """
        clean_sentence = sentence.strip()
        if not clean_sentence:
            return float("nan")

        context_text = " ".join(context_sentences).strip()
        
        # Calculate context token count using tiktoken
        context_token_count = len(self.tokenizer.encode(context_text)) if context_text else 0

        # Construct full prompt
        full_prompt = f"{context_text} {clean_sentence}".strip()

        try:
            data = await self._call_ollama(full_prompt)
            # Ollama logprobs is a list of dicts: [{'token': str, 'logprob': float}]
            logprobs_data = data.get("logprobs", [])
            
            if not logprobs_data:
                # Try to fall back to 'response_metadata' or check structure
                logger.debug("No logprobs returned from Ollama API.")
                return float("nan")

            # Extract numeric logprobs
            logprobs = []
            for item in logprobs_data:
                val = item.get("logprob")
                if val is not None:
                    logprobs.append(float(val))

            # Slice from context_token_count onward to isolate target sentence tokens
            sentence_log_probs = logprobs[context_token_count:]
            
            if not sentence_log_probs:
                logger.debug(
                    f"Empty target sentence logprobs after slicing. "
                    f"(Total logprobs: {len(logprobs)}, Context tokens: {context_token_count})"
                )
                return float("nan")

            mean_logprob = sum(sentence_log_probs) / len(sentence_log_probs)
            perplexity = math.exp(-mean_logprob)
            return perplexity

        except Exception as e:
            logger.error(f"Error scoring single sentence: {e}")
            return float("nan")

    @staticmethod
    def _replace_nan_scores(scores: np.ndarray) -> np.ndarray:
        """Replaces NaN values with column median. Handles all-NaN edge case.

        Args:
            scores: NumPy array containing perplexity scores.

        Returns:
            NumPy array with NaN values replaced.
        """
        scores_copy = scores.copy()
        nan_mask = np.isnan(scores_copy)
        
        if not np.any(nan_mask):
            return scores_copy

        valid_scores = scores_copy[~nan_mask]
        if len(valid_scores) == 0:
            # All values are NaN, fall back to default baseline perplexity (e.g. 15.0)
            median_val = 15.0
        else:
            median_val = float(np.median(valid_scores))

        scores_copy[nan_mask] = median_val
        return scores_copy

    async def score_sentences(
        self,
        sentences: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Scores all sentences concurrently (bounded by semaphore) or via embeddings distance fallback.

        Returns np.ndarray of shape (len(sentences),) with float32 perplexity/distance values.
        First `context_window` sentences get score = mean of remaining scores
        (no prior context available).

        Args:
            sentences: List of sentences to score.
            show_progress: Whether to show a progress bar.

        Returns:
            NumPy array of perplexity/distance scores.
        """
        n = len(sentences)
        scores = np.full(n, np.nan, dtype=np.float32)
        
        if n == 0:
            return scores
            
        if n <= self.context_window:
            # Not enough sentences to establish context windows; return baseline
            scores[:] = 15.0
            return scores

        # Check if we should use embedding-based distance fallback
        import os
        provider = os.environ.get("LLM_PROVIDER", "ollama").lower()
        if self.embeddings is not None or provider in ["nvidia", "openai"]:
            if self.embeddings is None:
                from asc.embedding.embedder import ASCEmbedder
                self.embeddings = ASCEmbedder(provider=provider)
                
            logger.info("Using embedding-based semantic distance for chunk boundary detection.")
            
            # Embed all sentences in one batch asynchronously
            sentence_embeddings = await self.embeddings.aembed_documents(sentences)
            sentence_embeddings_np = [np.array(emb) for emb in sentence_embeddings]
            
            # Compute distance for each sentence relative to its context window average
            for idx in range(self.context_window, n):
                start_idx = max(0, idx - self.context_window)
                context_embs = sentence_embeddings_np[start_idx:idx]
                context_mean = np.mean(context_embs, axis=0)
                current_emb = sentence_embeddings_np[idx]
                
                norm_c = np.linalg.norm(context_mean)
                norm_s = np.linalg.norm(current_emb)
                if norm_c == 0.0 or norm_s == 0.0:
                    similarity = 0.0
                else:
                    similarity = np.dot(context_mean, current_emb) / (norm_c * norm_s)
                
                distance = 1.0 - float(similarity)
                # Scale semantic distance to fit baseline perplexity score distribution for boundary detector compatibility
                pseudo_perplexity = distance * 10.0 + 10.0
                scores[idx] = pseudo_perplexity

            # First context_window sentences get score = mean of remaining scores
            remaining_scores = scores[self.context_window:]
            valid_remaining = remaining_scores[~np.isnan(remaining_scores)]
            mean_remaining = float(np.mean(valid_remaining)) if len(valid_remaining) > 0 else 15.0
            scores[0:self.context_window] = mean_remaining

            final_scores = self._replace_nan_scores(scores)
            return final_scores

        # Default Ollama-based perplexity scoring
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def worker(idx: int) -> None:
            # Context sentences
            start_idx = max(0, idx - self.context_window)
            context_sents = sentences[start_idx:idx]
            
            async with semaphore:
                score = await self._score_single(sentences[idx], context_sents)
                scores[idx] = score

        # Prepare concurrent scoring tasks for sentences past the context window
        tasks = [worker(i) for i in range(self.context_window, n)]

        if show_progress:
            from tqdm.asyncio import tqdm
            await tqdm.gather(*tasks, desc="Evaluating Perplexities")
        else:
            await asyncio.gather(*tasks)

        # First context_window sentences get score = mean of remaining scores
        remaining_scores = scores[self.context_window:]
        valid_remaining = remaining_scores[~np.isnan(remaining_scores)]
        
        if len(valid_remaining) > 0:
            mean_remaining = float(np.mean(valid_remaining))
        else:
            mean_remaining = 15.0

        scores[0:self.context_window] = mean_remaining

        # Replace any remaining NaNs in the rest of the array with the median
        final_scores = self._replace_nan_scores(scores)
        return final_scores


if __name__ == "__main__":
    # Setup simple logging for standalone testing
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    # Standard sample Wikipedia paragraph with thematic shift
    wikipedia_sentences = [
        "In computer science, artificial intelligence (AI) is intelligence demonstrated by machines.",
        "Early AI research focused on symbolic logic and problem-solving methodologies.",
        "During the late 20th century, expert systems became highly popular for business applications.",
        "However, these systems were rigid and failed to adapt to new inputs dynamically.",
        "This caused a lack of progress and led to the period known as the AI winter.",
        "In contrast, genomics is the interdisciplinary study of the complete genetic mapping of organisms.",
        "Deoxyribonucleic acid contains the complex instructions necessary for cellular functions.",
        "Modern genome sequencing allows researchers to decode biological blueprints in hours.",
        "Biologists use these sequences to identify disease-causing mutations and variations.",
        "Computational models are now widely used to simulate complex genetic interactions."
    ]

    async def main() -> None:
        print("Starting PerplexityScorer standalone test...")
        scorer = PerplexityScorer(model="llama3.2:3b", base_url="http://localhost:11434")
        
        start_time = time.time()
        # Scoring with progress bar
        scores = await scorer.score_sentences(wikipedia_sentences, show_progress=True)
        duration = time.time() - start_time
        
        print("\n" + "="*80)
        print(f"Standalone Test completed in {duration:.2f} seconds.")
        print("="*80)
        
        for idx, (sent, score) in enumerate(zip(wikipedia_sentences, scores)):
            preview = sent[:60] + "..." if len(sent) > 60 else sent
            print(f"Sentence {idx:02d} | Score: {score:6.2f} | Preview: {preview}")
        print("="*80)

    # Run loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Test cancelled.")
