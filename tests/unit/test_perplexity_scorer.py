"""Unit tests for the PerplexityScorer."""

import pytest
import math
import asyncio
import httpx
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from asc.chunker.perplexity_scorer import PerplexityScorer


# 1. _build_context: correct window slicing at start/end/middle
def test_build_context() -> None:
    scorer = PerplexityScorer(model="test", context_window=3)
    sentences = ["s0", "s1", "s2", "s3", "s4", "s5"]
    
    # Start: index 0 (no context)
    assert scorer._build_context(sentences, 0) == ""
    # Middle: index 2 (window size 3, context is s0, s1)
    assert scorer._build_context(sentences, 2) == "s0 s1"
    # End: index 5 (context is s2 s3 s4)
    assert scorer._build_context(sentences, 5) == "s2 s3 s4"


# 2. _replace_nan_scores: handles partial and all-NaN arrays
def test_replace_nan_scores() -> None:
    scorer = PerplexityScorer(model="test")
    
    # Partial NaN array -> replaced with median (12.0)
    arr_partial = np.array([10.0, np.nan, 12.0, 15.0, np.nan], dtype=np.float32)
    replaced_partial = scorer._replace_nan_scores(arr_partial)
    assert replaced_partial[1] == 12.0
    assert replaced_partial[4] == 12.0
    
    # All NaN array -> replaced with fallback baseline (15.0)
    arr_all = np.array([np.nan, np.nan], dtype=np.float32)
    replaced_all = scorer._replace_nan_scores(arr_all)
    assert replaced_all[0] == 15.0
    assert replaced_all[1] == 15.0


# 3. score_sentences: with mocked _call_ollama, verify concurrency limit respected
@pytest.mark.asyncio
async def test_score_sentences_concurrency() -> None:
    # Set context_window=2, max_concurrent=2
    scorer = PerplexityScorer(model="test", context_window=2, max_concurrent=2)
    sentences = ["s0", "s1", "s2", "s3", "s4"]
    
    mock_response = {
        "response": "ok",
        "logprobs": [{"token": "ok", "logprob": -1.0}]
    }
    
    active_calls = 0
    max_active_calls = 0
    
    async def mock_call(prompt):
        nonlocal active_calls, max_active_calls
        active_calls += 1
        max_active_calls = max(max_active_calls, active_calls)
        await asyncio.sleep(0.01)  # tiny delay to allow parallel execution
        active_calls -= 1
        return mock_response

    with patch.object(scorer, "_call_ollama", side_effect=mock_call):
        scores = await scorer.score_sentences(sentences)
        assert len(scores) == 5
        # Concurrency limit should be respected
        assert max_active_calls <= 2


# 4. _extract_sentence_logprobs / slicing: correct slicing of logprobs by context token count
@pytest.mark.asyncio
async def test_score_single_slicing(mock_ollama_response) -> None:
    scorer = PerplexityScorer(model="test")
    
    context = ["Hello", "world", "this", "is", "test"]  # 5 words / tokens
    target = "sentence to score"  # 3 words / tokens
    prompt = "Hello world this is test sentence to score"
    
    # logprobs response values for all 8 words
    logprob_vals = [-1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -3.0, -4.0]
    mock_resp = mock_ollama_response(prompt, logprob_vals)
    
    with patch.object(scorer, "_call_ollama", AsyncMock(return_value=mock_resp)):
        context_text = " ".join(context).strip()
        context_tokens = len(scorer.tokenizer.encode(context_text))
        
        perplexity = await scorer._score_single(target, context)
        
        # Expected logprobs are sliced from index context_tokens onward
        # Average of target logprobs: mean([-2.0, -3.0, -4.0]) = -3.0
        # Perplexity = exp(3.0)
        expected_ppl = math.exp(3.0)
        assert abs(perplexity - expected_ppl) < 1e-4


# 5. Retry logic: 3 retries on TimeoutException/ConnectError
@pytest.mark.asyncio
async def test_retry_logic() -> None:
    scorer = PerplexityScorer(model="test", timeout=1.0)
    
    call_count = 0
    async def mock_post(url, json):
        nonlocal call_count
        call_count += 1
        raise httpx.ConnectTimeout("Connect timed out")
        
    with patch("httpx.AsyncClient.post", side_effect=mock_post):
        with pytest.raises(httpx.HTTPError):
            # Should fail after all 3 attempts
            await scorer._call_ollama("test prompt")
            
    assert call_count == 3
