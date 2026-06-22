"""Unit tests for the BoundaryDetector and BoundaryDetectorConfig."""

import pytest
import numpy as np

from asc.chunker.boundary_detector import BoundaryDetector, BoundaryDetectorConfig


# 1. detect_boundaries: synthetic array with known spike → boundary at correct index
def test_detect_boundaries_spike(sample_perplexity_scores) -> None:
    config = BoundaryDetectorConfig(
        z_score_threshold=1.5,
        lag_window=5,
        min_chunk_sentences=2,
        smoothing_window=5,
        smoothing_poly_order=2,
        require_local_maximum=True
    )
    detector = BoundaryDetector(config)
    
    boundaries, diagnostics = detector.detect_boundaries(sample_perplexity_scores)
    
    # 0 and terminal index 15 should always be in boundaries
    assert boundaries[0] == 0
    assert boundaries[-1] == 15
    # The spike was at index 8. The detector should capture it and place a boundary at 8 or adjacent index.
    assert 8 in boundaries


# 2. _smooth_signal: output same length as input; non-negative
def test_smooth_signal_properties() -> None:
    detector = BoundaryDetector(BoundaryDetectorConfig(smoothing_window=5))
    
    # Negative values should be clipped
    signal = np.array([2.0, -4.0, 1.0, 10.0, -2.0, 5.0])
    smoothed = detector._smooth_signal(signal)
    
    assert len(smoothed) == len(signal)
    assert np.all(smoothed >= 0)


# 3. _rolling_zscore: first `lag` values are 0; spike produces z > 2
def test_rolling_zscore_spike() -> None:
    lag = 3
    detector = BoundaryDetector(BoundaryDetectorConfig(lag_window=lag))
    # Spike at index 5
    scores = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 50.0, 2.0], dtype=np.float32)
    
    z = detector._rolling_zscore(scores, lag=lag)
    
    # First `lag` values are 0
    for i in range(lag):
        assert z[i] == 0.0
        
    # The spike at index 5 should produce a massive rolling z-score spike > 2
    # Mean of window [2.0, 2.0, 2.0] is 2.0, std fallback is 1.0 -> (50.0 - 2.0)/1.0 = 48.0
    assert z[5] > 2.0


# 4. _enforce_min_chunk_size: boundaries too close get merged correctly
def test_enforce_min_chunk_size_merging() -> None:
    detector = BoundaryDetector()
    
    # min_chunk_sentences is 3. Total sentences is 10.
    # Candidates: [2, 4, 7]
    # 2 is < 3 from 0 -> skipped.
    # 4 is >= 3 from 0 -> kept [0, 4]
    # 7 is >= 3 from 4 -> kept [0, 4, 7]
    # tail check: 10 - 7 = 3 >= 3 -> kept [0, 4, 7]
    # result boundaries: [0, 4, 7, 10]
    boundaries = detector._enforce_min_chunk_size([2, 4, 7], min_size=3, total=10)
    assert boundaries == [0, 4, 7, 10]
    
    # Candidates: [3, 4, 7]
    # 3 is >= 3 from 0 -> kept [0, 3]
    # 4 is < 3 from 3 -> skipped.
    # 7 is >= 3 from 3 -> kept [0, 3, 7]
    # result boundaries: [0, 3, 7, 10]
    boundaries2 = detector._enforce_min_chunk_size([3, 4, 7], min_size=3, total=10)
    assert boundaries2 == [0, 3, 7, 10]


# 5. edge case: array shorter than smoothing_window
def test_edge_case_short_array() -> None:
    detector = BoundaryDetector(BoundaryDetectorConfig(smoothing_window=5))
    short_signal = np.array([2.0, 4.0, 3.0])
    
    smoothed = detector._smooth_signal(short_signal)
    
    # Should return signal unchanged (but clipped)
    assert np.array_equal(smoothed, np.clip(short_signal, 0, None))


# 6. edge case: all-zero perplexity array → no boundaries
def test_edge_case_all_zeros() -> None:
    detector = BoundaryDetector(BoundaryDetectorConfig(lag_window=3, z_score_threshold=2.0))
    zeros = np.zeros(10, dtype=np.float32)
    
    boundaries, diagnostics = detector.detect_boundaries(zeros)
    
    # Z-scores should all be 0. No candidate indices.
    # Final boundaries should only be [0, 10]
    assert boundaries == [0, 10]
