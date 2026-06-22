"""Boundary detector module for Adaptive Semantic Chunking.

Uses Savitzky-Golay filtering, rolling z-scores, and local maxima
detection to locate semantic boundaries in a perplexity score sequence.
"""

import logging
import numpy as np
from pydantic import BaseModel
from scipy.signal import savgol_filter, argrelmax
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class BoundaryDetectorConfig(BaseModel):
    z_score_threshold: float = 2.0      # std devs above mean to flag boundary
    lag_window: int = 5                  # lookback window for rolling z-score
    min_chunk_sentences: int = 3         # minimum sentences per chunk
    smoothing_window: int = 5            # Savitzky-Golay window (must be odd)
    smoothing_poly_order: int = 2        # Savitzky-Golay polynomial order
    require_local_maximum: bool = True   # boundary must also be local max


class BoundaryDetector:
    """
    Detects semantic boundaries in a perplexity score sequence.
    
    Novel boundary detection using three conditions:
    (1) Rolling z-score spike above threshold
    (2) Local maximum in smoothed signal
    (3) Minimum chunk size constraint
    
    This combination reduces false positives while reliably detecting semantic transitions.
    """
    
    def __init__(self, config: BoundaryDetectorConfig | None = None) -> None:
        self.config = config or BoundaryDetectorConfig()
        
    def detect_boundaries(
        self,
        perplexity_scores: np.ndarray
    ) -> tuple[list[int], dict]:
        """
        Main method. Returns:
        - boundaries: list of sentence indices that are chunk boundaries
        - diagnostics: dict with smoothed_scores, z_scores, candidates, final_boundaries
        """
        n = len(perplexity_scores)
        if n == 0:
            return [], {}

        # 1. Smooth the signal
        smoothed_scores = self._smooth_signal(perplexity_scores)

        # 2. Compute rolling z-scores on smoothed signal
        z_scores = self._rolling_zscore(smoothed_scores, self.config.lag_window)

        # 3. Locate candidates (Z-score spike + optional Local Maxima gate)
        candidate_mask = z_scores >= self.config.z_score_threshold

        if self.config.require_local_maximum:
            # Require the spike to be a local maximum in smoothed signal
            local_max_mask = self._local_maxima_mask(smoothed_scores)
            candidate_indices = np.where(candidate_mask & local_max_mask)[0].tolist()
        else:
            candidate_indices = np.where(candidate_mask)[0].tolist()

        # 4. Enforce minimum chunk constraints
        final_boundaries = self._enforce_min_chunk_size(
            candidate_indices,
            min_size=self.config.min_chunk_sentences,
            total=n,
        )

        diagnostics = {
            "smoothed_scores": smoothed_scores,
            "z_scores": z_scores,
            "candidates": candidate_indices,
            "final_boundaries": final_boundaries,
        }

        return final_boundaries, diagnostics
        
    def _smooth_signal(self, scores: np.ndarray) -> np.ndarray:
        """
        scipy.signal.savgol_filter with configured window and poly order.
        Clips output to non-negative values.
        Handles edge case: if len(scores) < smoothing_window, return scores unchanged.
        """
        n = len(scores)
        if n < self.config.smoothing_window:
            logger.debug(
                f"Signal length {n} is smaller than smoothing window {self.config.smoothing_window}. "
                f"Returning raw scores."
            )
            return np.clip(scores, 0, None)

        window = self.config.smoothing_window
        polyorder = self.config.smoothing_poly_order

        # Savitzky-Golay window must be odd
        if window % 2 == 0:
            window -= 1

        # Polynomial order must be less than window
        if polyorder >= window:
            polyorder = max(1, window - 1)

        try:
            smoothed = savgol_filter(scores, window_length=window, polyorder=polyorder)
            return np.clip(smoothed, 0, None)
        except Exception as e:
            logger.error(f"Savitzky-Golay filtering failed: {e}. Returning raw clipped scores.")
            return np.clip(scores, 0, None)
            
    def _rolling_zscore(
        self,
        scores: np.ndarray,
        lag: int
    ) -> np.ndarray:
        """
        For each position i >= lag:
          mu = mean(scores[i-lag:i])
          sigma = std(scores[i-lag:i]) or 1.0 if sigma==0
          z[i] = (scores[i] - mu) / sigma
        Positions i < lag: z[i] = 0.0
        """
        n = len(scores)
        z = np.zeros(n, dtype=np.float32)

        for i in range(n):
            if i < lag:
                z[i] = 0.0
                continue

            window = scores[i - lag:i]
            mu = float(np.mean(window))
            sigma = float(np.std(window))

            if sigma == 0.0:
                sigma = 1.0

            z[i] = (scores[i] - mu) / sigma

        return z
        
    def _local_maxima_mask(
        self,
        scores: np.ndarray,
        order: int = 2
    ) -> np.ndarray:
        """
        Boolean mask. Uses scipy.signal.argrelmax with order=order.
        Returns np.ndarray[bool] of same length as scores.
        """
        mask = np.zeros(len(scores), dtype=bool)
        if len(scores) <= order * 2:
            return mask

        maxima_indices = argrelmax(scores, order=order)[0]
        mask[maxima_indices] = True
        return mask
        
    def _enforce_min_chunk_size(
        self,
        candidates: list[int],
        min_size: int,
        total: int
    ) -> list[int]:
        """
        Given candidate boundary positions, remove those that would create
        a chunk shorter than min_size sentences.
        Algorithm: iterate candidates, only keep if distance from last 
        kept boundary >= min_size. Always include position 0 and total.
        """
        # Ensure candidates are sorted and exclude 0/total for processing
        clean_candidates = sorted(list(set([c for c in candidates if 0 < c < total])))

        kept = [0]
        for c in clean_candidates:
            # Distance from last kept boundary must be at least min_size
            if c - kept[-1] >= min_size:
                kept.append(c)

        # Enforce that the last chunk (from kept[-1] to total) is at least min_size
        # If not, merge the last chunk by dropping the last boundary, repeat if needed
        while len(kept) > 1 and (total - kept[-1]) < min_size:
            logger.debug(
                f"Dropping boundary at index {kept[-1]} to satisfy "
                f"tail chunk constraint (tail size {total - kept[-1]} < {min_size})"
            )
            kept.pop()

        # Always include total at the end
        if kept[-1] != total:
            kept.append(total)

        return kept
        
    def visualize(
        self,
        perplexity_scores: np.ndarray,
        boundaries: list[int],
        sentences: list[str] | None = None,
        save_path: str | None = None,
        show: bool = True
    ) -> None:
        """
        4-panel matplotlib figure:
        Panel 1: Raw perplexity scores (blue line)
        Panel 2: Smoothed perplexity + detected boundaries (red dashed vertical lines)
        Panel 3: Rolling z-scores + threshold line (orange horizontal)
        Panel 4: Final chunks visualization (colored spans)
        Title: "Adaptive Semantic Chunking — Boundary Detection"
        """
        sns.set_theme(style="whitegrid")
        n = len(perplexity_scores)
        x = np.arange(n)

        # Create 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
        fig.suptitle("Adaptive Semantic Chunking — Boundary Detection", fontsize=16, weight="bold", y=0.98)

        # Panel 1: Raw perplexity scores
        axes[0].plot(x, perplexity_scores, "o-", color="royalblue", linewidth=1.5, label="Raw Perplexity")
        axes[0].set_title("Panel 1: Raw Perplexity Scores", fontsize=12, weight="bold")
        axes[0].set_ylabel("Perplexity")
        axes[0].legend(loc="upper left")

        # Get smoothed scores for boundary plotting
        smoothed = self._smooth_signal(perplexity_scores)

        # Panel 2: Smoothed perplexity + detected boundaries
        axes[1].plot(x, smoothed, "g-", linewidth=2, label="Smoothed Perplexity")
        mid_boundaries = [b for b in boundaries if 0 < b < n]
        for b in mid_boundaries:
            axes[1].axvline(x=b, color="red", linestyle="--", linewidth=1.5, alpha=0.8)
            axes[1].text(b + 0.1, np.max(smoothed) * 0.9, f"Boundary @ {b}", color="red", weight="bold", fontsize=9)
            
        axes[1].set_title("Panel 2: Smoothed Signal & Detected Boundaries", fontsize=12, weight="bold")
        axes[1].set_ylabel("Smoothed Perplexity")
        axes[1].legend(loc="upper left")

        # Panel 3: Rolling z-scores + threshold line
        z_scores = self._rolling_zscore(smoothed, self.config.lag_window)
        axes[2].plot(x, z_scores, "purple", linewidth=1.8, label="Rolling Z-Score")
        axes[2].axhline(y=self.config.z_score_threshold, color="orange", linestyle=":", linewidth=2, label="Surprise Threshold")
        
        # Mark boundary candidates
        for b in mid_boundaries:
            axes[2].plot(b, z_scores[b], "ro", markersize=6)

        axes[2].set_title("Panel 3: Surprise Rolling Z-Scores", fontsize=12, weight="bold")
        axes[2].set_ylabel("Z-Score")
        axes[2].legend(loc="upper left")

        # Panel 4: Final chunks visualization (colored spans)
        colors = ["#E8F8F5", "#FEF9E7", "#EBF5FB", "#F5EEF8", "#FDEDEC"]
        
        axes[3].set_ylim(0, 1)
        axes[3].get_yaxis().set_visible(False)
        
        for idx in range(len(boundaries) - 1):
            start = boundaries[idx]
            end = boundaries[idx + 1]
            color = colors[idx % len(colors)]
            axes[3].axvspan(start, end, alpha=0.7, color=color)
            
            mid = (start + end) / 2
            preview = f"Chunk {idx+1}\n(size: {end-start} sents)"
            if sentences and start < len(sentences):
                snippet = sentences[start][:25].strip() + "..."
                preview += f"\n\"{snippet}\""
            axes[3].text(mid, 0.5, preview, ha="center", va="center", fontsize=9, weight="bold", color="#34495E")

        axes[3].set_title("Panel 4: Final Coherent Semantic Chunks", fontsize=12, weight="bold")
        axes[3].set_xlabel("Sentence Index")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"Saved diagnostic plots to: {save_path}")
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    # Setup testing logger
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting BoundaryDetector unit tests...")

    # Define unit tests
    def test_smooth_signal() -> None:
        detector = BoundaryDetector(BoundaryDetectorConfig(smoothing_window=3))
        # Test signal shorter than window
        short_signal = np.array([5.0, 10.0])
        smoothed = detector._smooth_signal(short_signal)
        assert np.array_equal(smoothed, short_signal)

        # Test standard signal smoothing and clipping
        signal = np.array([2.0, 4.0, -1.0, 8.0, 2.0])
        smoothed_clipped = detector._smooth_signal(signal)
        assert len(smoothed_clipped) == 5
        assert np.all(smoothed_clipped >= 0)
        logger.info("test_smooth_signal passed.")

    def test_rolling_zscore() -> None:
        detector = BoundaryDetector(BoundaryDetectorConfig(lag_window=2))
        scores = np.array([1.0, 1.0, 1.0, 10.0, 1.0])
        z = detector._rolling_zscore(scores, lag=2)
        assert len(z) == 5
        assert z[0] == 0.0
        assert z[1] == 0.0
        assert abs(z[3] - 9.0) < 1e-5
        logger.info("test_rolling_zscore passed.")

    def test_local_maxima_mask() -> None:
        detector = BoundaryDetector()
        scores = np.array([1.0, 2.0, 5.0, 2.0, 1.0, 4.0, 1.0])
        mask = detector._local_maxima_mask(scores, order=1)
        assert mask[2] == True
        assert mask[5] == True
        assert mask[0] == False
        assert mask[1] == False
        logger.info("test_local_maxima_mask passed.")

    def test_enforce_min_chunk_size() -> None:
        detector = BoundaryDetector()
        boundaries = detector._enforce_min_chunk_size([2, 5, 8], min_size=3, total=10)
        assert boundaries == [0, 5, 10]
        logger.info("test_enforce_min_chunk_size passed.")

    def test_end_to_end_detector() -> None:
        config = BoundaryDetectorConfig(
            z_score_threshold=1.5,
            lag_window=3,
            min_chunk_sentences=3,
            smoothing_window=3,
            smoothing_poly_order=1,
            require_local_maximum=True
        )
        detector = BoundaryDetector(config)
        perplexity_scores = np.array([10.0, 10.0, 10.0, 12.0, 15.0, 50.0, 15.0, 12.0, 10.0, 10.0])
        boundaries, diag = detector.detect_boundaries(perplexity_scores)
        
        assert boundaries[0] == 0
        assert boundaries[-1] == 10
        for i in range(len(boundaries) - 1):
            assert (boundaries[i+1] - boundaries[i]) >= 3
            
        logger.info(f"test_end_to_end_detector passed. Boundaries: {boundaries}")

    # Run tests
    test_smooth_signal()
    test_rolling_zscore()
    test_local_maxima_mask()
    test_enforce_min_chunk_size()
    test_end_to_end_detector()
    logger.info("All boundary detector tests passed successfully!")
