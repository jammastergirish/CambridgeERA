"""Tests for experiment/activation_separation_analysis.py — metrics and plotting."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import numpy as np
import pytest

from activation_separation_analysis import (
    compute_separation_metrics,
    plot_separation_analysis,
)


# ---------------------------------------------------------------------------
# compute_separation_metrics
# ---------------------------------------------------------------------------

class TestComputeSeparationMetrics:
    def test_returns_all_expected_keys(self):
        np.random.seed(42)
        forget = np.random.randn(20, 16).astype(np.float32)
        retain = np.random.randn(20, 16).astype(np.float32)
        result = compute_separation_metrics(forget, retain)
        expected_keys = {
            "cosine_distance",
            "euclidean_distance",
            "linear_discriminability_auc",
            "variance_ratio",
            "forget_centroid_norm",
            "retain_centroid_norm",
        }
        assert set(result.keys()) == expected_keys

    def test_identical_activations_give_zero_distance(self):
        np.random.seed(42)
        activations = np.random.randn(30, 8).astype(np.float32)
        result = compute_separation_metrics(activations, activations.copy())
        assert abs(result["cosine_distance"]) < 1e-5
        assert abs(result["euclidean_distance"]) < 1e-5

    def test_identical_activations_give_zero_variance_ratio(self):
        np.random.seed(42)
        activations = np.random.randn(30, 8).astype(np.float32)
        result = compute_separation_metrics(activations, activations.copy())
        assert abs(result["variance_ratio"]) < 1e-5

    def test_well_separated_clusters_have_high_distance(self):
        """Two clusters far apart should have large euclidean & cosine distance."""
        np.random.seed(42)
        forget = np.random.randn(50, 8).astype(np.float32) + 10.0
        retain = np.random.randn(50, 8).astype(np.float32) - 10.0
        result = compute_separation_metrics(forget, retain)
        assert result["euclidean_distance"] > 10.0
        assert result["cosine_distance"] > 0.5

    def test_well_separated_clusters_have_high_auc(self):
        """Clearly separable clusters should yield AUC well above 0.5."""
        np.random.seed(42)
        forget = np.random.randn(50, 8).astype(np.float32) + 10.0
        retain = np.random.randn(50, 8).astype(np.float32) - 10.0
        result = compute_separation_metrics(forget, retain)
        assert result["linear_discriminability_auc"] > 0.9

    def test_cosine_distance_is_bounded(self):
        np.random.seed(42)
        forget = np.random.randn(20, 8).astype(np.float32)
        retain = np.random.randn(20, 8).astype(np.float32)
        result = compute_separation_metrics(forget, retain)
        # Cosine distance ∈ [0, 2]
        assert 0.0 <= result["cosine_distance"] <= 2.0

    def test_centroid_norms_are_positive(self):
        np.random.seed(42)
        forget = np.random.randn(20, 8).astype(np.float32) + 5.0
        retain = np.random.randn(20, 8).astype(np.float32) + 3.0
        result = compute_separation_metrics(forget, retain)
        assert result["forget_centroid_norm"] > 0
        assert result["retain_centroid_norm"] > 0

    def test_variance_ratio_positive_for_separated_clusters(self):
        np.random.seed(42)
        forget = np.random.randn(30, 4).astype(np.float32) + 5.0
        retain = np.random.randn(30, 4).astype(np.float32) - 5.0
        result = compute_separation_metrics(forget, retain)
        assert result["variance_ratio"] > 0

    def test_single_sample_per_class(self):
        """Even with 1 sample per class, metrics should not crash."""
        np.random.seed(42)
        forget = np.array([[1.0, 2.0, 3.0, 4.0]])
        retain = np.array([[5.0, 6.0, 7.0, 8.0]])
        result = compute_separation_metrics(forget, retain)
        # Should return valid numbers (AUC may fall back to 0.5)
        assert "cosine_distance" in result
        assert "euclidean_distance" in result

    def test_opposite_centroids_high_cosine_distance(self):
        """If centroids point in opposite directions, cosine distance ≈ 2."""
        forget = np.array([[1.0, 0.0, 0.0, 0.0]] * 10, dtype=np.float32)
        retain = np.array([[-1.0, 0.0, 0.0, 0.0]] * 10, dtype=np.float32)
        result = compute_separation_metrics(forget, retain)
        assert abs(result["cosine_distance"] - 2.0) < 0.01


# ---------------------------------------------------------------------------
# plot_separation_analysis (smoke tests)
# ---------------------------------------------------------------------------

def _make_dummy_results(num_layers: int = 4, offset: float = 0.0):
    """Create dummy per-layer results for plotting tests."""
    results = []
    for layer in range(num_layers):
        results.append({
            "layer": layer,
            "cosine_distance": 0.1 + layer * 0.05 + offset,
            "euclidean_distance": 1.0 + layer * 0.5 + offset,
            "linear_discriminability_auc": 0.6 + layer * 0.05 + offset * 0.1,
            "variance_ratio": 0.5 + layer * 0.1 + offset,
            "forget_centroid_norm": 5.0 + layer,
            "retain_centroid_norm": 4.5 + layer,
        })
    return results


class TestPlotSeparationAnalysis:
    def test_smoke_creates_png(self, temp_dir):
        results_a = _make_dummy_results(num_layers=4, offset=0.0)
        results_b = _make_dummy_results(num_layers=4, offset=0.1)
        plot_separation_analysis(results_a, results_b, temp_dir, title="Test")
        assert os.path.isfile(os.path.join(temp_dir, "activation_separation_analysis.png"))

    def test_smoke_without_title(self, temp_dir):
        results_a = _make_dummy_results(num_layers=3)
        results_b = _make_dummy_results(num_layers=3, offset=0.2)
        plot_separation_analysis(results_a, results_b, temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "activation_separation_analysis.png"))

    def test_single_layer(self, temp_dir):
        """Should work even with just one layer."""
        results_a = _make_dummy_results(num_layers=1)
        results_b = _make_dummy_results(num_layers=1, offset=0.05)
        plot_separation_analysis(results_a, results_b, temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "activation_separation_analysis.png"))
