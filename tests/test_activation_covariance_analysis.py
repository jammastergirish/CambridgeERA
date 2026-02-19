"""Tests for experiment/activation_covariance_analysis.py — metrics, comparison, plotting."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import numpy as np
import pytest

from activation_covariance_analysis import (
    compute_covariance_metrics,
    compare_spectra,
    plot_covariance_analysis,
)


# ---------------------------------------------------------------------------
# compute_covariance_metrics
# ---------------------------------------------------------------------------

class TestComputeCovarianceMetrics:
    def test_returns_expected_keys(self):
        np.random.seed(42)
        activations = np.random.randn(100, 8).astype(np.float32)
        result = compute_covariance_metrics(activations)
        assert "eigenvalues" in result
        assert "effective_rank" in result
        assert "spectral_entropy" in result
        assert "trace" in result

    def test_identity_covariance(self):
        """Uncorrelated axes → eigenvalues ≈ equal → high effective rank."""
        np.random.seed(42)
        activations = np.random.randn(500, 4).astype(np.float32)
        result = compute_covariance_metrics(activations)
        assert result["effective_rank"] >= 3  # near‐full rank

    def test_rank_one_activations(self):
        """All activations along one direction → effective rank = 1."""
        direction = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        activations = np.outer(np.random.randn(100), direction).astype(np.float32)
        result = compute_covariance_metrics(activations)
        assert result["effective_rank"] == 1

    def test_spectral_entropy_positive(self):
        np.random.seed(42)
        activations = np.random.randn(80, 8).astype(np.float32)
        result = compute_covariance_metrics(activations)
        assert result["spectral_entropy"] > 0

    def test_trace_equals_sum_eigenvalues(self):
        np.random.seed(42)
        activations = np.random.randn(60, 6).astype(np.float32)
        result = compute_covariance_metrics(activations)
        assert abs(result["trace"] - sum(result["eigenvalues"])) < 0.01

    def test_max_eigenvalue_first(self):
        np.random.seed(42)
        activations = np.random.randn(50, 4).astype(np.float32)
        result = compute_covariance_metrics(activations)
        assert result["max_eigenvalue"] == pytest.approx(result["eigenvalues"][0])

    def test_top_k_concentration_bounded(self):
        np.random.seed(42)
        activations = np.random.randn(100, 8).astype(np.float32)
        result = compute_covariance_metrics(activations, top_k=4)
        assert 0.0 <= result["top_k_concentration"] <= 1.0


# ---------------------------------------------------------------------------
# compare_spectra
# ---------------------------------------------------------------------------

class TestCompareSpectra:
    def test_identical_spectra_zero_distance(self):
        spectrum = compute_covariance_metrics(np.random.randn(100, 4).astype(np.float32))
        result = compare_spectra(spectrum, spectrum)
        assert abs(result["wasserstein_distance"]) < 1e-6
        assert result["effective_rank_change"] == 0

    def test_different_spectra_nonzero_distance(self):
        np.random.seed(42)
        spec_a = compute_covariance_metrics(np.random.randn(100, 4).astype(np.float32))
        spec_b = compute_covariance_metrics(np.random.randn(100, 4).astype(np.float32) * 5)
        result = compare_spectra(spec_a, spec_b)
        assert result["wasserstein_distance"] > 0

    def test_entropy_change_sign(self):
        np.random.seed(42)
        # high-rank → low-rank should decrease entropy
        high_rank = np.random.randn(200, 8).astype(np.float32)
        low_rank = np.outer(np.random.randn(200), np.ones(8)).astype(np.float32)
        spec_high = compute_covariance_metrics(high_rank)
        spec_low = compute_covariance_metrics(low_rank)
        result = compare_spectra(spec_high, spec_low)
        assert result["entropy_change"] < 0


# ---------------------------------------------------------------------------
# plot_covariance_analysis (smoke tests)
# ---------------------------------------------------------------------------

def _make_dummy_cov_results(num_layers=3):
    results = []
    for layer in range(num_layers):
        results.append({
            "layer": layer * 4,
            "forget_eff_rank_a": 10 + layer,
            "forget_eff_rank_b": 12 + layer,
            "retain_eff_rank_a": 11 + layer,
            "retain_eff_rank_b": 11 + layer,
            "forget_entropy_a": 2.0 + layer * 0.1,
            "forget_entropy_b": 2.1 + layer * 0.1,
            "retain_entropy_a": 2.0 + layer * 0.1,
            "retain_entropy_b": 2.0 + layer * 0.1,
            "forget_wasserstein": 0.5 + layer * 0.1,
            "retain_wasserstein": 0.3 + layer * 0.05,
            "forget_top10_change": 0.1 + layer * 0.02,
            "retain_top10_change": 0.05 + layer * 0.01,
        })
    return results


class TestPlotCovarianceAnalysis:
    def test_smoke_creates_png(self, temp_dir):
        results = _make_dummy_cov_results()
        plot_covariance_analysis(results, temp_dir, title="Test")
        assert os.path.isfile(os.path.join(temp_dir, "covariance_analysis.png"))

    def test_smoke_without_title(self, temp_dir):
        results = _make_dummy_cov_results(2)
        plot_covariance_analysis(results, temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "covariance_analysis.png"))
