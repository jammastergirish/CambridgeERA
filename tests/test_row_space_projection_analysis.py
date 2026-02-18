"""Tests for experiment/row_space_projection_analysis.py — projection metrics and plotting."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import numpy as np
import pytest

from row_space_projection_analysis import compute_row_space_projection, plot_row_space_projections


# ---------------------------------------------------------------------------
# compute_row_space_projection
# ---------------------------------------------------------------------------

class TestComputeRowSpaceProjection:
    def test_returns_expected_keys(self):
        np.random.seed(42)
        activations = [np.random.randn(4, 8).astype(np.float32)]
        weight_update = np.random.randn(16, 8).astype(np.float32)
        result = compute_row_space_projection(activations, weight_update)
        assert result is not None
        assert set(result.keys()) == {
            "projection_norm", "original_norm", "projection_ratio",
            "variance_ratio", "top_alignments", "num_samples",
        }

    def test_rejects_empty_activations(self):
        assert compute_row_space_projection([], np.ones((4, 4))) is None

    def test_rejects_none_weight_update(self):
        assert compute_row_space_projection([np.ones((2, 4))], None) is None

    def test_projection_ratio_bounded(self):
        np.random.seed(42)
        activations = [np.random.randn(10, 8).astype(np.float32)]
        weight_update = np.random.randn(4, 8).astype(np.float32)
        result = compute_row_space_projection(activations, weight_update)
        assert result is not None
        assert 0.0 <= result["projection_ratio"] <= 2.0

    def test_aligned_activations_high_ratio(self):
        """Activations in the row space of dW should have high projection ratio."""
        np.random.seed(42)
        weight_update = np.random.randn(4, 8).astype(np.float32)
        U, _, _ = np.linalg.svd(weight_update.T, full_matrices=False)
        # Create activations that lie in the top-k row space
        activations = [U[:, :4] @ np.random.randn(4, 5)  ]  # shape (8, 5), transpose
        activations = [(U[:, :4] @ np.random.randn(4, 10)).T.astype(np.float32)]  # (10, 8)
        result = compute_row_space_projection(activations, weight_update, top_k=4)
        assert result is not None
        assert result["projection_ratio"] > 0.5

    def test_top_alignments_length(self):
        np.random.seed(42)
        activations = [np.random.randn(10, 8).astype(np.float32)]
        weight_update = np.random.randn(4, 8).astype(np.float32)
        result = compute_row_space_projection(activations, weight_update)
        assert result is not None
        assert len(result["top_alignments"]) <= 5

    def test_num_samples_correct(self):
        np.random.seed(42)
        activations = [np.random.randn(5, 8).astype(np.float32),
                        np.random.randn(3, 8).astype(np.float32)]
        weight_update = np.random.randn(4, 8).astype(np.float32)
        result = compute_row_space_projection(activations, weight_update)
        assert result is not None
        assert result["num_samples"] == 8  # 5 + 3

    def test_multiple_activation_batches(self):
        np.random.seed(42)
        act1 = np.random.randn(4, 8).astype(np.float32)
        act2 = np.random.randn(6, 8).astype(np.float32)
        weight_update = np.random.randn(4, 8).astype(np.float32)
        result = compute_row_space_projection([act1, act2], weight_update)
        assert result is not None
        assert result["num_samples"] == 10


# ---------------------------------------------------------------------------
# plot_row_space_projections (smoke tests)
# ---------------------------------------------------------------------------

def _make_dummy_layer_results(num_layers=3):
    results = []
    for i in range(num_layers):
        results.append({
            "layer": i * 4,
            "avg_forget_proj": 0.3 + i * 0.05,
            "avg_retain_proj": 0.2 + i * 0.03,
            "avg_diff": 0.1 + i * 0.02,
        })
    return results


def _make_dummy_per_weight_results(count=6):
    results = []
    for j in range(count):
        results.append({
            "forget_proj_ratio": 0.3 + j * 0.05,
            "retain_proj_ratio": 0.2 + j * 0.03,
            "forget_stronger": True if j % 2 == 0 else False,
        })
    return results


class TestPlotRowSpaceProjections:
    def test_smoke_creates_png(self, temp_dir):
        layer_res = _make_dummy_layer_results()
        per_weight_res = _make_dummy_per_weight_results()
        plot_row_space_projections(layer_res, per_weight_res, temp_dir, title="Test")
        assert os.path.isfile(os.path.join(temp_dir, "row_space_projections.png"))

    def test_smoke_without_title(self, temp_dir):
        layer_res = _make_dummy_layer_results(2)
        per_weight_res = _make_dummy_per_weight_results(4)
        plot_row_space_projections(layer_res, per_weight_res, temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "row_space_projections.png"))

    def test_single_layer(self, temp_dir):
        layer_res = _make_dummy_layer_results(1)
        per_weight_res = _make_dummy_per_weight_results(1)
        plot_row_space_projections(layer_res, per_weight_res, temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "row_space_projections.png"))
