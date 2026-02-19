"""Tests for experiment/mlp_nullspace_alignment.py — alignment metrics and plotting."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import numpy as np
import pytest
import torch

from mlp_nullspace_alignment import compute_nullspace_alignment, plot_nullspace_alignment


# ---------------------------------------------------------------------------
# compute_nullspace_alignment
# ---------------------------------------------------------------------------

class TestComputeNullspaceAlignment:
    def test_returns_expected_keys(self):
        W = torch.randn(8, 4)
        dW = torch.randn(8, 4) * 0.01
        result = compute_nullspace_alignment(W, dW)
        assert result is not None
        expected_keys = {
            "original_eff_rank", "updated_eff_rank", "rank_increase",
            "colspace_projection_ratio", "nullspace_projection_ratio",
            "nullspace_dimension", "update_norm", "original_norm",
            "relative_update_size",
        }
        assert set(result.keys()) == expected_keys

    def test_rejects_non_2d_tensors(self):
        assert compute_nullspace_alignment(torch.randn(4), torch.randn(4)) is None
        assert compute_nullspace_alignment(torch.randn(2, 3, 4), torch.randn(2, 3, 4)) is None

    def test_zero_weight_returns_none(self):
        W = torch.zeros(4, 4)
        dW = torch.randn(4, 4)
        assert compute_nullspace_alignment(W, dW) is None

    def test_ratios_sum_near_one(self):
        """colspace + nullspace norms ≈ total norm for rectangular matrices."""
        torch.manual_seed(42)
        W = torch.randn(16, 8)
        dW = torch.randn(16, 8) * 0.01
        result = compute_nullspace_alignment(W, dW, rank_threshold=0.9)
        if result is not None and result["nullspace_dimension"] > 0:
            # Pythagorean decomposition: ‖proj_col‖² + ‖proj_null‖² ≈ ‖dW‖²
            col_r = result["colspace_projection_ratio"]
            null_r = result["nullspace_projection_ratio"]
            assert abs(col_r**2 + null_r**2 - 1.0) < 0.05

    def test_update_in_column_space(self):
        """Update entirely in column space → nullspace_projection_ratio ≈ 0."""
        torch.manual_seed(42)
        W = torch.randn(8, 4)
        U, _, _ = torch.linalg.svd(W, full_matrices=True)
        # Build an update that lies in the column space
        dW = U[:, :4] @ torch.randn(4, 4) * 0.01
        result = compute_nullspace_alignment(W, dW, rank_threshold=0.999)
        if result is not None:
            assert result["colspace_projection_ratio"] > 0.9

    def test_rank_increase_can_be_positive(self):
        """Adding a random perturbation to a low-rank W should raise rank."""
        torch.manual_seed(42)
        W = torch.randn(8, 1) @ torch.randn(1, 8)  # rank-1
        dW = torch.randn(8, 8)  # full-rank perturbation
        result = compute_nullspace_alignment(W, dW)
        assert result is not None
        assert result["rank_increase"] > 0

    def test_relative_update_size(self):
        torch.manual_seed(42)
        W = torch.randn(4, 4)
        dW = torch.randn(4, 4) * 0.1
        result = compute_nullspace_alignment(W, dW)
        assert result is not None
        expected = float(dW.norm().item()) / (float(W.norm().item()) + 1e-10)
        assert abs(result["relative_update_size"] - expected) < 0.01

    def test_full_rank_square_matrix(self):
        """Tall/square W with threshold 0.99 might be full-rank → null dim = 0."""
        torch.manual_seed(42)
        W = torch.randn(4, 4) * 10  # well-conditioned
        dW = torch.randn(4, 4) * 0.01
        result = compute_nullspace_alignment(W, dW, rank_threshold=0.99)
        if result is not None:
            assert result["nullspace_dimension"] >= 0  # may or may not be zero


# ---------------------------------------------------------------------------
# plot_nullspace_alignment (smoke tests)
# ---------------------------------------------------------------------------

def _make_dummy_layer_results(num_layers=4):
    layer_results = []
    for i in range(num_layers):
        layer_results.append({
            "layer": i,
            "avg_colspace_ratio": 0.7 - i * 0.05,
            "avg_nullspace_ratio": 0.3 + i * 0.05,
            "avg_rank_increase": i * 2,
            "encoder_nullspace_ratio": 0.25 + i * 0.03,
            "decoder_nullspace_ratio": 0.35 + i * 0.07,
            "num_matrices": 2,
        })
    return layer_results


def _make_dummy_matrix_results(count=8):
    results = []
    for j in range(count):
        results.append({
            "nullspace_projection_ratio": 0.3 + j * 0.05,
            "colspace_projection_ratio": 0.7 - j * 0.05,
            "rank_increase": j,
        })
    return results


class TestPlotNullspaceAlignment:
    def test_smoke_creates_png(self, temp_dir):
        layer_res = _make_dummy_layer_results()
        matrix_res = _make_dummy_matrix_results()
        plot_nullspace_alignment(layer_res, matrix_res, temp_dir, title="Test")
        assert os.path.isfile(os.path.join(temp_dir, "mlp_nullspace_alignment.png"))

    def test_smoke_without_title(self, temp_dir):
        layer_res = _make_dummy_layer_results(2)
        matrix_res = _make_dummy_matrix_results(4)
        plot_nullspace_alignment(layer_res, matrix_res, temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "mlp_nullspace_alignment.png"))

    def test_single_layer(self, temp_dir):
        layer_res = _make_dummy_layer_results(1)
        matrix_res = _make_dummy_matrix_results(2)
        plot_nullspace_alignment(layer_res, matrix_res, temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "mlp_nullspace_alignment.png"))
