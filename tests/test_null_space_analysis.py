"""Tests for experiment/null_space_analysis.py — SVD metrics, alignment, and aggregation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import math
import torch
import pytest

from null_space_analysis import (
    compute_null_space_projection,
    analyze_subspace_alignment,
    aggregate_by_component,
    plot_null_space_results,
    COMPONENT_LABELS,
)


# ---------------------------------------------------------------------------
# compute_null_space_projection
# ---------------------------------------------------------------------------

class TestComputeNullSpaceProjection:
    def test_zero_matrix_returns_zero_rank(self):
        weight_change = torch.zeros(10, 10)
        result = compute_null_space_projection(weight_change)
        assert result["effective_rank"] == 0
        assert result["max_singular_value"] == 0.0

    def test_identity_returns_full_rank(self):
        weight_change = torch.eye(8)
        result = compute_null_space_projection(weight_change)
        # All singular values are 1.0, so effective rank should equal matrix size
        assert result["effective_rank"] == 8
        assert abs(result["max_singular_value"] - 1.0) < 1e-5

    def test_rank_one_matrix(self):
        """A rank-1 matrix should have effective_rank = 1 and high top10 variance."""
        row = torch.randn(1, 16)
        weight_change = row.T @ row  # rank 1, shape [16, 16]
        result = compute_null_space_projection(weight_change)
        assert result["effective_rank"] == 1
        assert result["top10_variance_ratio"] > 0.99

    def test_top10_variance_ratio_is_bounded(self):
        weight_change = torch.randn(20, 20)
        result = compute_null_space_projection(weight_change)
        assert 0.0 < result["top10_variance_ratio"] <= 1.0

    def test_singular_value_decay_less_than_one_for_full_rank(self):
        """For a full-rank random matrix with > 10 singular values, decay < 1."""
        torch.manual_seed(42)
        weight_change = torch.randn(30, 30)
        result = compute_null_space_projection(weight_change)
        assert result["singular_value_decay"] < 1.0

    def test_singular_value_decay_one_for_small_matrix(self):
        """If matrix has ≤ 10 singular values, decay should default to 1.0."""
        weight_change = torch.randn(5, 5)
        result = compute_null_space_projection(weight_change)
        assert result["singular_value_decay"] == 1.0

    def test_empty_tensor_returns_defaults(self):
        weight_change = torch.tensor([])
        result = compute_null_space_projection(weight_change)
        assert result["effective_rank"] == 0

    def test_1d_tensor_returns_defaults(self):
        weight_change = torch.randn(10)
        result = compute_null_space_projection(weight_change)
        assert result["effective_rank"] == 0

    def test_custom_rank_threshold(self):
        """Lower threshold → lower effective rank."""
        torch.manual_seed(42)
        weight_change = torch.randn(20, 20)
        result_low = compute_null_space_projection(weight_change, rank_threshold=0.5)
        result_high = compute_null_space_projection(weight_change, rank_threshold=0.99)
        assert result_low["effective_rank"] <= result_high["effective_rank"]

    def test_scaled_identity_max_singular_value(self):
        weight_change = 3.0 * torch.eye(10)
        result = compute_null_space_projection(weight_change)
        assert abs(result["max_singular_value"] - 3.0) < 1e-5


# ---------------------------------------------------------------------------
# analyze_subspace_alignment
# ---------------------------------------------------------------------------

class TestAnalyzeSubspaceAlignment:
    def test_identical_matrices_give_perfect_alignment(self):
        weight = torch.randn(16, 16)
        result = analyze_subspace_alignment(weight, weight)
        assert abs(result["subspace_alignment"] - 1.0) < 1e-4
        assert abs(result["grassmann_distance"]) < 0.01

    def test_orthogonal_shift_gives_low_alignment(self):
        """Multiplying by an orthogonal rotation should reduce alignment."""
        torch.manual_seed(42)
        weight_before = torch.randn(32, 32)
        # Create an orthogonal matrix via QR decomposition
        random_matrix = torch.randn(32, 32)
        orthogonal, _ = torch.linalg.qr(random_matrix)
        weight_after = orthogonal @ weight_before
        result = analyze_subspace_alignment(weight_before, weight_after, num_top_vectors=10)
        # After rotation, alignment should be noticeably below 1.0
        assert result["subspace_alignment"] < 0.95

    def test_singular_value_ratio_for_scaled_matrix(self):
        weight_before = torch.randn(10, 10)
        weight_after = 2.0 * weight_before
        result = analyze_subspace_alignment(weight_before, weight_after)
        # σ₁(2W) / σ₁(W) ≈ 2.0
        assert abs(result["singular_value_ratio"] - 2.0) < 0.1

    def test_returns_empty_for_empty_tensor(self):
        result = analyze_subspace_alignment(torch.tensor([]), torch.tensor([]))
        assert result == {}

    def test_returns_empty_for_1d_tensor(self):
        result = analyze_subspace_alignment(torch.randn(10), torch.randn(10))
        assert result == {}

    def test_grassmann_distance_is_nonnegative(self):
        torch.manual_seed(42)
        weight_before = torch.randn(16, 16)
        weight_after = torch.randn(16, 16)
        result = analyze_subspace_alignment(weight_before, weight_after)
        assert result["grassmann_distance"] >= 0.0

    def test_num_top_vectors_capped_to_matrix_rank(self):
        """Asking for more top vectors than the matrix has should not crash."""
        weight = torch.randn(4, 4)
        result = analyze_subspace_alignment(weight, weight, num_top_vectors=100)
        assert abs(result["subspace_alignment"] - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# aggregate_by_component
# ---------------------------------------------------------------------------

class TestAggregateByComponent:
    def test_groups_by_component(self):
        results = [
            {"component": "qkv", "top10_variance_ratio": 0.8, "subspace_alignment": 0.9},
            {"component": "qkv", "top10_variance_ratio": 0.7, "subspace_alignment": 0.85},
            {"component": "proj", "top10_variance_ratio": 0.6, "subspace_alignment": 0.7},
        ]
        aggregated = aggregate_by_component(results)
        assert len(aggregated["qkv"]["null_space"]) == 2
        assert len(aggregated["proj"]["null_space"]) == 1
        assert len(aggregated["mlp_expand"]["null_space"]) == 0

    def test_ignores_unknown_components(self):
        results = [
            {"component": "other", "top10_variance_ratio": 0.5, "subspace_alignment": 0.5},
        ]
        aggregated = aggregate_by_component(results)
        for label in COMPONENT_LABELS:
            assert aggregated[label]["null_space"] == []

    def test_empty_input(self):
        aggregated = aggregate_by_component([])
        for label in COMPONENT_LABELS:
            assert aggregated[label]["null_space"] == []
            assert aggregated[label]["alignment"] == []

    def test_values_are_correct(self):
        results = [
            {"component": "mlp_expand", "top10_variance_ratio": 0.95, "subspace_alignment": 0.88},
        ]
        aggregated = aggregate_by_component(results)
        assert abs(aggregated["mlp_expand"]["null_space"][0] - 0.95) < 1e-8
        assert abs(aggregated["mlp_expand"]["alignment"][0] - 0.88) < 1e-8


# ---------------------------------------------------------------------------
# plot_null_space_results (smoke test)
# ---------------------------------------------------------------------------

class TestPlotNullSpaceResults:
    def test_smoke_creates_png(self, temp_dir):
        component_results = {
            "qkv": {"null_space": [0.8, 0.9], "alignment": [0.7, 0.8]},
            "proj": {"null_space": [0.6], "alignment": [0.5]},
            "mlp_expand": {"null_space": [0.95], "alignment": [0.9]},
            "mlp_contract": {"null_space": [0.85], "alignment": [0.75]},
        }
        plot_null_space_results(component_results, temp_dir, title="Test")
        assert os.path.isfile(os.path.join(temp_dir, "null_space_visualization.png"))

    def test_no_crash_with_empty_data(self, temp_dir):
        component_results = {label: {"null_space": [], "alignment": []} for label in COMPONENT_LABELS}
        plot_null_space_results(component_results, temp_dir)
        assert not os.path.isfile(os.path.join(temp_dir, "null_space_visualization.png"))
