"""Tests for experiment/analyze_mlp_vs_attn.py — summary logic and plot smoke tests."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import pandas as pd
import pytest

from analyze_mlp_vs_attn import (
    build_mlp_attn_summary,
    plot_magnitude_comparison,
    plot_detailed_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_per_layer_df(
    num_layers: int = 4,
    include_stable_rank: bool = True,
    include_empirical_rank: bool = False,
    mlp_scale: float = 2.0,
    attn_scale: float = 1.0,
) -> pd.DataFrame:
    """Build a minimal per-layer DataFrame with MLP and Attention rows."""
    rows = []
    for layer in range(num_layers):
        for group, scale in [("mlp", mlp_scale), ("attn", attn_scale)]:
            row = {
                "layer": layer,
                "group": group,
                "dW_fro_layer": (layer + 1) * scale,
                "W_fro_layer": 10.0 + layer,
                "dW_fro_layer_rel": ((layer + 1) * scale) / (10.0 + layer),
                "max_dW_spectral": 0.5 * scale,
                "max_W_spectral": 5.0,
                "max_dW_spectral_rel": 0.1,
                "count_mats": 2,
            }
            if include_stable_rank:
                row["mean_dW_stable_rank"] = 3.0 + layer * 0.5 if group == "mlp" else 2.0 + layer * 0.3
            if include_empirical_rank:
                row["mean_dW_empirical_rank"] = 10.0 + layer if group == "mlp" else 8.0 + layer
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_mlp_attn_summary
# ---------------------------------------------------------------------------

class TestBuildMlpAttnSummary:
    def test_returns_one_row_per_layer(self):
        per_layer_df = _make_per_layer_df(num_layers=5)
        summary = build_mlp_attn_summary(per_layer_df)
        assert len(summary) == 5

    def test_layer_indices_are_correct(self):
        per_layer_df = _make_per_layer_df(num_layers=3)
        summary = build_mlp_attn_summary(per_layer_df)
        layers = [row["layer"] for row in summary]
        assert layers == [0, 1, 2]

    def test_frobenius_values_match_input(self):
        per_layer_df = _make_per_layer_df(num_layers=2, mlp_scale=3.0, attn_scale=1.5)
        summary = build_mlp_attn_summary(per_layer_df)

        # Layer 0: mlp_frobenius = (0+1)*3.0 = 3.0, attn_frobenius = (0+1)*1.5 = 1.5
        assert abs(summary[0]["mlp_frobenius"] - 3.0) < 1e-8
        assert abs(summary[0]["attn_frobenius"] - 1.5) < 1e-8
        # Layer 1: mlp_frobenius = (1+1)*3.0 = 6.0, attn_frobenius = (1+1)*1.5 = 3.0
        assert abs(summary[1]["mlp_frobenius"] - 6.0) < 1e-8
        assert abs(summary[1]["attn_frobenius"] - 3.0) < 1e-8

    def test_ratio_computation(self):
        per_layer_df = _make_per_layer_df(num_layers=1, mlp_scale=4.0, attn_scale=2.0)
        summary = build_mlp_attn_summary(per_layer_df)
        # ratio = 4.0 / (2.0 + 1e-10) ≈ 2.0
        assert abs(summary[0]["ratio_mlp_attn"] - 2.0) < 1e-5

    def test_stable_rank_included_when_present(self):
        per_layer_df = _make_per_layer_df(include_stable_rank=True)
        summary = build_mlp_attn_summary(per_layer_df)
        assert summary[0]["mlp_stable_rank"] is not None
        assert summary[0]["attn_stable_rank"] is not None

    def test_stable_rank_none_when_absent(self):
        per_layer_df = _make_per_layer_df(include_stable_rank=False)
        summary = build_mlp_attn_summary(per_layer_df)
        assert summary[0]["mlp_stable_rank"] is None
        assert summary[0]["attn_stable_rank"] is None

    def test_empty_when_no_matching_groups(self):
        """If the DataFrame only has 'other' group, summary should be empty."""
        df = pd.DataFrame([
            {"layer": 0, "group": "other", "dW_fro_layer": 1.0},
            {"layer": 1, "group": "other", "dW_fro_layer": 2.0},
        ])
        summary = build_mlp_attn_summary(df)
        assert summary == []

    def test_skips_layers_with_only_one_group(self):
        """Layers that have MLP but no Attention (or vice versa) are skipped."""
        df = pd.DataFrame([
            {"layer": 0, "group": "mlp", "dW_fro_layer": 1.0},
            {"layer": 0, "group": "attn", "dW_fro_layer": 2.0},
            {"layer": 1, "group": "mlp", "dW_fro_layer": 3.0},
            # Layer 1 has no attn row
        ])
        summary = build_mlp_attn_summary(df)
        assert len(summary) == 1
        assert summary[0]["layer"] == 0

    def test_ratio_high_when_mlp_dominates(self):
        per_layer_df = _make_per_layer_df(num_layers=1, mlp_scale=10.0, attn_scale=0.1)
        summary = build_mlp_attn_summary(per_layer_df)
        assert summary[0]["ratio_mlp_attn"] > 50  # 10/0.1 = 100

    def test_ratio_low_when_attn_dominates(self):
        per_layer_df = _make_per_layer_df(num_layers=1, mlp_scale=0.1, attn_scale=10.0)
        summary = build_mlp_attn_summary(per_layer_df)
        assert summary[0]["ratio_mlp_attn"] < 0.02  # 0.1/10 = 0.01


# ---------------------------------------------------------------------------
# Plotting smoke tests
# ---------------------------------------------------------------------------

class TestPlotMagnitudeComparison:
    def test_smoke_creates_png(self, temp_dir):
        """Smoke test: verify plot_magnitude_comparison produces a PNG."""
        per_layer_df = _make_per_layer_df(num_layers=4)
        plot_magnitude_comparison(per_layer_df, temp_dir, title="Test")
        assert os.path.isfile(os.path.join(temp_dir, "mlp_vs_attn_magnitude.png"))

    def test_no_crash_with_empty_groups(self, temp_dir):
        """If there are no MLP or Attention rows, the function should not crash."""
        df = pd.DataFrame([
            {"layer": 0, "group": "other", "dW_fro_layer": 1.0},
        ])
        plot_magnitude_comparison(df, temp_dir)
        # No PNG expected since there's nothing to plot
        assert not os.path.isfile(os.path.join(temp_dir, "mlp_vs_attn_magnitude.png"))


class TestPlotDetailedAnalysis:
    def test_smoke_creates_png(self, temp_dir):
        """Smoke test: verify plot_detailed_analysis produces a PNG."""
        per_layer_df = _make_per_layer_df(num_layers=4, include_stable_rank=True)
        plot_detailed_analysis(per_layer_df, temp_dir, title="Test Detailed")
        assert os.path.isfile(os.path.join(temp_dir, "mlp_vs_attn_detailed.png"))

    def test_smoke_with_empirical_rank(self, temp_dir):
        """Smoke test: detailed plot works when empirical rank is included."""
        per_layer_df = _make_per_layer_df(
            num_layers=3, include_stable_rank=True, include_empirical_rank=True,
        )
        plot_detailed_analysis(per_layer_df, temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "mlp_vs_attn_detailed.png"))

    def test_no_crash_without_stable_rank(self, temp_dir):
        """If stable rank column is absent, detailed plot should be skipped gracefully."""
        per_layer_df = _make_per_layer_df(include_stable_rank=False)
        plot_detailed_analysis(per_layer_df, temp_dir)
        # No PNG expected without stable rank data
        assert not os.path.isfile(os.path.join(temp_dir, "mlp_vs_attn_detailed.png"))
