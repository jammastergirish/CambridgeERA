"""Tests for experiment/linear_probe_analysis.py — probe training and plotting."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import numpy as np
import pytest

from linear_probe_analysis import train_probe, plot_probe_results


# ---------------------------------------------------------------------------
# train_probe
# ---------------------------------------------------------------------------

class TestTrainProbe:
    def test_returns_expected_keys(self):
        np.random.seed(42)
        forget = np.random.randn(30, 8).astype(np.float32)
        retain = np.random.randn(30, 8).astype(np.float32)
        result = train_probe(forget, retain)
        assert set(result.keys()) == {
            "test_accuracy", "train_accuracy", "selectivity", "auc", "majority_baseline",
        }

    def test_majority_baseline_balanced(self):
        np.random.seed(42)
        forget = np.random.randn(20, 4).astype(np.float32)
        retain = np.random.randn(20, 4).astype(np.float32)
        result = train_probe(forget, retain)
        assert abs(result["majority_baseline"] - 0.5) < 0.01

    def test_majority_baseline_imbalanced(self):
        np.random.seed(42)
        forget = np.random.randn(10, 4).astype(np.float32)
        retain = np.random.randn(40, 4).astype(np.float32)
        result = train_probe(forget, retain)
        assert result["majority_baseline"] == round(40 / 50, 4)

    def test_well_separated_clusters_high_accuracy(self):
        np.random.seed(42)
        forget = np.random.randn(50, 4).astype(np.float32) + 10.0
        retain = np.random.randn(50, 4).astype(np.float32) - 10.0
        result = train_probe(forget, retain)
        assert result["test_accuracy"] > 0.9
        assert result["auc"] > 0.9

    def test_random_features_near_baseline(self):
        np.random.seed(42)
        forget = np.random.randn(50, 4).astype(np.float32)
        retain = np.random.randn(50, 4).astype(np.float32)
        result = train_probe(forget, retain)
        # Selectivity should be near zero for random features
        assert abs(result["selectivity"]) < 0.3

    def test_selectivity_is_accuracy_minus_baseline(self):
        np.random.seed(42)
        forget = np.random.randn(30, 4).astype(np.float32) + 5.0
        retain = np.random.randn(30, 4).astype(np.float32) - 5.0
        result = train_probe(forget, retain)
        expected_selectivity = result["test_accuracy"] - result["majority_baseline"]
        assert abs(result["selectivity"] - round(expected_selectivity, 4)) < 0.001

    def test_different_seeds_give_different_splits(self):
        np.random.seed(42)
        forget = np.random.randn(40, 8).astype(np.float32) + 3.0
        retain = np.random.randn(40, 8).astype(np.float32) - 3.0
        r1 = train_probe(forget, retain, seed=1, layer_index=0)
        r2 = train_probe(forget, retain, seed=2, layer_index=0)
        # May or may not differ, but should not crash
        assert "test_accuracy" in r1 and "test_accuracy" in r2

    def test_auc_is_bounded(self):
        np.random.seed(42)
        forget = np.random.randn(30, 4).astype(np.float32)
        retain = np.random.randn(30, 4).astype(np.float32)
        result = train_probe(forget, retain)
        assert 0.0 <= result["auc"] <= 1.0

    def test_custom_regularisation(self):
        np.random.seed(42)
        forget = np.random.randn(30, 4).astype(np.float32) + 3.0
        retain = np.random.randn(30, 4).astype(np.float32) - 3.0
        r_strong = train_probe(forget, retain, regularisation_strength=100.0)
        r_weak = train_probe(forget, retain, regularisation_strength=0.001)
        # Both should return valid results
        assert r_strong["test_accuracy"] >= 0.0
        assert r_weak["test_accuracy"] >= 0.0


# ---------------------------------------------------------------------------
# plot_probe_results (smoke tests)
# ---------------------------------------------------------------------------

def _make_dummy_probe_results(num_layers=4):
    results = []
    for layer in range(num_layers):
        results.append({
            "layer": layer,
            "test_accuracy": 0.5 + layer * 0.1,
            "train_accuracy": 0.6 + layer * 0.08,
            "selectivity": 0.0 + layer * 0.1,
            "auc": 0.5 + layer * 0.1,
            "majority_baseline": 0.5,
        })
    return results


class TestPlotProbeResults:
    def test_smoke_creates_png(self, temp_dir):
        results = _make_dummy_probe_results()
        plot_probe_results(results, temp_dir, title="Test")
        assert os.path.isfile(os.path.join(temp_dir, "linear_probe_analysis.png"))

    def test_smoke_without_title(self, temp_dir):
        results = _make_dummy_probe_results(3)
        plot_probe_results(results, temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "linear_probe_analysis.png"))

    def test_single_layer(self, temp_dir):
        results = _make_dummy_probe_results(1)
        plot_probe_results(results, temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "linear_probe_analysis.png"))
