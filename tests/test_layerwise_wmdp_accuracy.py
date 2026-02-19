"""Tests for experiment/layerwise_wmdp_accuracy.py — scoring and plotting."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import numpy as np
import pytest

from layerwise_wmdp_accuracy import plot_wmdp_lens_results


# ---------------------------------------------------------------------------
# plot_wmdp_lens_results (smoke tests)
# ---------------------------------------------------------------------------

def _make_dummy_lens_results(num_layers=4):
    results = []
    for layer in range(num_layers):
        results.append({
            "layer": layer,
            "accuracy": 0.25 + layer * 0.1,
            "correct": 10 + layer * 5,
            "total": 50,
        })
    return results


class TestPlotWmdpLensResults:
    def test_smoke_creates_png(self, temp_dir):
        results = _make_dummy_lens_results()
        plot_wmdp_lens_results(results, 0.55, "logit", temp_dir, title="Test")
        assert os.path.isfile(os.path.join(temp_dir, "wmdp_lens_analysis.png"))

    def test_smoke_without_title(self, temp_dir):
        results = _make_dummy_lens_results(3)
        plot_wmdp_lens_results(results, 0.4, "tuned", temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "wmdp_lens_analysis.png"))

    def test_single_layer(self, temp_dir):
        results = _make_dummy_lens_results(1)
        plot_wmdp_lens_results(results, 0.3, "logit", temp_dir)
        assert os.path.isfile(os.path.join(temp_dir, "wmdp_lens_analysis.png"))
