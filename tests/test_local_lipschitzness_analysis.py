"""Tests for experiment/local_lipschitzness_analysis.py — summary builder and plotting."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import numpy as np
import pytest

from local_lipschitzness_analysis import build_summary_rows, plot_lipschitzness_analysis


# ---------------------------------------------------------------------------
# build_summary_rows
# ---------------------------------------------------------------------------

class TestBuildSummaryRows:
    def test_returns_four_rows(self):
        dummy = [1.0, 2.0, 3.0]
        rows = build_summary_rows(dummy, dummy, dummy, dummy, dummy, dummy,
                                   dummy, dummy, dummy, dummy, dummy, dummy)
        assert len(rows) == 4

    def test_row_keys(self):
        dummy = [1.0, 2.0]
        rows = build_summary_rows(dummy, dummy, dummy, dummy, dummy, dummy,
                                   dummy, dummy, dummy, dummy, dummy, dummy)
        expected_keys = {
            "model", "data", "avg_lipschitz", "std_lipschitz",
            "avg_gradient_norm", "std_gradient_norm",
            "avg_output_variance", "std_output_variance",
        }
        for row in rows:
            assert set(row.keys()) == expected_keys

    def test_model_labels(self):
        dummy = [1.0]
        rows = build_summary_rows(dummy, dummy, dummy, dummy, dummy, dummy,
                                   dummy, dummy, dummy, dummy, dummy, dummy)
        models = [r["model"] for r in rows]
        assert models == ["A", "A", "B", "B"]

    def test_data_labels(self):
        dummy = [1.0]
        rows = build_summary_rows(dummy, dummy, dummy, dummy, dummy, dummy,
                                   dummy, dummy, dummy, dummy, dummy, dummy)
        data = [r["data"] for r in rows]
        assert data == ["forget", "retain", "forget", "retain"]

    def test_avg_lipschitz_value(self):
        forget_a = [2.0, 4.0]
        retain_a = [1.0, 3.0]
        rows = build_summary_rows(
            forget_a, retain_a, [], [], [], [],
            [], [], [], [], [], [],
        )
        assert rows[0]["avg_lipschitz"] == pytest.approx(3.0)
        assert rows[1]["avg_lipschitz"] == pytest.approx(2.0)

    def test_empty_lists(self):
        rows = build_summary_rows([], [], [], [], [], [],
                                   [], [], [], [], [], [])
        for row in rows:
            assert row["avg_lipschitz"] == 0.0


# ---------------------------------------------------------------------------
# plot_lipschitzness_analysis (smoke tests)
# ---------------------------------------------------------------------------

def _dummy_list(n=10, base=1.0, scale=0.1):
    np.random.seed(42)
    return list(base + np.random.randn(n) * scale)


class TestPlotLipschitznessAnalysis:
    def test_smoke_creates_png(self, temp_dir):
        plot_lipschitzness_analysis(
            _dummy_list(), _dummy_list(), _dummy_list(), _dummy_list(),
            _dummy_list(), _dummy_list(), _dummy_list(), _dummy_list(),
            _dummy_list(), _dummy_list(), _dummy_list(), _dummy_list(),
            temp_dir, title="Test",
        )
        assert os.path.isfile(os.path.join(temp_dir, "lipschitzness_analysis.png"))

    def test_smoke_without_title(self, temp_dir):
        plot_lipschitzness_analysis(
            _dummy_list(5), _dummy_list(5), _dummy_list(5), _dummy_list(5),
            _dummy_list(5), _dummy_list(5), _dummy_list(5), _dummy_list(5),
            _dummy_list(5), _dummy_list(5), _dummy_list(5), _dummy_list(5),
            temp_dir,
        )
        assert os.path.isfile(os.path.join(temp_dir, "lipschitzness_analysis.png"))
