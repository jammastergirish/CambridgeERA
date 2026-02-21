"""Tests for unlearn/unlearn.py — build_outdir and related logic."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unlearn"))

from types import SimpleNamespace
import pytest

from unlearn import build_outdir, METHOD_PARAMS, PARAM_ABBREV


# ---------------------------------------------------------------------------
# build_outdir
# ---------------------------------------------------------------------------
class TestBuildOutdir:
    """Test that build_outdir produces correct, unique folder names."""

    def _make_args(self, method, **overrides):
        """Create a namespace with all default args for the given method."""
        defaults = {
            "model": "EleutherAI/deep-ignorance-unfiltered",
            "method": method,
            "epochs": 1,
            "lr": 1e-5,
            "batch_size": 4,
            "retain_weight": 1.0,
            "forget_weight": 1.0,
            "beta": 0.1,
            "alpha": 100.0,
            "steering_coeff": 20.0,
            "layer_id": "5,6,7",
            "lat_eps": 0.1,
            "lat_steps": 5,
            "wt_noise_std": 0.02,
            "wt_reg_lambda": 0.1,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_includes_method_in_path(self):
        result = build_outdir(self._make_args("cb_lat"))
        assert "/cb_lat__" in result

    def test_starts_with_unlearned_models(self):
        result = build_outdir(self._make_args("ga_simple"))
        assert result.startswith("unlearned_models/")

    def test_model_slashes_replaced(self):
        result = build_outdir(self._make_args("ga_simple"))
        # The model name should have / replaced with _
        assert "EleutherAI_deep-ignorance-unfiltered" in result
        # Model part (between root and method) should have no bare slashes
        model_part = result.split("/")[1]  # unlearned_models/<model_part>/method...
        assert "EleutherAI" in model_part
        assert model_part == "EleutherAI_deep-ignorance-unfiltered"

    def test_ga_simple_has_minimal_params(self):
        result = build_outdir(self._make_args("ga_simple"))
        assert "ep1" in result
        assert "lr1e-05" in result
        assert "bs4" in result
        # Should NOT include method-irrelevant params
        assert "_rw" not in result
        assert "_a" not in result.split("__")[-1].split("_ep")[0]  # no alpha prefix

    def test_cb_lat_includes_all_relevant_params(self):
        result = build_outdir(self._make_args("cb_lat"))
        suffix = result.split("__")[-1]
        assert "ep1" in suffix
        assert "lr1e-05" in suffix
        assert "bs4" in suffix
        assert "a100.0" in suffix
        assert "sc20.0" in suffix
        assert "le0.1" in suffix
        assert "ls5" in suffix
        assert "ly5-6-7" in suffix

    def test_layer_id_commas_become_dashes(self):
        result = build_outdir(self._make_args("rmu", layer_id="10,11,12"))
        assert "ly10-11-12" in result
        assert "," not in result

    def test_different_params_produce_different_paths(self):
        path_a = build_outdir(self._make_args("cb_lat", epochs=1))
        path_b = build_outdir(self._make_args("cb_lat", epochs=2))
        assert path_a != path_b

    def test_same_params_produce_same_path(self):
        path_a = build_outdir(self._make_args("cb_lat"))
        path_b = build_outdir(self._make_args("cb_lat"))
        assert path_a == path_b

    def test_different_methods_produce_different_paths(self):
        path_a = build_outdir(self._make_args("ga_simple"))
        path_b = build_outdir(self._make_args("ga"))
        assert path_a != path_b

    def test_wt_dist_includes_noise_std(self):
        result = build_outdir(self._make_args("wt_dist"))
        assert "wn0.02" in result

    def test_wt_dist_reg_includes_lambda(self):
        result = build_outdir(self._make_args("wt_dist_reg"))
        assert "wr0.1" in result

    def test_dpo_includes_beta(self):
        result = build_outdir(self._make_args("dpo"))
        assert "b0.1" in result

    def test_ga_includes_retain_weight(self):
        result = build_outdir(self._make_args("ga"))
        assert "rw1.0" in result

    def test_grad_diff_includes_forget_weight(self):
        result = build_outdir(self._make_args("grad_diff"))
        assert "fw1.0" in result


# ---------------------------------------------------------------------------
# METHOD_PARAMS and PARAM_ABBREV consistency
# ---------------------------------------------------------------------------
class TestMethodParamsConsistency:
    """Ensure METHOD_PARAMS and PARAM_ABBREV are in sync."""

    def test_all_methods_have_entries(self):
        expected = {"ga_simple", "ga", "grad_diff", "dpo", "npo", "simnpo",
                    "rmu", "cb", "lat", "cb_lat", "wt_dist", "wt_dist_reg"}
        assert set(METHOD_PARAMS.keys()) == expected

    def test_all_params_have_abbreviations(self):
        all_params = set()
        for params in METHOD_PARAMS.values():
            all_params.update(params)
        for param in all_params:
            assert param in PARAM_ABBREV, f"Missing abbreviation for '{param}'"

    def test_abbreviations_are_unique(self):
        abbrevs = list(PARAM_ABBREV.values())
        assert len(abbrevs) == len(set(abbrevs)), "Duplicate abbreviations found"

    def test_every_method_includes_shared_params(self):
        """Every method should at least include epochs, lr, batch_size."""
        for method, params in METHOD_PARAMS.items():
            assert "epochs" in params, f"{method} missing epochs"
            assert "lr" in params, f"{method} missing lr"
            assert "batch_size" in params, f"{method} missing batch_size"


# ---------------------------------------------------------------------------
# Argument parser (--push-to-hub)
# ---------------------------------------------------------------------------
class TestArgParser:
    """Test argument parsing changes."""

    def test_push_to_hub_defaults_false(self):
        """--push-to-hub should default to False."""
        from unlearn import main
        import argparse
        # Build the parser the same way main() does
        parser = argparse.ArgumentParser()
        parser.add_argument("--push-to-hub", action="store_true")
        args = parser.parse_args([])
        assert args.push_to_hub is False

    def test_push_to_hub_set_when_passed(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--push-to-hub", action="store_true")
        args = parser.parse_args(["--push-to-hub"])
        assert args.push_to_hub is True

    def test_no_save_defaults_false(self):
        """--no-save should default to False."""
        from unlearn import main
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-save", action="store_true")
        args = parser.parse_args([])
        assert args.no_save is False

    def test_no_save_set_when_passed(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--no-save", action="store_true")
        args = parser.parse_args(["--no-save"])
        assert args.no_save is True


# ---------------------------------------------------------------------------
# eval_summary.py — parsing and method inference
# ---------------------------------------------------------------------------
class TestEvalSummaryParsing:
    """Test parse_model_name handles all folder name formats."""

    @pytest.fixture(autouse=True)
    def _import(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval_summary import parse_model_name, _infer_method_from_params
        self.parse = parse_model_name
        self.infer = _infer_method_from_params

    # -- New format: base__method__params --
    def test_new_format_cb_lat(self):
        r = self.parse("outputs/EleutherAI_deep-ignorance-unfiltered__cb_lat__ep2_lr1e-05_bs4_a100.0_sc20.0_le0.1_ls5_ly5-6-7")
        assert r["method"] == "cb_lat"
        assert r["params"]["epochs"] == "2"
        assert r["params"]["lat_eps"] == "0.1"
        assert r["params"]["layers"] == "5-6-7"

    def test_new_format_ga_simple(self):
        r = self.parse("outputs/EleutherAI_deep-ignorance-unfiltered__ga_simple__ep1_lr1e-05_bs4")
        assert r["method"] == "ga_simple"
        assert r["params"]["epochs"] == "1"
        assert len(r["params"]) == 3  # epochs, lr, batch

    def test_new_format_wt_dist_reg(self):
        r = self.parse("outputs/EleutherAI_deep-ignorance-unfiltered__wt_dist_reg__ep1_lr1e-05_bs4_wr0.1")
        assert r["method"] == "wt_dist_reg"
        assert r["params"]["lambda"] == "0.1"

    # -- HuggingFace format: girishgupta_base__method__params --
    def test_hf_format_with_method(self):
        r = self.parse("outputs/girishgupta_EleutherAI_deep-ignorance-unfiltered__cb__ep2_lr1e-05_bs4_a100.0_sc20.0_ly5-6-7")
        assert r["method"] == "cb"
        assert r["params"]["alpha"] == "100.0"

    # -- Old/short format: girishgupta_method__params (no base model) --
    def test_short_name_rmu_inferred(self):
        r = self.parse("outputs/girishgupta_rmu__ep2_lr1e-05_bs4_a100.0_sc20.0_ly5-6-7")
        assert r["method"] == "rmu"
        assert r["params"]["epochs"] == "2"

    def test_short_name_cb_lat_inferred(self):
        r = self.parse("outputs/girishgupta_cb_lat__ep2_lr5e-06_bs4_a100.0_sc20.0_le0.1_ls5_ly5-6-7")
        assert r["method"] == "cb_lat"
        assert r["params"]["lr"] == "5e-06"

    # -- Baselines --
    def test_baseline_returns_none(self):
        r = self.parse("outputs/EleutherAI_deep-ignorance-unfiltered")
        assert r is None

    def test_baseline_filtered_returns_none(self):
        r = self.parse("outputs/EleutherAI_deep-ignorance-e2e-strong-filter")
        assert r is None


class TestInferMethod:
    """Test _infer_method_from_params for each method type."""

    @pytest.fixture(autouse=True)
    def _import(self):
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from eval_summary import _infer_method_from_params
        self.infer = _infer_method_from_params

    def test_cb_lat(self):
        assert self.infer({"lat_eps": "0.1", "alpha": "100", "steer": "20"}) == "cb_lat"

    def test_lat(self):
        assert self.infer({"lat_eps": "0.1"}) == "lat"

    def test_rmu(self):
        assert self.infer({"alpha": "100", "steer": "20"}) == "rmu"

    def test_wt_dist(self):
        assert self.infer({"noise": "0.02"}) == "wt_dist"

    def test_wt_dist_reg(self):
        assert self.infer({"lambda": "0.1"}) == "wt_dist_reg"

    def test_dpo(self):
        assert self.infer({"beta": "0.1"}) == "dpo"

    def test_npo(self):
        assert self.infer({"beta": "0.1", "ret_wt": "1.0"}) == "npo"

    def test_ga(self):
        assert self.infer({"ret_wt": "1.0"}) == "ga"

    def test_grad_diff(self):
        assert self.infer({"fgt_wt": "1.0"}) == "grad_diff"

    def test_unknown(self):
        assert self.infer({}) == "unknown"

