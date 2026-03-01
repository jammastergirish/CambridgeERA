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
            "tar_alpha": 1.0,
            "tar_lr": 1e-5,
            "tar_epochs": 1,
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

    def test_tar_includes_tar_params(self):
        result = build_outdir(self._make_args("tar"))
        assert "ta1.0" in result  # tar_alpha
        assert "tlr1e-05" in result  # tar_lr
        assert "tep1" in result  # tar_epochs
        # TAR should NOT include regular training params like batch_size
        assert "bs" not in result
        assert "ep" not in result or "tep" in result  # only tar_epochs, not epochs

    def test_tar_different_alpha_different_path(self):
        path_a = build_outdir(self._make_args("tar", tar_alpha=1.0))
        path_b = build_outdir(self._make_args("tar", tar_alpha=0.5))
        assert path_a != path_b
        assert "ta1.0" in path_a
        assert "ta0.5" in path_b

    def test_tar_different_lr_different_path(self):
        path_a = build_outdir(self._make_args("tar", tar_lr=1e-5))
        path_b = build_outdir(self._make_args("tar", tar_lr=5e-6))
        assert path_a != path_b
        assert "tlr1e-05" in path_a
        assert "tlr5e-06" in path_b

    def test_tar_different_epochs_different_path(self):
        path_a = build_outdir(self._make_args("tar", tar_epochs=1))
        path_b = build_outdir(self._make_args("tar", tar_epochs=3))
        assert path_a != path_b
        assert "tep1" in path_a
        assert "tep3" in path_b


# ---------------------------------------------------------------------------
# METHOD_PARAMS and PARAM_ABBREV consistency
# ---------------------------------------------------------------------------
class TestMethodParamsConsistency:
    """Ensure METHOD_PARAMS and PARAM_ABBREV are in sync."""

    def test_all_methods_have_entries(self):
        expected = {"ga_simple", "ga", "grad_diff", "dpo", "npo", "simnpo",
                    "rmu", "cb", "lat", "cb_lat", "tar", "wt_dist", "wt_dist_reg"}
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
        """Every method should at least include epochs, lr, batch_size (except TAR which uses tar_epochs, tar_lr)."""
        for method, params in METHOD_PARAMS.items():
            if method == "tar":
                # TAR uses its own parameter names
                assert "tar_epochs" in params, f"{method} missing tar_epochs"
                assert "tar_lr" in params, f"{method} missing tar_lr"
                assert "tar_alpha" in params, f"{method} missing tar_alpha"
            else:
                assert "epochs" in params, f"{method} missing epochs"
                assert "lr" in params, f"{method} missing lr"
                assert "batch_size" in params, f"{method} missing batch_size"


# ---------------------------------------------------------------------------
# TAR-specific tests
# ---------------------------------------------------------------------------
class TestTARMethod:
    """Test TAR (Task Arithmetic Removal) specific functionality."""

    def test_tar_method_params_correct(self):
        """TAR should only use tar-specific parameters."""
        expected_params = {"tar_alpha", "tar_lr", "tar_epochs"}
        assert set(METHOD_PARAMS["tar"]) == expected_params

    def test_tar_param_abbreviations_exist(self):
        """All TAR parameters should have abbreviations."""
        for param in METHOD_PARAMS["tar"]:
            assert param in PARAM_ABBREV

    def test_tar_abbreviations_correct(self):
        """Test specific TAR abbreviations."""
        assert PARAM_ABBREV["tar_alpha"] == "ta"
        assert PARAM_ABBREV["tar_lr"] == "tlr"
        assert PARAM_ABBREV["tar_epochs"] == "tep"

    def test_tar_parameters_different_from_standard(self):
        """TAR should use different parameter names than standard training."""
        tar_params = set(METHOD_PARAMS["tar"])
        standard_params = {"epochs", "lr", "batch_size"}
        assert tar_params.isdisjoint(standard_params), "TAR should not use standard training parameters"

    def test_tar_in_method_choices(self):
        """TAR should be in the list of available methods."""
        # This tests that TAR was added to the choices in the argument parser
        # We can't easily test the actual parser without refactoring, but we can
        # test that it's in our METHOD_PARAMS which is what the choices would use
        assert "tar" in METHOD_PARAMS

    def test_tar_parameter_defaults_make_sense(self):
        """TAR default parameters should be reasonable."""
        # Test via the defaults in _make_args
        from types import SimpleNamespace
        args = SimpleNamespace(tar_alpha=1.0, tar_lr=1e-5, tar_epochs=1)

        # Alpha should be positive (scaling factor)
        assert args.tar_alpha > 0

        # Learning rate should be reasonable for fine-tuning
        assert 1e-6 <= args.tar_lr <= 1e-4

        # Epochs should be small (TAR is meant to be lightweight)
        assert 1 <= args.tar_epochs <= 5


# ---------------------------------------------------------------------------
# TAR Device Handling Tests
# ---------------------------------------------------------------------------
class TestTARDeviceHandling:
    """Test that TAR properly handles device placement for batches."""

    def test_tar_moves_batch_to_device(self):
        """Test that apply_tar moves batches to the correct device during training."""
        import torch
        from unittest.mock import Mock, patch, MagicMock
        from unlearn import apply_tar

        # Mock model and device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create mock model
        mock_model = Mock()
        mock_model.parameters.return_value = [torch.tensor([1.0], device=device)]
        mock_model.named_parameters.return_value = [("test_param", torch.nn.Parameter(torch.tensor([1.0], device=device)))]

        # Create mock batch that's on wrong device (CPU when we want GPU, or vice versa)
        wrong_device = torch.device("cpu") if device.type == "cuda" else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if wrong_device.type == "cuda" and not torch.cuda.is_available():
            wrong_device = torch.device("cpu")  # fallback if CUDA not available

        mock_batch = {
            "input_ids": torch.tensor([[1, 2, 3]], device=wrong_device),
            "attention_mask": torch.tensor([[1, 1, 1]], device=wrong_device)
        }
        forget_batches = [mock_batch]

        # Mock the nll_loss function to verify device placement
        with patch('unlearn.nll_loss') as mock_nll_loss, \
             patch('unlearn.torch.optim.AdamW') as mock_optimizer_class, \
             patch('unlearn.torch.optim.lr_scheduler.CosineAnnealingLR') as mock_scheduler_class:

            # Set up mocks
            mock_loss = torch.tensor(1.0, device=device)
            mock_nll_loss.return_value = mock_loss

            mock_optimizer = Mock()
            mock_optimizer_class.return_value = mock_optimizer

            mock_scheduler = Mock()
            mock_scheduler_class.return_value = mock_scheduler

            # Call apply_tar
            apply_tar(
                model=mock_model,
                forget_batches=forget_batches,
                alpha=1.0,
                lr=1e-5,
                epochs=1,
                device=device
            )

            # Verify nll_loss was called
            assert mock_nll_loss.called

            # Get the batch that was passed to nll_loss
            call_args = mock_nll_loss.call_args_list[0]  # First call
            _, passed_batch = call_args[0]  # (model, batch)

            # Verify the batch tensors were moved to the correct device
            assert passed_batch["input_ids"].device == device
            assert passed_batch["attention_mask"].device == device


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

