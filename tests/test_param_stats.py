"""Tests for experiment/param_stats.py — SmartLoader and stats computation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

import torch
import numpy as np
import pytest

from utils import stable_rank_and_spectral, extract_layer, classify_granular


# ---------------------------------------------------------------------------
# SmartLoader — single-file safetensors
# ---------------------------------------------------------------------------
class TestSmartLoaderSingleFile:
    def test_loads_all_param_names(self, safetensors_single_model):
        from param_stats import SmartLoader

        dir_a, _ = safetensors_single_model
        loader = SmartLoader(dir_a)
        names = loader.get_all_param_names()

        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "model.layers.0.mlp.gate_proj.weight" in names
        assert "model.embed_tokens.weight" in names
        assert len(names) == 5

    def test_get_param_returns_correct_tensor(self, safetensors_single_model, sample_weights):
        from param_stats import SmartLoader

        weight_a, _ = sample_weights
        dir_a, _ = safetensors_single_model
        loader = SmartLoader(dir_a)

        name = "model.layers.0.self_attn.q_proj.weight"
        tensor = loader.get_param(name, "cpu", torch.float32)

        assert tensor is not None
        assert tensor.shape == weight_a[name].shape
        assert torch.allclose(tensor, weight_a[name], atol=1e-6)

    def test_get_param_nonexistent_returns_none(self, safetensors_single_model):
        from param_stats import SmartLoader

        dir_a, _ = safetensors_single_model
        loader = SmartLoader(dir_a)

        tensor = loader.get_param("nonexistent.weight", "cpu", torch.float32)
        assert tensor is None

    def test_is_safetensors_flag(self, safetensors_single_model):
        from param_stats import SmartLoader

        dir_a, _ = safetensors_single_model
        loader = SmartLoader(dir_a)
        assert loader.is_safetensors is True


# ---------------------------------------------------------------------------
# SmartLoader — sharded safetensors
# ---------------------------------------------------------------------------
class TestSmartLoaderSharded:
    def test_loads_all_param_names(self, safetensors_sharded_model):
        from param_stats import SmartLoader

        loader = SmartLoader(safetensors_sharded_model)
        names = loader.get_all_param_names()

        assert len(names) == 5
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert "model.layers.1.self_attn.q_proj.weight" in names

    def test_loads_params_across_shards(self, safetensors_sharded_model, sample_weights):
        from param_stats import SmartLoader

        weight_a, _ = sample_weights
        loader = SmartLoader(safetensors_sharded_model)

        # Load a param from each shard and verify correctness
        for name, expected_tensor in weight_a.items():
            tensor = loader.get_param(name, "cpu", torch.float32)
            assert tensor is not None, f"Failed to load {name}"
            assert torch.allclose(tensor, expected_tensor, atol=1e-6), f"Mismatch for {name}"


# ---------------------------------------------------------------------------
# SmartLoader — error handling
# ---------------------------------------------------------------------------
class TestSmartLoaderErrors:
    def test_nonexistent_path_raises(self, temp_dir):
        from param_stats import SmartLoader

        fake_path = os.path.join(temp_dir, "does_not_exist")
        os.makedirs(fake_path)  # exists but has no weights
        with pytest.raises(FileNotFoundError):
            SmartLoader(fake_path)


# ---------------------------------------------------------------------------
# Stats computation — verify math on known tensors
# ---------------------------------------------------------------------------
class TestStatsComputation:
    """Verify the core stats computation that happens in the main loop."""

    def test_frobenius_norm_of_difference(self, safetensors_single_model, sample_weights):
        from param_stats import SmartLoader

        weight_a, weight_b = sample_weights
        dir_a, dir_b = safetensors_single_model
        loader_a = SmartLoader(dir_a)
        loader_b = SmartLoader(dir_b)

        name = "model.layers.0.self_attn.q_proj.weight"
        Wa = loader_a.get_param(name, "cpu", torch.float32)
        Wb = loader_b.get_param(name, "cpu", torch.float32)
        dW = Wb - Wa

        # The perturbation was 0.01 * randn for an 8x8 matrix
        # Frobenius norm should be small (roughly 0.01 * sqrt(64) ≈ 0.08)
        dW_fro = float(dW.float().norm().item())
        assert 0 < dW_fro < 0.5  # sanity bound

        # Relative norm should also be small
        W_fro = float(Wa.float().norm().item())
        assert W_fro > 0
        assert dW_fro / W_fro < 0.2

    def test_stable_rank_of_identity_perturbation(self, safetensors_single_model):
        from param_stats import SmartLoader

        dir_a, dir_b = safetensors_single_model
        loader_a = SmartLoader(dir_a)
        loader_b = SmartLoader(dir_b)

        # The q_proj weight in model A is identity (8x8)
        name = "model.layers.0.self_attn.q_proj.weight"
        Wa = loader_a.get_param(name, "cpu", torch.float32)
        Wb = loader_b.get_param(name, "cpu", torch.float32)
        dW = Wb - Wa

        # Stable rank of the identity matrix should be close to its dimension
        sr_W, spec_W = stable_rank_and_spectral(Wa, use_svd=True)
        assert abs(sr_W - 8.0) < 0.5  # identity 8x8 -> stable rank = 8

        # dW is small random perturbation — stable rank should be reasonable
        sr_dW, spec_dW = stable_rank_and_spectral(dW, use_svd=True)
        assert sr_dW > 0
        assert spec_dW > 0

    def test_utility_functions_on_param_names(self):
        """Verify extract_layer and classify_granular work for the param names we use."""
        assert extract_layer("model.layers.0.self_attn.q_proj.weight") == 0
        assert extract_layer("model.layers.1.mlp.down_proj.weight") == 1
        assert extract_layer("model.embed_tokens.weight") is None

        assert classify_granular("model.layers.0.self_attn.q_proj.weight") == "qkv"
        assert classify_granular("model.layers.0.mlp.gate_proj.weight") == "mlp_expand"
        assert classify_granular("model.layers.1.mlp.down_proj.weight") == "mlp_contract"
        assert classify_granular("model.embed_tokens.weight") == "other"
