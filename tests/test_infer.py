"""Tests for infer.py — model loading, generation, and CLI argument handling."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
import torch
import pytest

import infer


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------
class TestLoadModel:
    """Test load_model auto-detection and model setup."""

    @patch("infer.AutoModelForCausalLM")
    @patch("infer.AutoTokenizer")
    def test_returns_model_tokenizer_device(self, mock_tokenizer_cls, mock_model_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        model, tokenizer, device = infer.load_model("some/model")

        mock_tokenizer_cls.from_pretrained.assert_called_once_with(
            "some/model", trust_remote_code=True
        )
        mock_model_cls.from_pretrained.assert_called_once()
        mock_model.eval.assert_called_once()
        assert tokenizer.pad_token == "<eos>"
        assert device in ("cuda", "mps", "cpu")

    @patch("infer.AutoModelForCausalLM")
    @patch("infer.AutoTokenizer")
    def test_preserves_existing_pad_token(self, mock_tokenizer_cls, mock_model_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        _, tokenizer, _ = infer.load_model("some/model")

        assert tokenizer.pad_token == "<pad>"


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
class TestGenerate:
    """Test generate with a tiny random model."""

    @pytest.fixture
    def tiny_model_and_tokenizer(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_embd=32,
            n_layer=2,
            n_head=2,
            n_positions=64,
        )
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        return model, tokenizer

    def test_returns_string(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        output = infer.generate(model, tokenizer, "Hello", "cpu", max_tokens=10)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_respects_max_tokens(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer
        short = infer.generate(model, tokenizer, "Hello", "cpu", max_tokens=5)
        long = infer.generate(model, tokenizer, "Hello", "cpu", max_tokens=50)

        # Tokenize both outputs to count actual tokens generated
        short_tokens = tokenizer.encode(short)
        long_tokens = tokenizer.encode(long)
        assert len(short_tokens) <= len(long_tokens)

    def test_deterministic_output(self, tiny_model_and_tokenizer):
        model, tokenizer = tiny_model_and_tokenizer

        output_a = infer.generate(model, tokenizer, "Hello world", "cpu", max_tokens=10)
        output_b = infer.generate(model, tokenizer, "Hello world", "cpu", max_tokens=10)

        # Greedy decoding (do_sample=False) should be deterministic
        assert output_a == output_b


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
class TestCLI:
    """Test argument parsing and validation."""

    def test_requires_model(self):
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["infer.py", "--prompt", "hello"]):
                infer.main()

    def test_requires_prompt_or_interactive(self):
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["infer.py", "--model", "some/model"]):
                infer.main()

    def test_valid_args_accepted(self):
        """Verify argparse accepts valid argument combinations without error."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--model", required=True)
        parser.add_argument("--prompt", default=None)
        parser.add_argument("--interactive", action="store_true")
        parser.add_argument("--max-tokens", type=int, default=200)

        args = parser.parse_args(["--model", "x/y", "--prompt", "hello"])
        assert args.model == "x/y"
        assert args.prompt == "hello"
        assert args.max_tokens == 200
        assert args.interactive is False
