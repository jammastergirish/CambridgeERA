"""Tests for experiment/eval_mmlu.py — data loading and scoring logic."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

from unittest.mock import patch, MagicMock
import numpy as np
import torch
import pytest


# ---------------------------------------------------------------------------
# load_mmlu
# ---------------------------------------------------------------------------
class TestLoadMmlu:
    """Test load_mmlu with a mocked HuggingFace dataset."""

    def _make_fake_dataset(self, num_items=20):
        """Create a list of dicts that mimics what load_dataset returns."""
        subjects = ["math", "history", "biology", "physics"]
        items = []
        for i in range(num_items):
            items.append({
                "question": f"Question {i}?",
                "choices": [f"A{i}", f"B{i}", f"C{i}", f"D{i}"],
                "answer": i % 4,
                "subject": subjects[i % len(subjects)],
            })
        return items

    @patch("eval_mmlu.load_dataset")
    def test_returns_correct_structure(self, mock_load_dataset):
        from eval_mmlu import load_mmlu
        mock_load_dataset.return_value = self._make_fake_dataset(10)

        items = load_mmlu(max_samples=None, seed=42)

        assert len(items) == 10
        for item in items:
            assert "question" in item
            assert "choices" in item
            assert "answer" in item
            assert "subject" in item
            assert isinstance(item["answer"], int)
            assert len(item["choices"]) == 4

    @patch("eval_mmlu.load_dataset")
    def test_subsampling(self, mock_load_dataset):
        from eval_mmlu import load_mmlu
        mock_load_dataset.return_value = self._make_fake_dataset(50)

        items = load_mmlu(max_samples=10, seed=42)

        assert len(items) == 10

    @patch("eval_mmlu.load_dataset")
    def test_shuffling_is_deterministic(self, mock_load_dataset):
        from eval_mmlu import load_mmlu
        mock_load_dataset.return_value = self._make_fake_dataset(20)

        items_a = load_mmlu(max_samples=5, seed=42)
        mock_load_dataset.return_value = self._make_fake_dataset(20)
        items_b = load_mmlu(max_samples=5, seed=42)

        # Same seed should produce same order
        assert [item["question"] for item in items_a] == [item["question"] for item in items_b]

    @patch("eval_mmlu.load_dataset")
    def test_different_seeds_give_different_order(self, mock_load_dataset):
        from eval_mmlu import load_mmlu
        mock_load_dataset.return_value = self._make_fake_dataset(20)

        items_a = load_mmlu(max_samples=10, seed=42)
        mock_load_dataset.return_value = self._make_fake_dataset(20)
        items_b = load_mmlu(max_samples=10, seed=99)

        # Different seeds should (almost certainly) produce different order
        questions_a = [item["question"] for item in items_a]
        questions_b = [item["question"] for item in items_b]
        assert questions_a != questions_b

    @patch("eval_mmlu.load_dataset")
    def test_skips_items_without_question(self, mock_load_dataset):
        from eval_mmlu import load_mmlu
        mock_load_dataset.return_value = [
            {"question": "", "choices": ["A", "B"], "answer": 0, "subject": "x"},
            {"question": "Real question?", "choices": ["A", "B"], "answer": 0, "subject": "x"},
        ]

        items = load_mmlu()
        assert len(items) == 1
        assert items[0]["question"] == "Real question?"


# ---------------------------------------------------------------------------
# score_multiple_choice
# ---------------------------------------------------------------------------
class TestScoreMultipleChoice:
    """Test score_multiple_choice with a tiny model.

    We don't test accuracy (random weights), just that the function
    runs correctly and returns the right structure.
    """

    @pytest.fixture
    def tiny_model_and_tokenizer(self):
        """Create a minimal GPT-2 model for testing."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Must use the real vocab size to avoid index-out-of-range errors
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

    def test_returns_correct_structure(self, tiny_model_and_tokenizer, sample_mmlu_items):
        from eval_mmlu import score_multiple_choice

        model, tokenizer = tiny_model_and_tokenizer
        results = score_multiple_choice(model, tokenizer, sample_mmlu_items, "cpu", max_length=64)

        assert len(results) == len(sample_mmlu_items)
        for result in results:
            assert "subject" in result
            assert "correct" in result
            assert "predicted" in result
            assert "answer" in result
            assert isinstance(result["correct"], bool)
            assert result["predicted"] in [0, 1, 2, 3]
            assert result["answer"] in [0, 1, 2, 3]

    def test_deterministic_with_same_seed(self, tiny_model_and_tokenizer, sample_mmlu_items):
        from eval_mmlu import score_multiple_choice

        model, tokenizer = tiny_model_and_tokenizer

        torch.manual_seed(0)
        results_a = score_multiple_choice(model, tokenizer, sample_mmlu_items[:2], "cpu", max_length=64)

        torch.manual_seed(0)
        results_b = score_multiple_choice(model, tokenizer, sample_mmlu_items[:2], "cpu", max_length=64)

        for result_a, result_b in zip(results_a, results_b):
            assert result_a["predicted"] == result_b["predicted"]
