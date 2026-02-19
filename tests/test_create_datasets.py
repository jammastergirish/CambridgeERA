"""Tests for create_datasets.py — forget-set and retain-set creation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock

from create_datasets import create_forget_set, create_retain_set


# ---------------------------------------------------------------------------
# Mock datasets
# ---------------------------------------------------------------------------
def _make_wmdp_dataset(n=25):
    """Return a list of dicts mimicking the WMDP-Bio HF dataset."""
    return [{"question": f"Question about bio topic {i}?", "choices": ["A", "B", "C", "D"], "answer": 0} for i in range(n)]


def _make_wikitext_dataset(n=30, short_count=5):
    """Return a list of dicts mimicking WikiText-2.

    First `short_count` entries are too short (< 50 chars) to pass the filter.
    """
    items = []
    for i in range(short_count):
        items.append({"text": f"Short {i}"})  # will be filtered out
    for i in range(n - short_count):
        items.append({"text": f"This is a sufficiently long passage of text from Wikipedia article number {i} about an interesting topic."})
    return items


# ---------------------------------------------------------------------------
# create_forget_set
# ---------------------------------------------------------------------------
class TestCreateForgetSet:
    @patch("create_datasets.load_dataset")
    def test_writes_correct_number_of_lines(self, mock_load, temp_dir):
        mock_load.return_value = _make_wmdp_dataset(25)
        outpath = os.path.join(temp_dir, "forget.txt")

        count = create_forget_set(outpath, num_samples=10)

        assert count == 10
        with open(outpath) as f:
            lines = f.readlines()
        assert len(lines) == 10

    @patch("create_datasets.load_dataset")
    def test_writes_all_when_fewer_than_limit(self, mock_load, temp_dir):
        mock_load.return_value = _make_wmdp_dataset(5)
        outpath = os.path.join(temp_dir, "forget.txt")

        count = create_forget_set(outpath, num_samples=100)

        assert count == 5
        with open(outpath) as f:
            lines = f.readlines()
        assert len(lines) == 5

    @patch("create_datasets.load_dataset")
    def test_strips_newlines_from_questions(self, mock_load, temp_dir):
        mock_load.return_value = [{"question": "Line1\nLine2\nLine3"}]
        outpath = os.path.join(temp_dir, "forget.txt")

        create_forget_set(outpath)

        with open(outpath) as f:
            content = f.read()
        assert "\n" == content[-1]  # ends with newline
        # The written line should have internal newlines replaced with spaces
        assert content.strip() == "Line1 Line2 Line3"

    @patch("create_datasets.load_dataset")
    def test_uses_prompt_key_when_no_question(self, mock_load, temp_dir):
        """Some WMDP versions use 'prompt' instead of 'question'."""
        mock_load.return_value = [{"prompt": "What is X?"}]
        outpath = os.path.join(temp_dir, "forget.txt")

        count = create_forget_set(outpath)

        assert count == 1
        with open(outpath) as f:
            assert f.read().strip() == "What is X?"

    @patch("create_datasets.load_dataset")
    def test_skips_entries_without_question_or_prompt(self, mock_load, temp_dir):
        mock_load.return_value = [
            {"question": "Valid question?"},
            {"other_field": "no question here"},
            {"question": "Another valid one?"},
        ]
        outpath = os.path.join(temp_dir, "forget.txt")

        count = create_forget_set(outpath)

        assert count == 2

    @patch("create_datasets.load_dataset")
    def test_creates_parent_directories(self, mock_load, temp_dir):
        mock_load.return_value = _make_wmdp_dataset(3)
        outpath = os.path.join(temp_dir, "nested", "dir", "forget.txt")

        create_forget_set(outpath, num_samples=3)

        assert os.path.isfile(outpath)

    @patch("create_datasets.load_dataset")
    def test_calls_load_dataset_correctly(self, mock_load, temp_dir):
        mock_load.return_value = _make_wmdp_dataset(1)
        outpath = os.path.join(temp_dir, "forget.txt")

        create_forget_set(outpath)

        mock_load.assert_called_once_with("cais/wmdp", "wmdp-bio", split="test")

    @patch("create_datasets.load_dataset")
    def test_empty_dataset(self, mock_load, temp_dir):
        mock_load.return_value = []
        outpath = os.path.join(temp_dir, "forget.txt")

        count = create_forget_set(outpath)

        assert count == 0
        with open(outpath) as f:
            assert f.read() == ""


# ---------------------------------------------------------------------------
# create_retain_set
# ---------------------------------------------------------------------------
class TestCreateRetainSet:
    @patch("create_datasets.load_dataset")
    def test_writes_correct_number_of_lines(self, mock_load, temp_dir):
        mock_load.return_value = _make_wikitext_dataset(30, short_count=5)
        outpath = os.path.join(temp_dir, "retain.txt")

        count = create_retain_set(outpath, num_samples=10)

        assert count == 10
        with open(outpath) as f:
            lines = f.readlines()
        assert len(lines) == 10

    @patch("create_datasets.load_dataset")
    def test_filters_short_texts(self, mock_load, temp_dir):
        mock_load.return_value = _make_wikitext_dataset(10, short_count=5)
        outpath = os.path.join(temp_dir, "retain.txt")

        count = create_retain_set(outpath, num_samples=100)

        # Only 5 of 10 entries pass the length filter
        assert count == 5

    @patch("create_datasets.load_dataset")
    def test_custom_min_length(self, mock_load, temp_dir):
        mock_load.return_value = [
            {"text": "Short"},        # 5 chars
            {"text": "A" * 20},       # 20 chars
            {"text": "B" * 100},      # 100 chars
        ]
        outpath = os.path.join(temp_dir, "retain.txt")

        # Only text > 30 chars should pass
        count = create_retain_set(outpath, min_length=30)

        assert count == 1

    @patch("create_datasets.load_dataset")
    def test_strips_newlines(self, mock_load, temp_dir):
        long_text = "A" * 60 + "\n" + "B" * 60
        mock_load.return_value = [{"text": long_text}]
        outpath = os.path.join(temp_dir, "retain.txt")

        create_retain_set(outpath, min_length=10)

        with open(outpath) as f:
            content = f.read().strip()
        assert "\n" not in content  # internal newlines replaced
        assert "A" * 60 in content
        assert "B" * 60 in content

    @patch("create_datasets.load_dataset")
    def test_calls_load_dataset_correctly(self, mock_load, temp_dir):
        mock_load.return_value = _make_wikitext_dataset(3, short_count=0)
        outpath = os.path.join(temp_dir, "retain.txt")

        create_retain_set(outpath)

        mock_load.assert_called_once_with("wikitext", "wikitext-2-raw-v1", split="train")

    @patch("create_datasets.load_dataset")
    def test_creates_parent_directories(self, mock_load, temp_dir):
        mock_load.return_value = _make_wikitext_dataset(3, short_count=0)
        outpath = os.path.join(temp_dir, "deep", "path", "retain.txt")

        create_retain_set(outpath, num_samples=3)

        assert os.path.isfile(outpath)

    @patch("create_datasets.load_dataset")
    def test_empty_after_filtering(self, mock_load, temp_dir):
        """All entries are too short — should write 0 lines."""
        mock_load.return_value = [{"text": "tiny"}, {"text": "also tiny"}]
        outpath = os.path.join(temp_dir, "retain.txt")

        count = create_retain_set(outpath)

        assert count == 0
        with open(outpath) as f:
            assert f.read() == ""

    @patch("create_datasets.load_dataset")
    def test_whitespace_only_text_filtered(self, mock_load, temp_dir):
        """Text that is only whitespace should become empty after strip() and be filtered."""
        mock_load.return_value = [{"text": "   \n\n   "}]
        outpath = os.path.join(temp_dir, "retain.txt")

        count = create_retain_set(outpath)

        assert count == 0
