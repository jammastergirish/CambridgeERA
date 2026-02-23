"""Tests for unlearn/analysis/analyze_runs.py."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unlearn", "analysis"))

from unittest.mock import patch, MagicMock
import pytest
import pandas as pd

import analyze_runs as ar


class MockSummary:
    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


class MockConfig:
    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


class MockRun:
    def __init__(self, id, name, state, config_data, summary_data):
        self.id = id
        self.name = name
        self.state = state
        self.config = MockConfig(config_data)
        self.summary = MockSummary(summary_data)


@pytest.fixture
def mock_api():
    api = MagicMock()
    
    # Run 1: Base model (no __ep)
    run1 = MockRun(
        id="base1",
        name="EleutherAI/deep-ignorance-unfiltered",
        state="finished",
        config_data={"hyperparameters": {"method": "unknown"}},
        summary_data={
            "eval_bench/mmlu/acc": 0.45,
            "eval_bench/wmdp_bio_robust_rewritten/acc": 0.25,
            "eval_bench/wmdp_bio_cloze_verified/acc": 0.36,
            "eval_bench/wmdp_bio_categorized_mcqa/acc": 0.52,
            "train/loss": None
        }
    )
    
    # Run 2: Unlearning method (has __ep)
    run2 = MockRun(
        id="run2",
        name="model__cb_lat__ep1",
        state="finished",
        config_data={"hyperparameters": {"method": "cb_lat"}},
        summary_data={
            "eval_bench/mmlu/acc": 0.44,
            "eval_bench/wmdp_bio_robust_rewritten/acc": 0.15,
            "eval_bench/wmdp_bio_cloze_verified/acc": 0.25,
            "eval_bench/wmdp_bio_categorized_mcqa/acc": 0.35,
            "train/loss": 2.5
        }
    )
    
    # Run 3: Incomplete run (missing evals)
    run3 = MockRun(
        id="run3",
        name="model__cb_lat__ep2",
        state="finished",
        config_data={"hyperparameters": {"method": "cb_lat"}},
        summary_data={
            "train/loss": 1.5
        }
    )
    
    # Run 4: Crashed run
    run4 = MockRun(
        id="run4",
        name="model__dpo__ep1",
        state="crashed",
        config_data={"hyperparameters": {"method": "dpo"}},
        summary_data={}
    )
    
    api.runs.return_value = [run1, run2, run3, run4]
    return api


class TestAnalyzeRuns:
    @patch('analyze_runs.wandb')
    @patch('analyze_runs.load_dotenv')
    @patch('builtins.print')
    def test_main_fetches_runs(self, mock_print, mock_load, mock_wandb, mock_api):
        mock_wandb.Api.return_value = mock_api
        ar.main()
        
        mock_api.runs.assert_called_once_with("cambridge_era")
        
        # Check printed output includes our baselines and methods
        printed_text = " ".join([call_args[0][0] for call_args in mock_print.call_args_list if call_args[0]])
        
        assert "Successfully processed 2 runs" in printed_text
        assert "BASELINES" in printed_text
        assert "EleutherAI/deep-ignorance-unfiltered" in printed_text
        assert "Best Models By Method" in printed_text
        assert "cb_lat" in printed_text
        
    @patch('analyze_runs.wandb')
    @patch('analyze_runs.load_dotenv')
    @patch('builtins.print')
    def test_main_handles_api_error(self, mock_print, mock_load, mock_wandb):
        mock_api = MagicMock()
        mock_api.runs.side_effect = Exception("API fetch failed")
        mock_wandb.Api.return_value = mock_api
        
        ar.main()
        
        printed_text = " ".join([call_args[0][0] for call_args in mock_print.call_args_list if call_args[0]])
        assert "Error fetching runs: API fetch failed" in printed_text
        
    @patch('analyze_runs.wandb')
    @patch('analyze_runs.load_dotenv')
    @patch('builtins.print')
    def test_main_no_baselines_instructions(self, mock_print, mock_load, mock_wandb):
        mock_api = MagicMock()
        
        # Only provide an unlearning run, no baseline
        run = MockRun(
            id="run1",
            name="model__cb_lat__ep1",
            state="finished",
            config_data={"hyperparameters": {"method": "cb_lat"}},
            summary_data={"eval_bench/mmlu/acc": 0.44}
        )
        mock_api.runs.return_value = [run]
        mock_wandb.Api.return_value = mock_api
        
        ar.main()
        
        printed_text = " ".join([call_args[0][0] for call_args in mock_print.call_args_list if call_args[0]])
        assert "No baseline runs found" in printed_text
        assert "uv run experiment/eval.py --model EleutherAI/deep-ignorance-unfiltered" in printed_text
