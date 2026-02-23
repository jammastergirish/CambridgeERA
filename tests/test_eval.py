"""Tests for experiment/eval.py — arg parsing, output writing, and high-level summary."""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiment"))

from unittest.mock import patch, MagicMock
import pytest

# Mock lm_eval before importing eval — eval.py has a top-level import
_mock_lm_eval = MagicMock()
_mock_lm_eval_tasks = MagicMock()
sys.modules.setdefault("lm_eval", _mock_lm_eval)
sys.modules.setdefault("lm_eval.tasks", _mock_lm_eval_tasks)

import eval as eval_module
from eval import _write_high_level_summary, DEFAULT_TASKS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def full_results():
    """Simulated lm_eval results dict with all expected tasks."""
    return {
        "mmlu": {
            "acc,none": 0.45,
            "acc_stderr,none": 0.004,
            "alias": "mmlu",
        },
        "mmlu_humanities": {
            "acc,none": 0.42,
            "acc_stderr,none": 0.007,
            "alias": " - humanities",
        },
        "wikitext": {
            "alias": "wikitext",
            "word_perplexity,none": 10.05,
            "word_perplexity_stderr,none": "N/A",
            "byte_perplexity,none": 1.54,
            "byte_perplexity_stderr,none": "N/A",
            "bits_per_byte,none": 0.62,
            "bits_per_byte_stderr,none": "N/A",
        },
        "wmdp_bio_categorized_mcqa": {
            "acc,none": 0.5263,
            "acc_stderr,none": 0.013,
            "alias": "wmdp_bio_categorized_mcqa",
        },
        "wmdp_bio_robust": {
            "acc,none": 0.4309,
            "acc_stderr,none": 0.017,
            "alias": " - wmdp_bio_robust",
        },
        "wmdp_bio_shortcut": {
            "acc,none": 0.7309,
            "acc_stderr,none": 0.022,
            "alias": " - wmdp_bio_shortcut",
        },
        "wmdp_bio_cloze_verified": {
            "alias": "wmdp_bio_cloze_verified",
            "acc_norm,none": 0.3652,
            "acc_norm_stderr,none": 0.015,
        },
        "wmdp_bio_robust_rewritten": {
            "acc,none": 0.2573,
            "acc_stderr,none": 0.009,
            "alias": "wmdp_bio_robust_rewritten",
        },
    }


@pytest.fixture
def minimal_results():
    """Results with only MMLU — tests graceful handling of missing tasks."""
    return {
        "mmlu": {
            "acc,none": 0.45,
            "acc_stderr,none": 0.004,
            "alias": "mmlu",
        },
    }


@pytest.fixture
def mock_lm_eval_results(full_results):
    """Return structure mimicking lm_eval.simple_evaluate output."""
    return {
        "results": full_results,
        "configs": {"mmlu": "some_config"},
    }


def _run_main(argv, mock_evaluate_return):
    """Helper: run eval_module.main() with mocked lm_eval and sys.argv."""
    with patch.object(eval_module.lm_eval, "simple_evaluate", return_value=mock_evaluate_return):
        with patch.object(eval_module, "TaskManager"):
            with patch("sys.argv", ["eval.py"] + argv):
                eval_module.main()


# ---------------------------------------------------------------------------
# _write_high_level_summary
# ---------------------------------------------------------------------------
class TestWriteHighLevelSummary:
    """Test that the markdown summary is correctly generated."""

    def test_creates_file(self, tmp_path, full_results):
        _write_high_level_summary(full_results, "test/model", str(tmp_path))
        assert (tmp_path / "high_level_summary.md").exists()

    def test_contains_model_name_in_header(self, tmp_path, full_results):
        _write_high_level_summary(full_results, "org/my-model", str(tmp_path))
        content = (tmp_path / "high_level_summary.md").read_text()
        assert "org/my-model" in content
        assert "# Eval Summary:" in content

    def test_contains_table_header(self, tmp_path, full_results):
        _write_high_level_summary(full_results, "m", str(tmp_path))
        content = (tmp_path / "high_level_summary.md").read_text()
        assert "| Benchmark | Score |" in content
        assert "|-----------|-------|" in content

    def test_all_rows_present_with_full_results(self, tmp_path, full_results):
        _write_high_level_summary(full_results, "m", str(tmp_path))
        content = (tmp_path / "high_level_summary.md").read_text()
        assert "| MMLU |" in content
        assert "| WikiText (word perplexity) |" in content
        assert "| WMDP Bio (categorized MCQ) |" in content
        assert "| ↳ Robust subset |" in content
        assert "| ↳ Shortcut subset |" in content
        assert "| WMDP Bio (cloze verified) |" in content
        assert "| WMDP Bio (robust rewritten) |" in content

    def test_accuracy_formatted_as_percentage(self, tmp_path, full_results):
        _write_high_level_summary(full_results, "m", str(tmp_path))
        content = (tmp_path / "high_level_summary.md").read_text()
        assert "45.0%" in content       # MMLU acc=0.45
        assert "25.7%" in content       # Robust rewritten acc=0.2573

    def test_perplexity_formatted_as_float(self, tmp_path, full_results):
        _write_high_level_summary(full_results, "m", str(tmp_path))
        content = (tmp_path / "high_level_summary.md").read_text()
        assert "10.05" in content       # word_perplexity
        assert "1005.0%" not in content  # NOT percentage

    def test_missing_tasks_are_skipped(self, tmp_path, minimal_results):
        _write_high_level_summary(minimal_results, "m", str(tmp_path))
        content = (tmp_path / "high_level_summary.md").read_text()
        assert "| MMLU |" in content
        assert "WikiText" not in content
        assert "WMDP" not in content
        assert "Robust subset" not in content

    def test_empty_results(self, tmp_path):
        _write_high_level_summary({}, "m", str(tmp_path))
        content = (tmp_path / "high_level_summary.md").read_text()
        assert "| Benchmark | Score |" in content
        lines = content.strip().split("\n")
        # header line, blank, table header, separator = 4 lines, no data rows
        assert len(lines) == 4

    def test_missing_metric_key_in_task(self, tmp_path):
        """Task present but expected metric key is absent → row skipped."""
        results = {"mmlu": {"some_other_metric,none": 0.99, "alias": "mmlu"}}
        _write_high_level_summary(results, "m", str(tmp_path))
        content = (tmp_path / "high_level_summary.md").read_text()
        assert "| MMLU |" not in content

    def test_file_ends_with_newline(self, tmp_path, full_results):
        _write_high_level_summary(full_results, "m", str(tmp_path))
        content = (tmp_path / "high_level_summary.md").read_text()
        assert content.endswith("\n")

    def test_row_count_matches_available_tasks(self, tmp_path, full_results):
        """All 7 headline rows should be present when all tasks exist."""
        _write_high_level_summary(full_results, "m", str(tmp_path))
        content = (tmp_path / "high_level_summary.md").read_text()
        # Count data rows (lines starting with "| " but not header/separator)
        data_lines = [l for l in content.strip().split("\n")
                      if l.startswith("| ") and "Benchmark" not in l and "---" not in l]
        assert len(data_lines) == 7


# ---------------------------------------------------------------------------
# DEFAULT_TASKS constant
# ---------------------------------------------------------------------------
class TestDefaultTasks:
    def test_default_tasks_is_list(self):
        assert isinstance(DEFAULT_TASKS, list)

    def test_default_tasks_contains_expected_benchmarks(self):
        assert "mmlu" in DEFAULT_TASKS
        assert "wikitext" in DEFAULT_TASKS
        assert "wmdp_bio_robust_rewritten" in DEFAULT_TASKS
        assert "wmdp_bio_cloze_verified" in DEFAULT_TASKS
        assert "wmdp_bio_categorized_mcqa" in DEFAULT_TASKS

    def test_default_tasks_has_five_entries(self):
        assert len(DEFAULT_TASKS) == 5


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
class TestArgParsing:
    """Test argument parsing via main()."""

    def test_model_is_required(self):
        """--model is a required argument → SystemExit(2)."""
        with pytest.raises(SystemExit):
            with patch("sys.argv", ["eval.py"]):
                with patch.object(eval_module.lm_eval, "simple_evaluate"):
                    with patch.object(eval_module, "TaskManager"):
                        eval_module.main()

    def test_custom_outdir_is_respected(self, mock_lm_eval_results, tmp_path):
        """When --outdir is provided, files are written there."""
        outdir = str(tmp_path / "custom_out")
        _run_main(["--model", "org/test", "--outdir", outdir], mock_lm_eval_results)
        assert os.path.isdir(outdir)
        assert os.path.isfile(os.path.join(outdir, "summary.json"))

    def test_outdir_auto_derived_when_not_set(self, mock_lm_eval_results, tmp_path):
        """When --outdir is omitted, it should be auto-derived from --model."""
        with patch.object(eval_module.lm_eval, "simple_evaluate",
                          return_value=mock_lm_eval_results) as mock_eval:
            with patch.object(eval_module, "TaskManager"):
                with patch("sys.argv", ["eval.py", "--model", "org/my-model"]):
                    with patch("os.makedirs"):
                        with patch("builtins.open", MagicMock()):
                            eval_module.main()
        assert mock_eval.called

    def test_custom_tasks_override(self, tmp_path):
        """--tasks should override DEFAULT_TASKS."""
        results = {"results": {"mmlu": {"acc,none": 0.5, "alias": "mmlu"}}, "configs": {}}
        outdir = str(tmp_path / "out")

        with patch.object(eval_module.lm_eval, "simple_evaluate",
                          return_value=results) as mock_eval:
            with patch.object(eval_module, "TaskManager"):
                with patch("sys.argv", ["eval.py", "--model", "m", "--outdir", outdir,
                                        "--tasks", "mmlu"]):
                    eval_module.main()

        assert mock_eval.call_args.kwargs["tasks"] == ["mmlu"]

    def test_default_seed_is_42(self, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        with patch.object(eval_module.lm_eval, "simple_evaluate",
                          return_value=mock_lm_eval_results) as mock_eval:
            with patch.object(eval_module, "TaskManager"):
                with patch("sys.argv", ["eval.py", "--model", "m", "--outdir", outdir]):
                    eval_module.main()
        assert mock_eval.call_args.kwargs["random_seed"] == 42


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
class TestDeviceResolution:
    """Test auto device resolution logic."""

    def test_explicit_device_is_passed_through(self, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        with patch.object(eval_module.lm_eval, "simple_evaluate",
                          return_value=mock_lm_eval_results) as mock_eval:
            with patch.object(eval_module, "TaskManager"):
                with patch("sys.argv", ["eval.py", "--model", "m", "--outdir", outdir,
                                        "--device", "cpu"]):
                    eval_module.main()
        assert mock_eval.call_args.kwargs["device"] == "cpu"


# ---------------------------------------------------------------------------
# Output file writing
# ---------------------------------------------------------------------------
class TestOutputWriting:
    """Test that main() writes all expected output files."""

    def test_writes_summary_json(self, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        _run_main(["--model", "org/test", "--outdir", outdir], mock_lm_eval_results)

        summary_path = os.path.join(outdir, "summary.json")
        assert os.path.isfile(summary_path)
        with open(summary_path) as f:
            data = json.load(f)
        assert data["model"] == "org/test"
        assert "results" in data
        assert "tasks" in data

    def test_writes_per_task_json(self, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        _run_main(["--model", "org/test", "--outdir", outdir], mock_lm_eval_results)

        for task_name in mock_lm_eval_results["results"]:
            task_path = os.path.join(outdir, f"{task_name}.json")
            assert os.path.isfile(task_path), f"Missing {task_name}.json"
            with open(task_path) as f:
                data = json.load(f)
            assert data["model"] == "org/test"
            assert data["task"] == task_name

    def test_writes_high_level_summary_md(self, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        _run_main(["--model", "org/test", "--outdir", outdir], mock_lm_eval_results)

        md_path = os.path.join(outdir, "high_level_summary.md")
        assert os.path.isfile(md_path)
        content = open(md_path).read()
        assert "org/test" in content
        assert "| Benchmark | Score |" in content

    def test_summary_json_contains_all_tasks(self, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        _run_main(["--model", "m", "--outdir", outdir], mock_lm_eval_results)

        with open(os.path.join(outdir, "summary.json")) as f:
            data = json.load(f)
        for task_name in mock_lm_eval_results["results"]:
            assert task_name in data["results"]

    def test_summary_json_configs_serialized(self, mock_lm_eval_results, tmp_path):
        """configs values are converted to strings."""
        outdir = str(tmp_path / "out")
        _run_main(["--model", "m", "--outdir", outdir], mock_lm_eval_results)

        with open(os.path.join(outdir, "summary.json")) as f:
            data = json.load(f)
        assert "configs" in data
        for v in data["configs"].values():
            assert isinstance(v, str)


# ---------------------------------------------------------------------------
# model_args construction
# ---------------------------------------------------------------------------
class TestModelArgs:
    """Test that model_args string is constructed correctly."""

    def test_default_dtype_auto_omits_dtype(self, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        with patch.object(eval_module.lm_eval, "simple_evaluate",
                          return_value=mock_lm_eval_results) as mock_eval:
            with patch.object(eval_module, "TaskManager"):
                with patch("sys.argv", ["eval.py", "--model", "org/test", "--outdir", outdir]):
                    eval_module.main()
        model_args = mock_eval.call_args.kwargs["model_args"]
        assert model_args == "pretrained=org/test"
        assert "dtype" not in model_args

    def test_explicit_dtype_is_included(self, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        with patch.object(eval_module.lm_eval, "simple_evaluate",
                          return_value=mock_lm_eval_results) as mock_eval:
            with patch.object(eval_module, "TaskManager"):
                with patch("sys.argv", ["eval.py", "--model", "org/test", "--outdir", outdir,
                                        "--dtype", "float16"]):
                    eval_module.main()
        model_args = mock_eval.call_args.kwargs["model_args"]
        assert model_args == "pretrained=org/test,dtype=float16"


# ---------------------------------------------------------------------------
# lm_eval.simple_evaluate call
# ---------------------------------------------------------------------------
class TestSimpleEvaluateCall:
    """Verify the arguments passed to lm_eval.simple_evaluate."""

    def test_passes_correct_kwargs(self, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        with patch.object(eval_module.lm_eval, "simple_evaluate",
                          return_value=mock_lm_eval_results) as mock_eval:
            with patch.object(eval_module, "TaskManager"):
                with patch("sys.argv", ["eval.py", "--model", "org/test", "--outdir", outdir,
                                        "--device", "cpu", "--batch-size", "16", "--seed", "7",
                                        "--limit", "100"]):
                    eval_module.main()

        kw = mock_eval.call_args.kwargs
        assert kw["model"] == "hf"
        assert kw["batch_size"] == "16"
        assert kw["device"] == "cpu"
        assert kw["limit"] == 100
        assert kw["random_seed"] == 7
        assert kw["numpy_random_seed"] == 7
        assert kw["torch_random_seed"] == 7

    def test_task_manager_receives_include_path(self, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        with patch.object(eval_module.lm_eval, "simple_evaluate",
                          return_value=mock_lm_eval_results):
            with patch.object(eval_module, "TaskManager") as mock_tm:
                with patch("sys.argv", ["eval.py", "--model", "m", "--outdir", outdir]):
                    eval_module.main()

        mock_tm.assert_called_once()
        assert mock_tm.call_args.kwargs["include_path"].endswith("lm_eval_tasks")


# ---------------------------------------------------------------------------
# W&B Logging
# ---------------------------------------------------------------------------
class TestWandbLogging:
    """Test Weights & Biases logging integration."""

    @patch("eval.wandb")
    def test_wandb_not_called_without_project(self, mock_wandb, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        with patch.object(eval_module.lm_eval, "simple_evaluate", return_value=mock_lm_eval_results):
            with patch.object(eval_module, "TaskManager"):
                with patch("sys.argv", ["eval.py", "--model", "org/test", "--outdir", outdir]):
                    eval_module.main()

        mock_wandb.init.assert_not_called()
        mock_wandb.log.assert_not_called()

    @patch("eval.wandb")
    def test_wandb_called_with_project(self, mock_wandb, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        with patch.object(eval_module.lm_eval, "simple_evaluate", return_value=mock_lm_eval_results):
            with patch.object(eval_module, "TaskManager"):
                with patch("sys.argv", ["eval.py", "--model", "org/test", "--outdir", outdir, 
                                        "--wandb-project", "my-test-proj"]):
                    eval_module.main()

        mock_wandb.init.assert_called_once()
        init_kwargs = mock_wandb.init.call_args.kwargs
        assert init_kwargs["project"] == "my-test-proj"
        # Defaults to model name if run_name not provided
        assert init_kwargs["name"] == "org/test"
        
        mock_wandb.log.assert_called_once()
        log_data = mock_wandb.log.call_args.args[0]
        assert "eval_bench/mmlu/acc" in log_data
        
        mock_wandb.summary.update.assert_called_once_with(log_data)
        mock_wandb.finish.assert_called_once()

    @patch("eval.wandb")
    def test_wandb_uses_custom_name(self, mock_wandb, mock_lm_eval_results, tmp_path):
        outdir = str(tmp_path / "out")
        with patch.object(eval_module.lm_eval, "simple_evaluate", return_value=mock_lm_eval_results):
            with patch.object(eval_module, "TaskManager"):
                with patch("sys.argv", ["eval.py", "--model", "org/test", "--outdir", outdir, 
                                        "--wandb-project", "my-test-proj", "--wandb-name", "custom-run-123"]):
                    eval_module.main()

        init_kwargs = mock_wandb.init.call_args.kwargs
        assert init_kwargs["name"] == "custom-run-123"

    @patch("eval.wandb")
    @patch("builtins.print")
    def test_wandb_import_error_handled_gracefully(self, mock_print, mock_wandb, mock_lm_eval_results, tmp_path):
        # Make importing wandb fail inside the try block
        # We patch sys.modules to trick the import inside main()
        import sys
        with patch.dict("sys.modules", {"wandb": None}):
            outdir = str(tmp_path / "out")
            with patch.object(eval_module.lm_eval, "simple_evaluate", return_value=mock_lm_eval_results):
                with patch.object(eval_module, "TaskManager"):
                    with patch("sys.argv", ["eval.py", "--model", "org/test", "--outdir", outdir, 
                                            "--wandb-project", "my-test-proj"]):
                        eval_module.main()

        # The fallback "Failed to log to W&B" should be printed without crashing
        printed_texts = " ".join([call.args[0] for call in mock_print.call_args_list if call.args])
        assert "WARNING: Failed to log to W&B" in printed_texts
