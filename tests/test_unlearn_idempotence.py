"""Tests for the unlearn.py idempotency logic."""

import os
import sys
import tempfile
import importlib.util
from unittest.mock import MagicMock, patch

import pytest

# Load unlearn.py dynamically since it shares a name with its parent folder
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
unlearn_path = os.path.join(repo_root, "unlearn", "unlearn.py")

spec = importlib.util.spec_from_file_location("unlearn_script", unlearn_path)
unlearn_script = importlib.util.module_from_spec(spec)
# Add sys.modules entry so any internal identical imports resolve to this
sys.modules["unlearn_script"] = unlearn_script 
spec.loader.exec_module(unlearn_script)


@patch("wandb.Api")
def test_check_wandb_only_already_finished(mock_wandb_api):
    """Test that --check-wandb-only exits 0 when a run is finished."""
    # Setup the mock API to return a run with state='finished'
    mock_api_instance = MagicMock()
    mock_wandb_api.return_value = mock_api_instance
    
    mock_run = MagicMock()
    mock_run.state = "finished"
    mock_api_instance.runs.return_value = [mock_run]

    # Create dummy data files so the parser defaults don't crash
    with tempfile.TemporaryDirectory() as tmpdir:
        forget_path = os.path.join(tmpdir, "forget.txt")
        retain_path = os.path.join(tmpdir, "retain.txt")
        with open(forget_path, "w") as f:
            f.write("test forget data\n")
        with open(retain_path, "w") as f:
            f.write("test retain data\n")

        with patch("sys.argv", [
            "unlearn.py", 
            "--model", "EleutherAI/deep-ignorance-unfiltered",
            "--method", "cb",
            "--forget-data", forget_path,
            "--retain-data", retain_path,
            "--check-wandb-only",
        ]):
            # It should call sys.exit(0)
            with pytest.raises(SystemExit) as exc:
                unlearn_script.main()
            assert exc.value.code == 0

        # Verify the API was actually queried with the right project AND exact display_name 
        mock_api_instance.runs.assert_called_once()
        args, kwargs = mock_api_instance.runs.call_args
        assert "cambridge_era" in args[0] # The default project name
        
        # This is CRITICAL: ensure the organization prefix is kept
        # Instead of just "cb__ep1_..."" it should be "EleutherAI_deep-ignorance-unfiltered/cb_..."
        assert "filters" in kwargs
        assert "display_name" in kwargs["filters"]
        requested_name = kwargs["filters"]["display_name"]
        
        # It must include the model name as the prefix, just like W&B does natively
        assert requested_name.startswith("EleutherAI_deep-ignorance-unfiltered/cb")


@patch("wandb.Api")
def test_check_wandb_only_not_finished(mock_wandb_api):
    """Test that --check-wandb-only exits 1 when NO run is finished."""
    # Setup the mock API to return a run with state='crashed'
    mock_api_instance = MagicMock()
    mock_wandb_api.return_value = mock_api_instance
    
    mock_run = MagicMock()
    mock_run.state = "crashed"
    mock_api_instance.runs.return_value = [mock_run]

    # Create dummy data files
    with tempfile.TemporaryDirectory() as tmpdir:
        forget_path = os.path.join(tmpdir, "forget.txt")
        retain_path = os.path.join(tmpdir, "retain.txt")
        with open(forget_path, "w") as f:
            f.write("test forget data\n")
        with open(retain_path, "w") as f:
            f.write("test retain data\n")

        with patch("sys.argv", [
            "unlearn.py", 
            "--model", "EleutherAI/deep-ignorance-unfiltered",
            "--method", "cb",
            "--forget-data", forget_path,
            "--retain-data", retain_path,
            "--check-wandb-only",
        ]):
            # It should call sys.exit(1) because it's not finished
            with pytest.raises(SystemExit) as exc:
                unlearn_script.main()
            assert exc.value.code == 1

        args, kwargs = mock_api_instance.runs.call_args
        assert kwargs["filters"]["display_name"].startswith("EleutherAI_deep-ignorance-unfiltered/cb")


@patch("wandb.Api")
def test_check_wandb_only_api_error(mock_wandb_api):
    """Test that --check-wandb-only defaults to 1 (run training) if API fails."""
    # Setup the mock API to raise an exception 
    mock_wandb_api.side_effect = Exception("Network offline")

    # Create dummy data files
    with tempfile.TemporaryDirectory() as tmpdir:
        forget_path = os.path.join(tmpdir, "forget.txt")
        retain_path = os.path.join(tmpdir, "retain.txt")
        with open(forget_path, "w") as f:
            f.write("test forget data\n")
        with open(retain_path, "w") as f:
            f.write("test retain data\n")

        with patch("sys.argv", [
            "unlearn.py", 
            "--model", "EleutherAI/deep-ignorance-unfiltered",
            "--method", "cb",
            "--forget-data", forget_path,
            "--retain-data", retain_path,
            "--check-wandb-only",
        ]):
            # It should swallow the error and call sys.exit(1) so training proceeds
            with pytest.raises(SystemExit) as exc:
                unlearn_script.main()
            assert exc.value.code == 1
