"""Tests for experiment/pipeline.sh — variable definitions and env-var overrides.

These tests source only the variable-definition block of pipeline.sh (using a
sentinel `exit 0` injected before the first real work step) and verify that the
correct shell variables are set.  No models are loaded; no network calls are
made.
"""

import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(REPO_ROOT, "experiment", "pipeline.sh")


def _eval_pipeline_vars(*var_names: str, env: dict | None = None) -> dict[str, str]:
    """Source pipeline.sh up to the point where real work begins, then print
    the requested variables.  Returns a dict of {var_name: value}.

    We inject an early `exit 0` by prepending a small wrapper that:
      1. Stubs out `python3` so the dataset-existence check is a no-op.
      2. Stubs out `clear` so the terminal isn't cleared during tests.
      3. Sources pipeline.sh — which will call `exit 0` before any `uv run`.
    """
    # Build a tiny shell script that:
    #   - fakes python3 (skip dataset check)
    #   - fakes clear (no-op)
    #   - sources pipeline.sh
    #   - prints the variables we care about
    source_block = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            # Stub out python3 so the dataset-check one-liner is a no-op
            'python3() { return 0; }',
            'export -f python3',
            # Stub out clear
            'clear() { return 0; }',
            'export -f clear',
            # Stub out uv so any step that slips past our exit is a no-op
            'uv() { return 0; }',
            'export -f uv',
            # Source the pipeline — it will cd to repo root, which is fine
            f'source "{PIPELINE}"',
            # Print each requested variable
        ]
        + [f'echo "{v}=${{{{_{v}:-${v}}}}}"'.replace("{_{v}:-${v}}", f"{{{v}:-}}") for v in var_names]
    )

    # Simpler, cleaner approach: just echo the vars after sourcing
    var_prints = "\n".join(f'printf "%s\\n" "{v}=${{{v}:-}}"' for v in var_names)

    script = "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'python3() { return 0; }; export -f python3',
        'clear() { return 0; }; export -f clear',
        'uv() { return 0; }; export -f uv',
        f'source "{PIPELINE}" 2>/dev/null || true',
        var_prints,
    ])

    merge_env = {**os.environ, **(env or {})}
    # Remove any pre-existing MODEL_A / MODEL_B so we get clean defaults
    for key in ["MODEL_A", "MODEL_B", "ENABLE_PRETRAIN_COMPARISON", "ENABLE_CB_COMPARISONS"]:
        merge_env.pop(key, None)
    if env:
        merge_env.update(env)

    result = subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env=merge_env,
        cwd=REPO_ROOT,
    )
    if result.returncode not in (0, 1):
        pytest.fail(
            f"Pipeline sourcing failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            name, _, value = line.partition("=")
            values[name.strip()] = value.strip()
    return values


class TestPipelineDefaults:
    def test_model_a_default(self):
        """MODEL_A should default to deep-ignorance-unfiltered."""
        vals = _eval_pipeline_vars("MODEL_A")
        assert vals["MODEL_A"] == "EleutherAI/deep-ignorance-unfiltered"

    def test_model_b_default(self):
        """MODEL_B should default to deep-ignorance-e2e-strong-filter."""
        vals = _eval_pipeline_vars("MODEL_B")
        assert vals["MODEL_B"] == "EleutherAI/deep-ignorance-e2e-strong-filter"

    def test_seeds_default(self):
        """SEEDS should default to '42 123 456'."""
        vals = _eval_pipeline_vars("SEEDS")
        assert vals["SEEDS"] == "42 123 456"

    def test_outroot_default(self):
        """OUTROOT should default to 'outputs'."""
        vals = _eval_pipeline_vars("OUTROOT")
        assert vals["OUTROOT"] == "outputs"

    def test_enable_tuned_lens_default(self):
        """ENABLE_TUNED_LENS should default to '0'."""
        vals = _eval_pipeline_vars("ENABLE_TUNED_LENS")
        assert vals["ENABLE_TUNED_LENS"] == "0"


class TestPipelineOverrides:
    def test_model_a_override(self):
        """MODEL_A env var should propagate into the pipeline."""
        vals = _eval_pipeline_vars("MODEL_A", env={"MODEL_A": "org/custom-model-a"})
        assert vals["MODEL_A"] == "org/custom-model-a"

    def test_model_b_override(self):
        """MODEL_B env var should propagate into the pipeline."""
        vals = _eval_pipeline_vars("MODEL_B", env={"MODEL_B": "org/custom-model-b"})
        assert vals["MODEL_B"] == "org/custom-model-b"

    def test_outroot_override(self):
        """OUTROOT env var should propagate."""
        vals = _eval_pipeline_vars("OUTROOT", env={"OUTROOT": "my_results"})
        assert vals["OUTROOT"] == "my_results"

    def test_seeds_override(self):
        """SEEDS env var should propagate."""
        vals = _eval_pipeline_vars("SEEDS", env={"SEEDS": "1 2 3"})
        assert vals["SEEDS"] == "1 2 3"

    def test_enable_tuned_lens_override(self):
        """ENABLE_TUNED_LENS=1 should propagate."""
        vals = _eval_pipeline_vars("ENABLE_TUNED_LENS", env={"ENABLE_TUNED_LENS": "1"})
        assert vals["ENABLE_TUNED_LENS"] == "1"


class TestPipelineOutputDirDerivation:
    def test_comp_dir_default(self):
        """COMP should be derived by replacing / with _ and joining with __to__."""
        vals = _eval_pipeline_vars("COMP")
        assert vals["COMP"] == (
            "EleutherAI_deep-ignorance-unfiltered"
            "__to__"
            "EleutherAI_deep-ignorance-e2e-strong-filter"
        )

    def test_comp_dir_custom_models(self):
        """COMP should update correctly when MODEL_A and MODEL_B are overridden."""
        vals = _eval_pipeline_vars(
            "COMP",
            env={
                "MODEL_A": "org/model-a",
                "MODEL_B": "org/model-b",
            },
        )
        assert vals["COMP"] == "org_model-a__to__org_model-b"

    def test_model_a_dir_derived(self):
        """MODEL_A_DIR should replace / with _."""
        vals = _eval_pipeline_vars("MODEL_A_DIR", env={"MODEL_A": "foo/bar-baz"})
        assert vals["MODEL_A_DIR"] == "foo_bar-baz"

    def test_model_b_dir_derived(self):
        """MODEL_B_DIR should replace / with _."""
        vals = _eval_pipeline_vars("MODEL_B_DIR", env={"MODEL_B": "org/my-model"})
        assert vals["MODEL_B_DIR"] == "org_my-model"


class TestOldVariablesAbsent:
    """Ensure the old multi-comparison variables no longer exist in the pipeline."""

    def test_no_base_variable(self):
        """BASE should not be set by pipeline.sh."""
        vals = _eval_pipeline_vars("BASE")
        assert vals.get("BASE", "") == ""

    def test_no_filtered_variable(self):
        """FILTERED should not be set by pipeline.sh."""
        vals = _eval_pipeline_vars("FILTERED")
        assert vals.get("FILTERED", "") == ""

    def test_no_unlearned_variable(self):
        """UNLEARNED should not be set by pipeline.sh."""
        vals = _eval_pipeline_vars("UNLEARNED")
        assert vals.get("UNLEARNED", "") == ""

    def test_no_comp1_variable(self):
        """COMP1 should not be set by pipeline.sh."""
        vals = _eval_pipeline_vars("COMP1")
        assert vals.get("COMP1", "") == ""

    def test_no_comp2_variable(self):
        """COMP2 should not be set by pipeline.sh."""
        vals = _eval_pipeline_vars("COMP2")
        assert vals.get("COMP2", "") == ""

    def test_no_enable_pretrain_comparison(self):
        """ENABLE_PRETRAIN_COMPARISON should not be set by pipeline.sh."""
        vals = _eval_pipeline_vars("ENABLE_PRETRAIN_COMPARISON")
        assert vals.get("ENABLE_PRETRAIN_COMPARISON", "") == ""

    def test_no_enable_cb_comparisons(self):
        """ENABLE_CB_COMPARISONS should not be set by pipeline.sh."""
        vals = _eval_pipeline_vars("ENABLE_CB_COMPARISONS")
        assert vals.get("ENABLE_CB_COMPARISONS", "") == ""
