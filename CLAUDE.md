 name: Use uv, never pip
 description: Always use uv (not pip) for package management and running commands
 type: feedback
 ---

 Never use `pip install` — this project uses `uv` for all Python package management.

 **Why:** User corrected me for trying `pip3 install pytest` instead of using uv.

 **How to apply:** Use `uv run --with <deps> <command>` for ad-hoc runs without a pyproject.toml. Never fall back to pip/pip3.

Write DRY code with function/variable names that make sense (i.e., not single letters, etc.)