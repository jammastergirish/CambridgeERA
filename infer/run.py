# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch",
#     "transformers",
#     "accelerate",
#     "streamlit",
# ]
# ///
"""
Wrapper to run the Streamlit app with proper command.

Usage:
  uv run infer/run.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()