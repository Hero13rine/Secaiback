"""Deployment-facing CLI entrypoint for SecAI evaluation."""
import os
import sys

# Ensure project root is importable when packaged or run externally.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from eva_start import main


if __name__ == "__main__":
    main()