"""Run: python examples/stimulus_demo.py  (works without installing the package)."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flybrain.demo import run

if __name__ == "__main__":
    run()
