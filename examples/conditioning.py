"""Run: python examples/conditioning.py  (works without installing the package)

Closes the demo's loop: teaches the mushroom body to associate two stimuli with reward
vs punishment and measures, with a reversal control, whether the memory is real."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flybrain.conditioning import run

if __name__ == "__main__":
    run()
