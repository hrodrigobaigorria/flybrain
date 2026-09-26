"""flybrain — a wiring-constrained fruit-fly connectome (MaleCNS v1.0) as a reusable
decision engine. You feed it an RGB frame and an optional reward/aversive signal; it
runs 166,700 spiking neurons and returns a motor readout plus its memory state.

Quick start:
    from flybrain import load
    brain = load()                 # builds from the local dataset
    result = brain.observe(rgb, "reward")   # rgb: (180, 320, 3) uint8
    print(result["turn"], result["memory"]["changed_edges"])
"""

from .config import Settings
from .controller import FlyController
from .data import verify

FlyBrain = FlyController  # friendly alias; same object


def load(settings=None):
    """Build the brain from the local dataset. Returns a FlyController."""
    return FlyController(settings or Settings())


__all__ = ["Settings", "FlyController", "FlyBrain", "load", "verify"]
