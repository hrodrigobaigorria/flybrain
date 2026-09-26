"""A non-trading demonstration: watch the brain react to a visual stimulus and form a
memory. This exercises the whole pipeline (light -> spikes -> motor readout -> memory)
without any market, broker or trading concept."""

import numpy as np

from .config import Settings
from .controller import FlyController


def half_bright_frame(side):
    """A 320x180 RGB frame with one half bright and the other dark."""
    f = np.zeros((180, 320, 3), np.uint8)
    if side == "left":
        f[:, :160] = 255
    else:
        f[:, 160:] = 255
    return f


def run(observations=4):
    brain = FlyController(Settings())
    print(
        f"Brain loaded: {brain.brain.n} neurons, "
        f"{len(brain.brain.circuit['edges'])} plastic KC->MBON edges\n"
    )

    for side in ("left", "right"):
        frame = half_bright_frame(side)
        result = None
        for _ in range(observations):
            result = brain.observe(frame, "none")  # "none" = no reward/punishment
        print(f"Stimulus: bright on the {side.upper()}")
        print(
            f"  DNp20  left {result['left_hz']:.2f} Hz | right {result['right_hz']:.2f} Hz"
            f" | diff {result['difference_hz']:+.2f} Hz  ->  readout: {result['turn']}"
        )
        print(
            f"  total spikes {result['total_spikes']}, KC spikes {result['KC_spikes']}\n"
        )

    reward = brain.observe(half_bright_frame("left"), "reward")
    print(f"Reward pulse delivered ({reward['stimulus_ms']:.0f} ms of dopamine):")
    print(
        f"  reward-cell spikes {reward['reward_spikes']}, "
        f"memory edges changed {reward['memory']['changed_edges']}\n"
    )

    print("This shows the mechanism works: input reaches the network, the motor cells")
    print("produce a readout, and pairing a stimulus with dopamine changes memory synapses.")
    print("The connectome is untuned: the readout is NOT guaranteed to track the light.")


if __name__ == "__main__":
    run()
