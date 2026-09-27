import os

import numpy as np
import pandas as pd
import pytest

from flybrain.config import Settings
from flybrain.controller import Decoder
from flybrain.rule import advance


def test_readout_decoder():
    a = pd.DataFrame(
        {"type": ["DNp20", "DNp20", "DNpe017"], "somaSide": ["L", "R", "L"]}
    )
    d = Decoder(np.array([1, 2, 3]), a, 2)
    assert d.decode(np.array([0, 10, 0]), 0.5)["turn"] == "straight"  # gate closed
    assert d.decode(np.array([0, 10, 1]), 0.5)["turn"] == "right"  # right > left
    assert d.decode(np.array([10, 0, 1]), 0.5)["turn"] == "left"
    assert d.decode(np.array([4, 4, 1]), 0.5)["turn"] == "straight"  # below threshold


def trace_protocol(order, frozen=False):
    k = np.zeros(2)
    d = np.zeros(1)
    u = np.zeros(2)
    w = np.zeros(2)
    gain = np.ones((1, 2))
    for phase in order:
        for _ in range(20):
            kh = np.array([20.0, 0.0]) if phase == "cue" else np.zeros(2)
            dh = np.array([30.0]) if phase == "reinforce" else np.zeros(1)
            advance(k, d, u, w, kh, dh, gain, 0.01, 0.001, frozen=frozen)
    return w


def test_memory_rule_temporal_specificity():
    paired = trace_protocol(["cue", "reinforce"])
    reverse = trace_protocol(["reinforce", "cue"])
    assert paired[0] < 0 and reverse[0] > 0
    assert paired[1] == 0 and reverse[1] == 0  # Unactivated input is unchanged.
    assert np.array_equal(trace_protocol(["cue", "reinforce"], True), np.zeros(2))


@pytest.mark.skipif(
    os.environ.get("FLYBRAIN_FULL_TEST") != "1",
    reason="Loads the full 166k-neuron graph; slow integration test",
)
def test_full_graph_stimulus_and_memory(tmp_path):
    from flybrain.controller import FlyController
    from flybrain.data import verify

    assert verify()["neurons"] == 166700
    c = FlyController(Settings())
    assert len(c.brain.post) == 25582938 and len(c.brain.circuit["edges"]) == 7835
    assert len(c.brain.retina) == 3335 and len(c.brain.r8) == 811
    white = np.full((180, 320, 3), 255, np.uint8)
    for _ in range(3):
        c.observe(white, "none")
    c.save(tmp_path / "before.npz")
    before = c.brain.weight[c.brain.circuit["edges"]].copy()
    reward = c.observe(white, "reward")
    assert reward["reward_spikes"] > 0 and reward["stimulus_ms"] == 200
    assert reward["KC_spikes"] > 0 and reward["memory"]["changed_edges"] > 0
    # A frozen brain must not change its memory synapses under the same stimulation.
    c.restore(tmp_path / "before.npz")
    c.brain.weights_frozen = True
    c.observe(white, "reward")
    assert np.array_equal(before, c.brain.weight[c.brain.circuit["edges"]])
    assert np.isfinite(c.brain.weight).all()


@pytest.mark.skipif(
    os.environ.get("FLYBRAIN_FULL_TEST") != "1",
    reason="Loads the full 166k-neuron graph; slow integration test",
)
def test_conditioning_forms_associative_memory():
    from flybrain.conditioning import run

    # Reversal control: pairing a stimulus with a dopamine population must depress ITS
    # matching MBON compartment more than pairing it with the other population does.
    effect = run(epochs=3, verbose=False)["pairing_effect"]
    for name in ("A", "B"):
        assert effect[name]["reward_pairing_on_reward_edges"] > 1e-3
        assert effect[name]["aversive_pairing_on_aversive_edges"] > 1e-3
