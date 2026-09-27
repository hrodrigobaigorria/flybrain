"""Associative conditioning assay: does the KC->MBON plasticity form a *usable* memory?

The demo shows synapses moving; it never shows the memory being useful. This closes that
loop with differential conditioning and reports, honestly, whether the committed
anti-Hebbian rule forms a memory that is *associative* -- routed to the compartment of the
dopamine cell it was paired with -- or whether it is just undifferentiated depression.

Honest premise, measured on this graph: the reconstructed visual pathway barely reaches
the mushroom body -- a full-white frame fires ~14 of 4064 Kenyon cells, half-bright ~3,
dark 0 -- so visual reward-learning has almost no stimulus-locked KC eligibility to work
with. To give the *mechanism* a fair test we activate Kenyon-cell subsets directly, as
optogenetic mushroom-body conditioning does in the fly, and pair them with the identified
dopamine cells (PAM11 reward -> MBON07, PPL101 aversive -> MBON11). Kenyon subsets,
injection current and schedule are engineered inputs; nothing here is validated behaviour.

Injecting a large Kenyon population also recruits dopamine cells through the recurrent
graph, so a naive "unpaired" control still depresses synapses, and the two dopamine
populations are lopsided (15 reward PAM11 cells vs 2 aversive PPL101 cells), so the
compartments are not equally sensitive. The clean control is therefore a 2x2 reversal:
run the experiment twice, once with A->reward/B->aversive and once reversed, and ask
whether pairing a stimulus with reward depresses ITS reward-compartment edges more than
pairing the same stimulus with aversive does (and symmetrically for aversive). That
difference cancels each stimulus's own background recruitment and isolates the effect of
the pairing itself -- the definition of an associative memory.
"""

import numpy as np

from .visual import VisualMemoryBrain

DARK = np.zeros((180, 320, 3), np.uint8)  # vision held fixed; KC injection is the only CS
MARGIN = 1e-3  # a reversal-controlled pairing effect must exceed this to count as associative


def conditioned_stimuli(brain, rng):
    """Two disjoint CS: halves of the Kenyon cells that actually drive MBON07/MBON11."""
    plastic_kc = np.unique(brain.circuit["pre"])  # global KC indices that drive the MBONs
    perm = rng.permutation(plastic_kc)
    half = len(perm) // 2
    return {"A": perm[:half].astype(np.int64), "B": perm[half : 2 * half].astype(np.int64)}


def compartments(brain):
    """Boolean masks over the plastic edges: which target the reward MBON (MBON07) vs the
    aversive MBON (MBON11). circuit['mb'] is [MBON07 x4, MBON11 x2]."""
    post = brain.post[brain.circuit["edges"]]
    mb = brain.circuit["mb"]
    return {"reward": np.isin(post, mb[:4]), "aversive": np.isin(post, mb[4:])}


def _present(brain, cs_kc, ms, current, *, learning, dan=None, dan_current=20.0):
    """One trial: clear neural transients (keep memory), inject `current` into the CS
    Kenyon cells, optionally co-activate a dopamine population, integrate `ms`."""
    brain.reset(keep_memory=True)
    pulses = [(cs_kc, np.float32(current))]
    if dan is not None:
        pulses.append((dan, np.float32(dan_current)))
    brain.rgb_step(DARK, ms, learning=learning, stimulation=pulses)
    mb = brain.counts[brain.circuit["mb"]]
    return {"MBON07": int(mb[:4].sum()), "MBON11": int(mb[4:].sum())}


def _efficacy(brain, cs_kc, comp_mask):
    """Mean efficacy (weight / baseline) over this stimulus's edges into one compartment."""
    sel = np.isin(brain.circuit["pre"], cs_kc) & comp_mask
    if not sel.any():
        return float("nan")
    return float((brain.weight[brain.circuit["edges"]][sel] / brain.baseline_plastic[sel]).mean())


def _background_dan(brain, cs):
    """How much each CS recruits the dopamine cells on its own (no explicit pulse)."""
    brain.weights_frozen = True
    rew, av = brain.circuit["reward"], brain.circuit["aversive"]
    out = {}
    for name, kc in cs.items():
        brain.reset(keep_memory=False)
        _present(brain, kc, 120.0, 40.0, learning=False)
        out[name] = {"reward_DAN": int(brain.counts[rew].sum()),
                     "aversive_DAN": int(brain.counts[av].sum())}
    return out


def _condition(brain, cs, assignment, comp, epochs, current, ms):
    """Wipe memory, train each stimulus paired with its assigned dopamine population,
    then read back per-compartment efficacy (depression = 1 - efficacy, baseline is 1)."""
    brain.reset(keep_memory=False)
    brain.weights_frozen = False
    for _ in range(epochs):
        for name in ("A", "B"):
            _present(brain, cs[name], ms, current, learning=True, dan=assignment[name])
    brain.weights_frozen = True
    return {
        name: {f"{c}_edges": 1.0 - _efficacy(brain, cs[name], comp[c])
               for c in ("reward", "aversive")}
        for name in ("A", "B")
    }


def run(epochs=8, current=40.0, ms=120.0, seed=0, verbose=True):
    """2x2 reversal: train A->reward/B->aversive, then the reverse, and test whether
    pairing controls which compartment each stimulus depresses. Returns numbers + verdict."""
    brain = VisualMemoryBrain()
    rng = np.random.default_rng(seed)
    cs = conditioned_stimuli(brain, rng)
    comp = compartments(brain)
    reward = brain.circuit["reward"].astype(np.int64)
    aversive = brain.circuit["aversive"].astype(np.int64)

    background = _background_dan(brain, cs)
    normal = _condition(brain, cs, {"A": reward, "B": aversive}, comp, epochs, current, ms)
    reversed_ = _condition(brain, cs, {"A": aversive, "B": reward}, comp, epochs, current, ms)

    # For each stimulus: paired with reward = whichever condition assigned it the reward DAN.
    paired_reward = {"A": normal, "B": reversed_}
    paired_aversive = {"A": reversed_, "B": normal}
    # Associative effect of the PAIRING itself, with the stimulus's own background cancelled:
    #   how much more does pairing-with-reward depress reward edges than pairing-with-aversive?
    effect = {
        name: {
            "reward_pairing_on_reward_edges":
                paired_reward[name][name]["reward_edges"] - paired_aversive[name][name]["reward_edges"],
            "aversive_pairing_on_aversive_edges":
                paired_aversive[name][name]["aversive_edges"] - paired_reward[name][name]["aversive_edges"],
        }
        for name in ("A", "B")
    }
    reward_associative = all(effect[n]["reward_pairing_on_reward_edges"] > MARGIN for n in ("A", "B"))
    aversive_associative = all(effect[n]["aversive_pairing_on_aversive_edges"] > MARGIN for n in ("A", "B"))
    verdict = (
        "associative memory formed in both compartments"
        if reward_associative and aversive_associative
        else "associative only in the aversive compartment (reward pairing not compartment-specific)"
        if aversive_associative
        else "associative only in the reward compartment"
        if reward_associative
        else "no associative, compartment-specific memory"
    )
    report = {
        "premise": "Direct Kenyon-cell activation as CS; vision does not recruit the mushroom body.",
        "dopamine_cells": {"reward_PAM11": int(len(reward)), "aversive_PPL101": int(len(aversive))},
        "kenyon_cells_with_plastic_edges": int(len(np.unique(brain.circuit["pre"]))),
        "CS_A_cells": int(len(cs["A"])),
        "CS_B_cells": int(len(cs["B"])),
        "epochs": epochs,
        "background_dan_recruitment": background,
        "depression_normal_AtoReward_BtoAversive": normal,
        "depression_reversed_AtoAversive_BtoReward": reversed_,
        "pairing_effect": effect,
        "verdict": verdict,
        "note": "Depression = 1 - weight/baseline on a stimulus's own KC->MBON synapses, per "
        "compartment. pairing_effect is the reversal-controlled associative signal: extra "
        "depression caused by pairing the stimulus with that compartment's dopamine cells, "
        "over pairing it with the other's. It cancels each stimulus's background recruitment.",
    }
    if verbose:
        _print(report)
    return report


def _print(r):
    print(f"Associative conditioning assay -- 2x2 reversal ({r['epochs']} epochs)")
    d = r["dopamine_cells"]
    print(f"  {r['kenyon_cells_with_plastic_edges']} Kenyon cells drive the MBONs "
          f"(CS_A={r['CS_A_cells']}, CS_B={r['CS_B_cells']} cells); "
          f"dopamine: {d['reward_PAM11']} reward vs {d['aversive_PPL101']} aversive cells\n")
    print("  depression (1 - weight/baseline) of each stimulus's own edges, per compartment:")
    for name in ("A", "B"):
        n, v = r["depression_normal_AtoReward_BtoAversive"][name], r["depression_reversed_AtoAversive_BtoReward"][name]
        print(f"    CS_{name}:  normal(A->rew,B->av)  rew_edges {n['reward_edges']:+.4f}  av_edges {n['aversive_edges']:+.4f}")
        print(f"          reversed(A->av,B->rew) rew_edges {v['reward_edges']:+.4f}  av_edges {v['aversive_edges']:+.4f}")
    print("\n  reversal-controlled pairing effect (associative if > 0):")
    for name in ("A", "B"):
        e = r["pairing_effect"][name]
        print(f"    CS_{name}: reward-pairing on reward edges {e['reward_pairing_on_reward_edges']:+.4f}"
              f"   |   aversive-pairing on aversive edges {e['aversive_pairing_on_aversive_edges']:+.4f}")
    print(f"\n  Verdict: {r['verdict']}.")


def main():
    run()


if __name__ == "__main__":
    main()
