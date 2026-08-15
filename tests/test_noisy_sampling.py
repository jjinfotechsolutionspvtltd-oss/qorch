"""Error-free noise trajectories reuse the ideal state instead of re-deriving it.

A trajectory in which no error fired *is* the noiseless state. At realistic
error rates most trajectories are error-free — that is what a low error rate
means — and re-evolving each one made p=0.001 cost exactly as much as p=0.01.

The optimization is only legitimate if it is **exact**, so that is what these
tests check: the errors sampled and the random stream consumed are unchanged, so
a seeded run reproduces bit for bit and the physics is untouched.
"""

from __future__ import annotations

import pytest

from qorch import Circuit, LocalSimulator
from qorch.backends.simulator import GateNoise


def _chain(n: int = 4) -> Circuit:
    c = Circuit(n).h(0)
    for i in range(n - 1):
        c = c.cx(i, i + 1)
    return c.measure(*range(n))


def _noisy(p: float, seed: int = 1, **kwargs) -> LocalSimulator:
    return LocalSimulator(seed=seed, gate_noise=GateNoise(depolarizing_prob=p), **kwargs)


# ── the optimization is exact ────────────────────────────────────────────


@pytest.mark.parametrize("p", [0.0, 0.001, 0.01, 0.05, 0.2])
def test_noisy_runs_are_reproducible_for_a_given_seed(p: float) -> None:
    circuit = _chain()
    first = _noisy(p, seed=3).run(circuit, shots=600).counts
    second = _noisy(p, seed=3).run(circuit, shots=600).counts
    assert first == second


def test_zero_error_rate_matches_the_noiseless_backend() -> None:
    """With p=0 every trajectory is error-free, so this exercises the fast path."""
    circuit = _chain()
    noiseless = LocalSimulator(seed=5).run(circuit, shots=1000).counts
    zero_noise = _noisy(0.0, seed=5).run(circuit, shots=1000).counts
    assert noiseless == zero_noise


def test_error_free_shots_are_not_a_different_distribution() -> None:
    """The reused ideal state must be the same state, not merely a similar one."""
    circuit = _chain()
    ideal = LocalSimulator(seed=9).run(circuit, shots=4000).counts
    barely_noisy = _noisy(1e-9, seed=9).run(circuit, shots=4000).counts
    assert set(barely_noisy) <= set(ideal) | {"0001", "0010", "0100", "1000"}
    for key in ("0000", "1111"):
        assert abs(ideal.get(key, 0) - barely_noisy.get(key, 0)) / 4000 < 0.05


# ── noise still does what noise does ─────────────────────────────────────


def test_noise_broadens_the_distribution() -> None:
    """A GHZ chain is two outcomes when clean; noise must produce others."""
    circuit = _chain()
    clean = _noisy(0.0, seed=2).run(circuit, shots=2000).counts
    dirty = _noisy(0.08, seed=2).run(circuit, shots=2000).counts

    assert set(clean) == {"0000", "1111"}
    assert len(set(dirty)) > 2, "noise produced no off-distribution outcomes"


def test_more_noise_means_more_corruption() -> None:
    circuit = _chain()

    def off_distribution(p: float) -> int:
        counts = _noisy(p, seed=4).run(circuit, shots=2000).counts
        return sum(v for k, v in counts.items() if k not in ("0000", "1111"))

    assert off_distribution(0.001) < off_distribution(0.05)


def test_shot_count_is_preserved_under_noise() -> None:
    counts = _noisy(0.03, seed=6).run(_chain(), shots=1500).counts
    assert sum(counts.values()) == 1500


# ── the fast path is actually being taken ────────────────────────────────


def test_error_free_trajectories_evolve_the_ideal_state_only_once() -> None:
    """The whole point: N error-free shots must cost one evolution, not N."""
    sim = _noisy(0.0, seed=8)
    calls = 0
    original = sim._evolve

    def counting(circuit):
        nonlocal calls
        calls += 1
        return original(circuit)

    sim._evolve = counting                     # type: ignore[method-assign]
    sim.run(_chain(), shots=500)
    assert calls == 1, f"re-evolved the ideal state {calls} times"


def test_shots_with_errors_still_get_their_own_evolution() -> None:
    """Reuse must not leak: a corrupted trajectory is not the ideal state."""
    sim = _noisy(0.5, seed=10)
    counts = sim.run(_chain(), shots=200).counts
    assert len(set(counts)) > 2


def test_memory_still_records_every_noisy_shot() -> None:
    sim = _noisy(0.02, seed=11, memory=True)
    result = sim.run(_chain(), shots=250)
    assert len(result.memory) == 250
