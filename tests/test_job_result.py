"""JobResult carries what a histogram alone cannot express.

Counts discard three things a caller often needs and cannot reconstruct: the
order shots arrived in, the fact that mitigation produced a *quasi*-distribution
which may be negative, and which physical qubit held which logical one. Each has
a field, each defaults to ``None``, and a backend that returns only counts is
still a complete implementation.
"""

from __future__ import annotations


import pytest

from qorch import Circuit, LocalSimulator
from qorch.backends.base import JobResult


def _result(**kwargs) -> JobResult:
    base = dict(counts={"00": 750, "11": 250}, shots=1000, backend_name="test")
    base.update(kwargs)
    return JobResult(**base)


# ── back-compat ──────────────────────────────────────────────────────────


def test_a_counts_only_result_is_still_valid() -> None:
    result = JobResult(counts={"0": 10}, shots=10, backend_name="x")
    assert result.memory is None
    assert result.quasi_probabilities is None
    assert result.expectation_values is None
    assert result.final_layout is None


def test_simulator_results_still_expose_counts() -> None:
    c = Circuit(2).h(0).cx(0, 1).measure(0, 1)
    result = LocalSimulator(seed=1).run(c, shots=1000)
    assert sum(result.counts.values()) == 1000
    assert set(result.counts) == {"00", "11"}


# ── probabilities ────────────────────────────────────────────────────────


def test_probabilities_normalize_counts() -> None:
    assert _result().probabilities == {"00": 0.75, "11": 0.25}


def test_probabilities_prefer_mitigated_quasi_probabilities() -> None:
    """If mitigation ran, returning raw counts would discard the correction."""
    result = _result(quasi_probabilities={"00": 0.8, "11": 0.25, "01": -0.05})
    assert result.probabilities == {"00": 0.8, "11": 0.25, "01": -0.05}


def test_quasi_probabilities_may_be_negative() -> None:
    """The point of a quasi-distribution: it does not fit in a histogram."""
    result = _result(quasi_probabilities={"0": 1.1, "1": -0.1})
    assert result.probabilities["1"] < 0


def test_zero_shots_yields_no_probabilities() -> None:
    assert JobResult(counts={}, shots=0, backend_name="x").probabilities == {}


# ── expectation values ───────────────────────────────────────────────────


def test_expectation_z_of_a_deterministic_zero() -> None:
    assert JobResult(counts={"0": 100}, shots=100, backend_name="x").expectation_z(0) == 1.0


def test_expectation_z_of_a_deterministic_one() -> None:
    assert JobResult(counts={"1": 100}, shots=100, backend_name="x").expectation_z(0) == -1.0


def test_expectation_z_of_a_superposition_is_near_zero() -> None:
    c = Circuit(1).h(0).measure(0)
    result = LocalSimulator(seed=4).run(c, shots=4000)
    assert abs(result.expectation_z(0)) < 0.1


def test_parity_expectation_on_a_bell_state_is_one() -> None:
    """|00> + |11> has even parity on every shot, so <ZZ> = 1."""
    c = Circuit(2).h(0).cx(0, 1).measure(0, 1)
    result = LocalSimulator(seed=2).run(c, shots=2000)
    assert result.parity_expectation((0, 1)) == pytest.approx(1.0)


def test_parity_expectation_follows_mitigation() -> None:
    result = _result(quasi_probabilities={"00": 1.0, "11": 0.0})
    assert result.parity_expectation((0, 1)) == pytest.approx(1.0)


def test_parity_expectation_rejects_an_out_of_range_qubit() -> None:
    with pytest.raises(ValueError, match="outside"):
        _result().parity_expectation((5,))


def test_parity_expectation_requires_a_qubit() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _result().parity_expectation(())


# ── per-shot memory ──────────────────────────────────────────────────────


def test_memory_is_off_by_default() -> None:
    c = Circuit(1).h(0).measure(0)
    assert LocalSimulator(seed=1).run(c, shots=50).memory is None


def test_memory_records_every_shot_in_order() -> None:
    c = Circuit(2).h(0).cx(0, 1).measure(0, 1)
    result = LocalSimulator(seed=1, memory=True).run(c, shots=200)

    assert result.memory is not None
    assert len(result.memory) == 200
    assert set(result.memory) <= {"00", "11"}


def test_memory_is_consistent_with_counts() -> None:
    """The histogram must be exactly the tally of the per-shot record."""
    c = Circuit(2).h(0).cx(0, 1).measure(0, 1)
    result = LocalSimulator(seed=3, memory=True).run(c, shots=300)

    tallied: dict[str, int] = {}
    for shot in result.memory:
        tallied[shot] = tallied.get(shot, 0) + 1
    assert tallied == result.counts


def test_memory_works_on_the_noisy_trajectory_path() -> None:
    from qorch.backends.simulator import GateNoise

    c = Circuit(2).h(0).cx(0, 1).measure(0, 1)
    result = LocalSimulator(
        seed=5, memory=True, gate_noise=GateNoise(depolarizing_prob=0.05)
    ).run(c, shots=100)
    assert len(result.memory) == 100


def test_memory_works_on_the_dynamic_path() -> None:
    """Dynamic circuits report the classical register, and memory must follow."""
    c = Circuit(2, num_clbits=2).h(0).measure_into(0, 0).x_if(1, 0, 1).measure_into(1, 1)
    result = LocalSimulator(seed=6, memory=True).run(c, shots=120)

    assert len(result.memory) == 120
    assert set(result.memory) <= {"00", "11"}


def test_memory_is_not_shared_between_runs() -> None:
    """A stale buffer would silently concatenate two experiments."""
    sim = LocalSimulator(seed=7, memory=True)
    c = Circuit(1).h(0).measure(0)
    first = sim.run(c, shots=40)
    second = sim.run(c, shots=25)
    assert len(first.memory) == 40
    assert len(second.memory) == 25


# ── layout provenance ────────────────────────────────────────────────────


def test_final_layout_can_be_carried_on_a_result() -> None:
    """So a caller can map a bit back to the physical qubit that produced it."""
    result = _result(final_layout=(2, 0, 1))
    assert result.final_layout == (2, 0, 1)
