"""Readout mitigation: the M1 differentiator, proven on a noisy simulated readout.

This is the test that justifies the whole layer existing (charter M1 gate): run the same
circuit raw vs. through mitigation and show the error shrinks.
"""

from __future__ import annotations

import math

from qorch import Circuit, LocalSimulator, ReadoutNoise
from qorch.mitigation.readout import ReadoutMitigator


def _total_variation(counts: dict[str, float], truth: dict[str, float], shots: int) -> float:
    """TV distance between a (possibly mitigated) histogram and the ideal distribution."""
    labels = set(counts) | set(truth)
    return 0.5 * sum(
        abs(counts.get(k, 0.0) / shots - truth.get(k, 0.0)) for k in labels
    )


def test_inverts_known_calibration_matrix():
    """1-qubit calibration: 5% flip both ways. Mitigation recovers the true 50/50."""
    # A[i][j] = P(measure i | prepared j); 0.95 correct, 0.05 flip
    mitigator = ReadoutMitigator.from_calibration_matrix(
        labels=["0", "1"],
        matrix=[[0.95, 0.05], [0.05, 0.95]],
    )
    # observed counts skewed by readout error from a true 50/50
    observed = {"0": 500, "1": 500}
    mitigated = mitigator.apply(observed)
    assert math.isclose(mitigated["0"], 500.0, abs_tol=1.0)
    assert math.isclose(mitigated["1"], 500.0, abs_tol=1.0)


def test_mitigation_reduces_error_on_noisy_simulator():
    """End-to-end: a noisy single-qubit superposition, mitigated, is closer to ideal.

    Uses asymmetric readout error (relaxation-dominated, P(0|1) > P(1|0)), which actually
    skews a 50/50 distribution — unlike a symmetric flip, which would leave it untouched.
    """
    shots = 20000
    p1_given0, p0_given1 = 0.05, 0.15  # asymmetric, as on real hardware
    circuit = Circuit(num_qubits=1).h(0)  # ideal: 50% '0', 50% '1'
    ideal = {"0": 0.5, "1": 0.5}

    noise = ReadoutNoise(p1_given0=p1_given0, p0_given1=p0_given1)
    raw = LocalSimulator(seed=123, readout_noise=noise).run(circuit, shots)

    # calibration matrix A[i][j] = P(measure i | prepared j)
    mitigator = ReadoutMitigator.from_calibration_matrix(
        labels=["0", "1"],
        matrix=[[1 - p1_given0, p0_given1], [p1_given0, 1 - p0_given1]],
    )
    mitigated = mitigator.apply(raw.counts)

    raw_err = _total_variation({k: float(v) for k, v in raw.counts.items()}, ideal, shots)
    mit_err = _total_variation(mitigated, ideal, shots)

    # the whole point: mitigation must measurably reduce the error
    assert mit_err < raw_err
    assert mit_err < 0.02
