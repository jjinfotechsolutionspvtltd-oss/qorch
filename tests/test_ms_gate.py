"""Tests for the Mølmer–Sørensen (XX) entangling gate across simulators.

Regression for defect A1: ``ms`` was in SUPPORTED_GATES and native to the
ion-trap backend but crashed ``LocalSimulator`` and was faked as ``cx`` inside
``IndianQPU``.
"""

from __future__ import annotations

import cmath
import math

from qorch import Circuit, LocalSimulator, DensitySimulator, IndianQPU


def test_ms_runs_on_local_simulator():
    """ms no longer crashes the statevector simulator."""
    c = Circuit(2).ms(0, 1, math.pi / 4).measure(0, 1)
    result = LocalSimulator(seed=1).run(c, shots=512)
    assert sum(result.counts.values()) == 512


def test_ms_pi_over_4_is_maximally_entangling():
    """ms(π/4) on |00> creates a Bell-like state: only 00 and 11 appear."""
    c = Circuit(2).ms(0, 1, math.pi / 4).measure(0, 1)
    result = LocalSimulator(seed=2).run(c, shots=4000).counts
    assert set(result) <= {"00", "11"}
    assert result.get("00", 0) > 1500
    assert result.get("11", 0) > 1500


def test_ms_matches_analytic_amplitudes():
    """XX(θ)|00> = cos θ|00> - i sin θ|11>; check probabilities for θ=π/6."""
    theta = math.pi / 6
    sim = LocalSimulator(seed=3)
    state = sim._evolve(Circuit(2).ms(0, 1, theta))
    # |00> amplitude index 0, |11> amplitude index 3
    assert abs(abs(state[0]) ** 2 - math.cos(theta) ** 2) < 1e-9
    assert abs(abs(state[3]) ** 2 - math.sin(theta) ** 2) < 1e-9
    assert abs(state[1]) < 1e-9 and abs(state[2]) < 1e-9
    # phase on |11> is -i sin θ
    assert cmath.isclose(state[3], -1j * math.sin(theta), abs_tol=1e-9)


def test_ms_runs_on_density_simulator():
    c = Circuit(2).ms(0, 1, math.pi / 4).measure(0, 1)
    result = DensitySimulator(seed=1).run(c, shots=512).counts
    assert set(result) <= {"00", "11"}


def test_ion_trap_native_circuit_runs():
    """A circuit using the ion-trap native entangler executes end-to-end."""
    qpu = IndianQPU.from_preset("iit-jodhpur-ion-trap", seed=1)
    c = Circuit(2).rx(0, math.pi / 2).ms(0, 1, math.pi / 4).measure(0, 1)
    result = qpu.run(c, shots=512)
    assert sum(result.counts.values()) == 512
