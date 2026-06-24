"""1Q tomography must be correct on qubits other than 0 (defect A8).

Previously ``_measure_expectation_1q`` measured the qubit twice and indexed the
result by the qubit number, so reconstruction on qubit>0 read the wrong bit.
"""

from __future__ import annotations

import math

from qorch import Circuit, LocalSimulator
from qorch.tomography import state_tomography_1q, purity, trace_rho


def test_tomography_pure_state_on_qubit_2():
    """H on qubit 2 of a 3-qubit register yields a pure |+> state via tomography."""
    sim = LocalSimulator(seed=4)
    c = Circuit(3).h(2)
    res = state_tomography_1q(sim, c, shots=8192, qubit=2)
    assert abs(trace_rho(res.rho) - 1.0) < 0.05
    assert abs(purity(res.rho) - 1.0) < 0.08
    # |+> has <X> ~ +1, <Y> ~ 0, <Z> ~ 0
    assert res.pauli_expectations["<X>"] > 0.9
    assert abs(res.pauli_expectations["<Z>"]) < 0.1


def test_tomography_excited_state_on_qubit_1():
    """rx(pi) on qubit 1 gives |1>: <Z> ~ -1."""
    sim = LocalSimulator(seed=5)
    c = Circuit(2).rx(1, math.pi)
    res = state_tomography_1q(sim, c, shots=8192, qubit=1)
    assert res.pauli_expectations["<Z>"] < -0.9
