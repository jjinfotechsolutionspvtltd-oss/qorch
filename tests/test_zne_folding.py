"""ZNE unitary folding must be valid for parametrized / non-self-inverse gates.

Regression for defect A4: ``_dagger`` previously just reversed the gate list,
which is only a true adjoint when every gate is self-inverse. Folding a circuit
with rotations then produced a circuit that was *not* logically equal to the
original, so the extrapolation had no meaning.
"""

from __future__ import annotations

from qorch import Circuit, LocalSimulator
from qorch.ir import Gate, inverse_gates
from qorch.mitigation.zne import fold_circuit


def _tvd(a, b, shots):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / shots


def test_inverse_gates_rotations_negate_angle():
    assert inverse_gates(Gate("rz", (0,), (0.7,))) == [Gate("rz", (0,), (-0.7,))]
    assert inverse_gates(Gate("ms", (0, 1), (0.4,))) == [Gate("ms", (0, 1), (-0.4,))]


def test_inverse_gates_sx_and_t():
    assert inverse_gates(Gate("sx", (0,))) == [Gate("sx", (0,))] * 3
    assert inverse_gates(Gate("t", (0,))) == [Gate("t", (0,))] * 7


def test_folding_parametrized_circuit_preserves_distribution():
    """C (C† C)^k must equal C logically — same distribution at every odd scale."""
    base = (Circuit(2)
            .ry(0, 0.9).rz(0, 0.4).sx(1).cx(0, 1).rx(1, 1.3).t(0)
            .measure(0, 1))
    ref = LocalSimulator(seed=11).run(base, shots=6000).counts
    prev_len = 0
    for scale in (1, 3, 5):
        folded = fold_circuit(base, scale)
        # gate count grows monotonically with scale; logical unitary unchanged
        assert len(folded.gates) >= prev_len
        prev_len = len(folded.gates)
        got = LocalSimulator(seed=11).run(folded, shots=6000).counts
        assert _tvd(ref, got, 6000) < 0.04, f"scale {scale} changed distribution"


def test_folding_single_qubit_rotation_is_identity_equivalent():
    base = Circuit(1).rx(0, 2.1).rz(0, 0.6).measure(0)
    ref = LocalSimulator(seed=3).run(base, shots=8000).counts
    got = LocalSimulator(seed=3).run(fold_circuit(base, 3), shots=8000).counts
    assert _tvd(ref, got, 8000) < 0.04
