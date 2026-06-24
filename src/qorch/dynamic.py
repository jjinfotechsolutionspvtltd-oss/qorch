"""Dynamic-circuit algorithms: quantum teleportation and a bit-flip repetition code.

Both rely on the Phase-2 IR features — mid-circuit measurement, classical
registers, and feed-forward (classically-conditioned gates) — that the static
``Circuit`` could not express. They run on any ``Backend`` whose ``run`` supports
dynamic circuits (``LocalSimulator`` does).
"""

from __future__ import annotations

from dataclasses import dataclass

from qorch.backends.base import Backend
from qorch.ir import Circuit, Gate, static_gates


def _prepend_prep(circuit: Circuit, prep: Circuit | None, target: int) -> Circuit:
    """Apply a 1-qubit state-prep circuit's gates onto ``target`` of ``circuit``."""
    if prep is None:
        return circuit
    out = circuit
    for g in static_gates(prep.gates):
        out = out._add_op(Gate(g.name, (target,) * len(g.qubits), g.params))
    return out


# ── Quantum teleportation ────────────────────────────────────────────────


def teleportation_circuit(state_prep: Circuit | None = None) -> Circuit:
    """Build a 3-qubit teleportation circuit.

    q0 = state to teleport, (q1, q2) = a shared Bell pair. Alice entangles and
    measures q0, q1 into c0, c1; Bob applies feed-forward X (on c1) and Z (on c0)
    to q2, which then holds the original state, finally measured into c2.

    ``state_prep`` is an optional 1-qubit circuit preparing the state on q0.
    """
    c = Circuit(3, num_clbits=3)
    c = _prepend_prep(c, state_prep, target=0)
    # Bell pair on (q1, q2)
    c = c.h(1).cx(1, 2)
    # Alice's basis change + measurement
    c = c.cx(0, 1).h(0)
    c = c.measure_into(0, 0).measure_into(1, 1)
    # Bob's feed-forward corrections
    c = c.x_if(2, 1, 1)   # X if c1 == 1
    c = c.z_if(2, 0, 1)   # Z if c0 == 1
    # read out the teleported qubit
    c = c.measure_into(2, 2)
    return c


def run_teleportation(
    backend: Backend,
    state_prep: Circuit | None = None,
    shots: int = 4096,
) -> dict[str, float]:
    """Teleport a state and return the marginal distribution of Bob's qubit (c2)."""
    counts = backend.run(teleportation_circuit(state_prep), shots=shots).counts
    marginal = {"0": 0.0, "1": 0.0}
    for key, n in counts.items():
        marginal[key[2]] += n
    total = sum(marginal.values()) or 1.0
    return {k: v / total for k, v in marginal.items()}


# ── 3-qubit bit-flip repetition code ─────────────────────────────────────


@dataclass(frozen=True)
class RepetitionCodeResult:
    """Outcome of a repetition-code run."""

    logical_distribution: dict[str, float]   # marginal of the decoded logical bit
    syndrome_distribution: dict[str, float]   # marginal of the 2-bit syndrome


def repetition_code_circuit(
    state_prep: Circuit | None = None,
    error_qubit: int | None = None,
) -> Circuit:
    """3-qubit bit-flip code: encode, (optional) inject one X error, correct.

    Qubits 0–2 are data, 3–4 syndrome ancillas. Classical bits: c0, c1 hold the
    syndrome, c2 the decoded logical value. The two-bit syndrome selects which
    data qubit (if any) to flip back via feed-forward — exactly the structure of
    a stabilizer-code correction.
    """
    c = Circuit(5, num_clbits=3)
    c = _prepend_prep(c, state_prep, target=0)
    # encode |ψ⟩ → α|000⟩ + β|111⟩
    c = c.cx(0, 1).cx(0, 2)
    # inject a single bit-flip error
    if error_qubit is not None:
        c = c.x(error_qubit)
    # syndrome extraction: s0 = parity(q0,q1), s1 = parity(q1,q2)
    c = c.cx(0, 3).cx(1, 3)
    c = c.cx(1, 4).cx(2, 4)
    c = c.measure_into(3, 0).measure_into(4, 1)
    # feed-forward correction keyed on the 2-bit syndrome
    c = c.x_if_all(0, ((0, 1), (1, 0)))   # syndrome 10 → flip q0
    c = c.x_if_all(1, ((0, 1), (1, 1)))   # syndrome 11 → flip q1
    c = c.x_if_all(2, ((0, 0), (1, 1)))   # syndrome 01 → flip q2
    # decode: after correction all data qubits agree; read q0
    c = c.measure_into(0, 2)
    return c


def run_repetition_code(
    backend: Backend,
    state_prep: Circuit | None = None,
    error_qubit: int | None = None,
    shots: int = 4096,
) -> RepetitionCodeResult:
    """Run the repetition code and return logical + syndrome marginals."""
    counts = backend.run(
        repetition_code_circuit(state_prep, error_qubit), shots=shots
    ).counts
    logical = {"0": 0.0, "1": 0.0}
    syndrome: dict[str, float] = {}
    for key, n in counts.items():
        logical[key[2]] += n
        syndrome[key[:2]] = syndrome.get(key[:2], 0.0) + n
    total = sum(logical.values()) or 1.0
    return RepetitionCodeResult(
        logical_distribution={k: v / total for k, v in logical.items()},
        syndrome_distribution={k: v / total for k, v in syndrome.items()},
    )
