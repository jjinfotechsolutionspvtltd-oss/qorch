"""QS-004 M2: the Qiskit adapter behind the same Backend interface.

The endianness translation is tested *now*, with no SDK installed (it is a pure function).
The live-execution tests activate automatically once ``pip install qorch[qiskit]`` provides
Qiskit Aer — they assert the same Circuit behaves identically on the simulator and the adapter,
which is the architectural proof that the HAL is genuinely hardware-agnostic.
"""

from __future__ import annotations

import pytest

from qorch import (
    Backend,
    Circuit,
    LocalSimulator,
    QiskitBackend,
    reorder_counts_qiskit_to_qorch,
)


# --- pure translation: testable without Qiskit ---------------------------
def test_endianness_reversal_maps_qubit_zero_to_leftmost():
    # Qiskit key '01' means c1=0, c0=1 → qubit0=1, qubit1=0 → qorch '10'
    assert reorder_counts_qiskit_to_qorch({"01": 7}) == {"10": 7}


def test_translation_strips_register_spaces_and_merges_keys():
    # both keys normalise to '01' → reversed '10', so counts merge
    out = reorder_counts_qiskit_to_qorch({"0 1": 3, "01": 2})
    assert out == {"10": 5}


def test_qiskit_backend_is_a_backend_without_importing_qiskit():
    # importing the class must not require the optional SDK
    assert issubclass(QiskitBackend, Backend)


def test_construction_does_not_touch_qiskit():
    # a fake BackendV2-like object; no real Qiskit needed to wrap it
    class FakeBackend:
        name = "fake"
        num_qubits = 5
        target = None

    qb = QiskitBackend(FakeBackend())
    assert qb.name == "fake"
    props = qb.properties()
    assert props.num_qubits == 5


def test_ibm_requires_a_token(monkeypatch):
    monkeypatch.delenv("QISKIT_IBM_TOKEN", raising=False)
    with pytest.raises(ValueError, match="token"):
        QiskitBackend.ibm("ibm_brisbane")


# --- live execution: activates when qiskit-aer is installed ---------------
def test_bell_state_on_aer_matches_convention():
    pytest.importorskip("qiskit_aer")
    bell = Circuit(num_qubits=2).h(0).cx(0, 1)
    counts = QiskitBackend.aer(seed=42).run(bell, shots=2000).counts
    assert set(counts) <= {"00", "11"}, "Bell correlation must survive translation"


def test_x_on_qubit0_is_leftmost_on_aer():
    pytest.importorskip("qiskit_aer")
    counts = QiskitBackend.aer(seed=1).run(Circuit(num_qubits=2).x(0), shots=200).counts
    # X on qubit 0 only → qorch '10' (leftmost = qubit 0), same as LocalSimulator
    assert counts == {"10": 200}


def test_cross_backend_parity_sim_vs_aer():
    """Same circuit, two backends, compatible distributions — the HAL proof."""
    pytest.importorskip("qiskit_aer")
    circuit = Circuit(num_qubits=2).h(0).cx(0, 1)
    sim = set(LocalSimulator(seed=1).run(circuit, 2000).counts)
    aer = set(QiskitBackend.aer(seed=1).run(circuit, 2000).counts)
    assert sim == aer <= {"00", "11"}
