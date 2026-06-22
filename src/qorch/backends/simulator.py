"""Dependency-free local statevector backend.

Proves the HAL end-to-end with zero external packages (ADR-6), so the slice runs on any
machine with no QPU account. At M2 this is swapped for / joined by Qiskit Aer or the
QS-001 engine behind the same ``Backend`` interface. An optional readout-noise model lets
us demonstrate the error-mitigation payoff without real hardware.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from qorch.backends.base import Backend, BackendProperties, JobResult
from qorch.ir import Circuit

_INV_SQRT2 = 1.0 / math.sqrt(2.0)

# Single-qubit gate matrices as 2x2 row-major tuples of complex amplitudes.
_GATES_1Q: dict[str, tuple[complex, complex, complex, complex]] = {
    "h": (_INV_SQRT2, _INV_SQRT2, _INV_SQRT2, -_INV_SQRT2),
    "x": (0, 1, 1, 0),
    "y": (0, -1j, 1j, 0),
    "z": (1, 0, 0, -1),
}

# Pauli error operators for the depolarizing channel (trajectory Monte Carlo).
_PAULIS: dict[str, tuple[complex, complex, complex, complex]] = {
    "x": (0, 1, 1, 0),
    "y": (0, -1j, 1j, 0),
    "z": (1, 0, 0, -1),
}


@dataclass(frozen=True)
class ReadoutNoise:
    """Asymmetric per-qubit readout error applied at measurement.

    Mirrors real hardware: T1 relaxation during readout makes a true |1> more likely to be
    reported as 0 than the reverse, so ``p0_given1`` typically exceeds ``p1_given0``.
    A symmetric error on a uniform distribution is a fixed point with nothing to mitigate —
    the asymmetry is what makes readout calibration worthwhile.
    """

    p1_given0: float = 0.0  # P(report 1 | true 0)
    p0_given1: float = 0.0  # P(report 0 | true 1)

    @property
    def active(self) -> bool:
        return self.p1_given0 > 0.0 or self.p0_given1 > 0.0


@dataclass(frozen=True)
class GateNoise:
    """Depolarizing error after each gate: with probability ``depolarizing_prob`` a uniformly
    random Pauli (X, Y, or Z) is applied to each qubit the gate touched.

    This noise scales with circuit depth — the precondition for zero-noise extrapolation to
    have anything to extrapolate (readout noise alone is depth-independent). Simulated by
    trajectory Monte Carlo so the backend stays a pure-statevector, dependency-free engine.
    """

    depolarizing_prob: float = 0.0

    @property
    def active(self) -> bool:
        return self.depolarizing_prob > 0.0


class LocalSimulator(Backend):
    name = "local-simulator"

    def __init__(
        self,
        seed: int | None = None,
        readout_noise: ReadoutNoise | None = None,
        gate_noise: GateNoise | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._noise = readout_noise or ReadoutNoise()
        self._gate_noise = gate_noise or GateNoise()

    def properties(self) -> BackendProperties:
        return BackendProperties(
            num_qubits=30,  # statevector practical ceiling; advisory only
            basis_gates=("h", "x", "z", "cx"),
            is_simulator=True,
            readout_fidelity=(),
        )

    def run(self, circuit: Circuit, shots: int = 1024) -> JobResult:
        self.validate(circuit)
        if self._gate_noise.active:
            counts = self._sample_trajectories(circuit, shots)
        else:
            counts = self._sample(self._evolve(circuit), circuit, shots)
        return JobResult(
            counts=counts,
            shots=shots,
            backend_name=self.name,
            metadata={
                "p1_given0": self._noise.p1_given0,
                "p0_given1": self._noise.p0_given1,
                "depolarizing_prob": self._gate_noise.depolarizing_prob,
            },
        )

    # --- statevector evolution -------------------------------------------
    def _evolve(self, circuit: Circuit) -> list[complex]:
        n = circuit.num_qubits
        state = [0j] * (1 << n)
        state[0] = 1 + 0j  # |0...0>
        for gate in circuit.gates:
            if gate.name == "cx":
                self._apply_cx(state, n, gate.qubits[0], gate.qubits[1])
            elif gate.name == "swap":
                self._apply_swap(state, n, gate.qubits[0], gate.qubits[1])
            else:
                self._apply_1q(state, n, _GATES_1Q[gate.name], gate.qubits[0])
        return state

    def _evolve_trajectory(self, circuit: Circuit) -> list[complex]:
        """One noisy trajectory: apply each gate, then a random Pauli error per touched qubit."""
        n = circuit.num_qubits
        state = [0j] * (1 << n)
        state[0] = 1 + 0j
        p = self._gate_noise.depolarizing_prob
        for gate in circuit.gates:
            if gate.name == "cx":
                self._apply_cx(state, n, gate.qubits[0], gate.qubits[1])
            elif gate.name == "swap":
                self._apply_swap(state, n, gate.qubits[0], gate.qubits[1])
            else:
                self._apply_1q(state, n, _GATES_1Q[gate.name], gate.qubits[0])
            for q in gate.qubits:
                if self._rng.random() < p:
                    pauli = _PAULIS[self._rng.choice(("x", "y", "z"))]
                    self._apply_1q(state, n, pauli, q)
        return state

    @staticmethod
    def _bit(index: int, n: int, qubit: int) -> int:
        # qubit 0 is the most-significant bit so bitstrings read leftmost = qubit 0
        return (index >> (n - 1 - qubit)) & 1

    def _apply_1q(self, state: list[complex], n: int, m: tuple[complex, ...], q: int) -> None:
        stride = 1 << (n - 1 - q)
        for i in range(1 << n):
            if not (i & stride):  # i has qubit q = 0; pair with qubit q = 1
                j = i | stride
                a, b = state[i], state[j]
                state[i] = m[0] * a + m[1] * b
                state[j] = m[2] * a + m[3] * b

    def _apply_swap(self, state: list[complex], n: int, q0: int, q1: int) -> None:
        self._apply_cx(state, n, q0, q1)
        self._apply_cx(state, n, q1, q0)
        self._apply_cx(state, n, q0, q1)

    def _apply_cx(self, state: list[complex], n: int, control: int, target: int) -> None:
        c_stride = 1 << (n - 1 - control)
        t_stride = 1 << (n - 1 - target)
        for i in range(1 << n):
            if (i & c_stride) and not (i & t_stride):
                j = i | t_stride
                state[i], state[j] = state[j], state[i]

    # --- measurement ------------------------------------------------------
    def _sample(self, amps: list[complex], circuit: Circuit, shots: int) -> dict[str, int]:
        n = circuit.num_qubits
        probs = [abs(a) ** 2 for a in amps]
        outcomes = self._rng.choices(range(len(probs)), weights=probs, k=shots)
        readout = circuit.readout_qubits
        counts: dict[str, int] = {}
        for idx in outcomes:
            bits = [self._bit(idx, n, q) for q in readout]
            if self._noise.active:
                bits = [self._apply_readout_noise(b) for b in bits]
            key = "".join(str(int(b)) for b in bits)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _sample_trajectories(self, circuit: Circuit, shots: int) -> dict[str, int]:
        """Monte Carlo: one independent noisy trajectory per shot, then one measurement each."""
        n = circuit.num_qubits
        readout = circuit.readout_qubits
        counts: dict[str, int] = {}
        for _ in range(shots):
            amps = self._evolve_trajectory(circuit)
            probs = [abs(a) ** 2 for a in amps]
            idx = self._rng.choices(range(len(probs)), weights=probs, k=1)[0]
            bits = [self._bit(idx, n, q) for q in readout]
            if self._noise.active:
                bits = [self._apply_readout_noise(b) for b in bits]
            key = "".join(str(int(b)) for b in bits)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _apply_readout_noise(self, bit: int) -> int:
        if bit == 0:
            return 1 if self._rng.random() < self._noise.p1_given0 else 0
        return 0 if self._rng.random() < self._noise.p0_given1 else 1
