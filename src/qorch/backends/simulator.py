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

from qorch.backends import gpu_kernel, numpy_kernel
from qorch.backends.base import Backend, BackendProperties, JobResult
from qorch.gates import GATES, gate_matrix, xx_matrix
from qorch.ir import Circuit, Measure, Reset, bound_params

_INV_SQRT2 = 1.0 / math.sqrt(2.0)

# Statevector width at which the numpy kernel starts paying for its own dispatch
# overhead. Measured, not guessed: pure Python is faster below this and numpy is
# roughly an order of magnitude faster well above it.
_NUMPY_MIN_QUBITS = 8

# Constant single-qubit matrices, taken from the registry so the simulator has
# no private copy to drift from (see qorch.gates).
_GATES_1Q: dict[str, tuple[complex, complex, complex, complex]] = {
    name: g.matrix(())
    for name, g in GATES.items()
    if g.arity == 1 and g.num_params == 0 and g.matrix is not None
}


def _gate_matrix(name: str, params: tuple[float, ...] = ()) -> tuple[complex, complex, complex, complex]:
    """Return 2x2 matrix (row-major) for any supported gate.

    Delegates to the gate registry so the simulator and the compiler cannot
    disagree about what a gate *is* — see :mod:`qorch.gates`.
    """
    return gate_matrix(name, params)


#: Backwards-compatible alias. The definition now lives in the gate registry,
#: where gate matrices belong — this module had been the de-facto owner of the
#: MS matrix, which made a simulator internal load-bearing for the hardware
#: adapter that imported it.
_xx_matrix = xx_matrix

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

    @classmethod
    def from_readout_fidelity(
        cls,
        readout_fidelity: float,
        asymmetric: bool = True,
    ) -> "ReadoutNoise":
        """Build readout noise model from average readout fidelity.

        With asymmetric=True (default), p(0|1) > p(1|0) reflecting T1 decay.
        """
        err = 1.0 - readout_fidelity
        if asymmetric:
            return cls(p1_given0=err * 0.3, p0_given1=err * 0.7)
        return cls(p1_given0=err * 0.5, p0_given1=err * 0.5)

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

    @classmethod
    def from_gate_fidelity(cls, gate_fidelity: float) -> "GateNoise":
        """Build gate noise from per-gate average fidelity."""
        return cls(depolarizing_prob=max(0.0, 1.0 - gate_fidelity))

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
        use_numpy: bool | None = None,
        memory: bool = False,
        use_gpu: bool = False,
    ) -> None:
        """``use_numpy=None`` (the default) picks the faster kernel per circuit.

        numpy is not automatically better. Its per-gate dispatch costs more than
        the pure-Python loop saves until the statevector is big enough to
        amortize it — measured on this machine, pure Python is ~7x faster at 2
        qubits and numpy ~10x faster at 12 (see ``_NUMPY_MIN_QUBITS``).

        Pass ``True`` or ``False`` to force a kernel. ``False`` is worth using in
        tests: the pure-Python path is what makes qorch run with no third-party
        packages at all, so it has to stay correct on its own.
        """
        self._rng = random.Random(seed)
        self._noise = readout_noise or ReadoutNoise()
        self._gate_noise = gate_noise or GateNoise()
        self._use_numpy = use_numpy
        self._memory = memory
        self._use_gpu = use_gpu
        if use_numpy and not numpy_kernel.is_available():
            raise ValueError("use_numpy=True but numpy is not installed")
        if use_gpu and not gpu_kernel.is_available():
            raise ValueError(
                "use_gpu=True but no CUDA device is available "
                "(install qorch[gpu] and check nvidia-smi)"
            )

    def _should_use_numpy(self, circuit: Circuit) -> bool:
        if self._use_numpy is not None:
            return self._use_numpy
        return (
            circuit.num_qubits >= _NUMPY_MIN_QUBITS and numpy_kernel.is_available()
        )

    def properties(self) -> BackendProperties:
        return BackendProperties(
            num_qubits=30,  # statevector practical ceiling; advisory only
            basis_gates=("h", "x", "y", "z", "sx", "rx", "ry", "rz", "cx", "swap", "id"),
            is_simulator=True,
            readout_fidelity=(),
        )

    def run(self, circuit: Circuit, shots: int = 1024) -> JobResult:
        self.validate(circuit)
        self._last_memory: list[str] = []
        if circuit.is_dynamic:
            counts = self._run_dynamic(circuit, shots)
        elif self._gate_noise.active:
            counts = self._sample_trajectories(circuit, shots)
        else:
            counts = self._sample(self._evolve(circuit), circuit, shots)
        return JobResult(
            counts=counts,
            memory=tuple(self._last_memory) if self._memory else None,
            shots=shots,
            backend_name=self.name,
            metadata={
                "p1_given0": self._noise.p1_given0,
                "p0_given1": self._noise.p0_given1,
                "depolarizing_prob": self._gate_noise.depolarizing_prob,
                "dynamic": circuit.is_dynamic,
            },
        )

    # --- dynamic execution (mid-circuit measurement + classical control) ---
    def _run_dynamic(self, circuit: Circuit, shots: int) -> dict[str, int]:
        """Execute a dynamic circuit shot-by-shot.

        Each shot is an independent trajectory: gates apply unitarily, a
        ``Measure`` collapses the state and writes a classical bit, conditional
        gates consult the classical register (feed-forward), and ``Reset`` returns
        a qubit to |0⟩. The result key is the classical register (cbit 0 leftmost).
        Readout noise, if configured, is applied to each measurement outcome.
        """
        n = circuit.num_qubits
        counts: dict[str, int] = {}
        for _ in range(shots):
            state = [0j] * (1 << n)
            state[0] = 1 + 0j
            creg = [0] * circuit.num_clbits
            for op in circuit.gates:
                if op.condition is not None:
                    if any(creg[cbit] != val for cbit, val in op.condition):
                        continue
                if isinstance(op, Measure):
                    outcome = self._measure_qubit(state, n, op.qubits[0])
                    if self._noise.active:
                        outcome = self._apply_readout_noise(outcome)
                    creg[op.cbit] = outcome
                elif isinstance(op, Reset):
                    if self._measure_qubit(state, n, op.qubits[0]) == 1:
                        self._apply_1q(state, n, _GATES_1Q["x"], op.qubits[0])
                elif op.name == "cx":
                    self._apply_cx(state, n, op.qubits[0], op.qubits[1])
                elif op.name == "swap":
                    self._apply_swap(state, n, op.qubits[0], op.qubits[1])
                elif op.name == "ms":
                    theta = float(op.params[0]) if op.params else 0.0
                    self._apply_2q(state, n, _xx_matrix(theta), op.qubits[0], op.qubits[1])
                else:
                    m = _gate_matrix(op.name, bound_params(op.params))
                    self._apply_1q(state, n, m, op.qubits[0])
            key = "".join(str(b) for b in creg)
            counts[key] = counts.get(key, 0) + 1
            if self._memory:
                self._last_memory.append(key)
        return counts

    def _measure_qubit(self, state: list[complex], n: int, q: int) -> int:
        """Projectively measure qubit ``q``: sample outcome, collapse, renormalize."""
        stride = 1 << (n - 1 - q)
        p1 = sum(abs(state[i]) ** 2 for i in range(1 << n) if i & stride)
        outcome = 1 if self._rng.random() < p1 else 0
        norm = math.sqrt(p1 if outcome else 1.0 - p1)
        for i in range(1 << n):
            bit = 1 if (i & stride) else 0
            if bit != outcome:
                state[i] = 0j
            elif norm > 0:
                state[i] /= norm
        return outcome

    # --- statevector evolution -------------------------------------------
    def _evolve(self, circuit: Circuit) -> list[complex]:
        """Evolve |0...0> through the circuit, via numpy when it is available.

        Both kernels implement the same convention and agree to floating-point
        precision; the numpy one just moves the per-amplitude loop into compiled
        code, which is what makes 14+ qubits practical.
        """
        if self._use_gpu:
            # Opt-in only, never automatic. The numpy path switches on above a
            # *measured* crossover; no such measurement exists for the GPU, and
            # inventing a threshold would present a guess as a tuning decision.
            return gpu_kernel.evolve(circuit)
        if self._should_use_numpy(circuit):
            return numpy_kernel.evolve(circuit)
        return self._evolve_python(circuit)

    def _evolve_python(self, circuit: Circuit) -> list[complex]:
        n = circuit.num_qubits
        state = [0j] * (1 << n)
        state[0] = 1 + 0j  # |0...0>
        for gate in circuit.gates:
            if gate.name == "cx":
                self._apply_cx(state, n, gate.qubits[0], gate.qubits[1])
            elif gate.name == "swap":
                self._apply_swap(state, n, gate.qubits[0], gate.qubits[1])
            elif gate.name == "ms":
                theta = float(gate.params[0]) if gate.params else 0.0
                self._apply_2q(state, n, _xx_matrix(theta), gate.qubits[0], gate.qubits[1])
            else:
                m = _gate_matrix(gate.name, bound_params(gate.params))
                self._apply_1q(state, n, m, gate.qubits[0])
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
            elif gate.name == "ms":
                theta = float(gate.params[0]) if gate.params else 0.0
                self._apply_2q(state, n, _xx_matrix(theta), gate.qubits[0], gate.qubits[1])
            else:
                m = _gate_matrix(gate.name, bound_params(gate.params))
                self._apply_1q(state, n, m, gate.qubits[0])
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

    def _apply_2q(self, state: list[complex], n: int, m: tuple[complex, ...],
                  q0: int, q1: int) -> None:
        """Apply an arbitrary 4×4 unitary (row-major, basis 00,01,10,11) to (q0, q1)."""
        s0 = 1 << (n - 1 - q0)
        s1 = 1 << (n - 1 - q1)
        for i in range(1 << n):
            if (i & s0) or (i & s1):
                continue  # process each 2-qubit subspace once from its 00 anchor
            i00 = i
            i01 = i | s1
            i10 = i | s0
            i11 = i | s0 | s1
            a, b, c, d = state[i00], state[i01], state[i10], state[i11]
            state[i00] = m[0] * a + m[1] * b + m[2] * c + m[3] * d
            state[i01] = m[4] * a + m[5] * b + m[6] * c + m[7] * d
            state[i10] = m[8] * a + m[9] * b + m[10] * c + m[11] * d
            state[i11] = m[12] * a + m[13] * b + m[14] * c + m[15] * d

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
            if self._memory:
                self._last_memory.append(key)
        return counts

    def _draw_error_pattern(
        self, circuit: Circuit
    ) -> list[tuple[int, int, str]]:
        """Sample which depolarizing errors fire, as (gate index, qubit, pauli).

        Drawn in exactly the order ``_evolve_trajectory`` used to draw them —
        per gate, per qubit, ``random()`` then ``choice()`` only if it fires — so
        the random stream is unchanged and a seeded run reproduces bit for bit.
        Evolution itself consumes no randomness, which is what makes separating
        the two safe.
        """
        p = self._gate_noise.depolarizing_prob
        pattern: list[tuple[int, int, str]] = []
        for index, gate in enumerate(circuit.gates):
            for q in gate.qubits:
                if self._rng.random() < p:
                    pattern.append((index, q, self._rng.choice(("x", "y", "z"))))
        return pattern

    def _evolve_with_errors(
        self, circuit: Circuit, pattern: list[tuple[int, int, str]]
    ) -> list[complex]:
        """Evolve applying a pre-drawn set of Pauli errors."""
        n = circuit.num_qubits
        state = [0j] * (1 << n)
        state[0] = 1 + 0j
        by_index: dict[int, list[tuple[int, str]]] = {}
        for index, qubit, pauli in pattern:
            by_index.setdefault(index, []).append((qubit, pauli))

        for index, gate in enumerate(circuit.gates):
            if gate.name == "cx":
                self._apply_cx(state, n, gate.qubits[0], gate.qubits[1])
            elif gate.name == "swap":
                self._apply_swap(state, n, gate.qubits[0], gate.qubits[1])
            elif gate.name == "ms":
                theta = float(gate.params[0]) if gate.params else 0.0
                self._apply_2q(state, n, _xx_matrix(theta), gate.qubits[0], gate.qubits[1])
            else:
                m = _gate_matrix(gate.name, bound_params(gate.params))
                self._apply_1q(state, n, m, gate.qubits[0])
            for qubit, pauli in by_index.get(index, ()):
                self._apply_1q(state, n, _PAULIS[pauli], qubit)
        return state

    def _sample_trajectories(self, circuit: Circuit, shots: int) -> dict[str, int]:
        """Monte Carlo: one independent noisy trajectory per shot.

        A trajectory in which no error fired *is* the noiseless state, so it does
        not need re-deriving. At realistic error rates most trajectories are
        error-free — that is what a low error rate means — and the old code
        re-evolved every one of them, making p=0.001 cost exactly as much as
        p=0.01. The ideal state is now computed at most once and reused.

        This is an exact optimization, not an approximation: the errors sampled
        and the random stream consumed are identical to before.
        """
        n = circuit.num_qubits
        readout = circuit.readout_qubits
        counts: dict[str, int] = {}
        ideal: list[complex] | None = None
        for _ in range(shots):
            pattern = self._draw_error_pattern(circuit)
            if pattern:
                amps = self._evolve_with_errors(circuit, pattern)
            else:
                if ideal is None:
                    ideal = self._evolve(circuit)
                amps = ideal
            probs = [abs(a) ** 2 for a in amps]
            idx = self._rng.choices(range(len(probs)), weights=probs, k=1)[0]
            bits = [self._bit(idx, n, q) for q in readout]
            if self._noise.active:
                bits = [self._apply_readout_noise(b) for b in bits]
            key = "".join(str(int(b)) for b in bits)
            counts[key] = counts.get(key, 0) + 1
            if self._memory:
                self._last_memory.append(key)
        return counts

    def _apply_readout_noise(self, bit: int) -> int:
        if bit == 0:
            return 1 if self._rng.random() < self._noise.p1_given0 else 0
        return 0 if self._rng.random() < self._noise.p0_given1 else 1
