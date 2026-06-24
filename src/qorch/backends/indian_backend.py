"""Indian QPU backend — simulates published Indian quantum hardware characteristics.

This adapter proves the HAL works for India's domestic hardware ecosystem.
Characteristics modeled on published specs from:
  - IIT Jodhpur trapped-ion processor (6 qubits, all-to-all, MS gate)
  - TIFR Mumbai superconducting processor (5 qubits, linear topology)
  - DRDO MIRAI Lab (6 qubits, grid topology)
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass

from qorch.backends.base import (
    Backend,
    BackendProperties,
    DeviceCalibration,
    JobResult,
    QubitCalibration,
)
from qorch.backends.simulator import _xx_matrix
from qorch.ir import Circuit, bound_params
from qorch.transpiler import IndianGateSet, decompose, route, CouplingMap
from qorch.transpiler.gateset import (
    IIT_JODHPUR_ION_TRAP,
    TIFR_SUPERCONDUCTING,
    DRDO_MIRAI,
)


@dataclass(frozen=True)
class IndianQPUConfig:
    """Configuration for a simulated Indian QPU."""

    name: str
    gate_set: IndianGateSet
    num_qubits: int
    coupling_map: CouplingMap
    # Noise characteristics typical of Indian hardware
    gate_fidelity: float = 0.99        # per-gate fidelity
    readout_fidelity: float = 0.95     # per-qubit readout fidelity
    t1_us: float = 50.0                # T1 relaxation time (μs)
    t2_us: float = 30.0                # T2 dephasing time (μs)
    gate_time_us: float = 0.5          # avg gate duration (μs)


# Pre-built Indian QPU configs
INDIAN_QPU_CONFIGS: dict[str, IndianQPUConfig] = {
    "iit-jodhpur-ion-trap": IndianQPUConfig(
        name="IIT Jodhpur Ion Trap",
        gate_set=IIT_JODHPUR_ION_TRAP,
        num_qubits=IIT_JODHPUR_ION_TRAP.num_qubits,
        coupling_map=CouplingMap(edges=()),
        gate_fidelity=0.985,
        readout_fidelity=0.93,
        t1_us=100.0,
        t2_us=80.0,
        gate_time_us=10.0,  # ion trap gates are slower
    ),
    "tifr-superconducting": IndianQPUConfig(
        name="TIFR Superconducting",
        gate_set=TIFR_SUPERCONDUCTING,
        num_qubits=TIFR_SUPERCONDUCTING.num_qubits,
        coupling_map=CouplingMap(edges=TIFR_SUPERCONDUCTING.coupling_map or ()),
        gate_fidelity=0.99,
        readout_fidelity=0.95,
        t1_us=50.0,
        t2_us=30.0,
        gate_time_us=0.5,
    ),
    "drdo-mirai": IndianQPUConfig(
        name="DRDO MIRAI",
        gate_set=DRDO_MIRAI,
        num_qubits=DRDO_MIRAI.num_qubits,
        coupling_map=CouplingMap(edges=DRDO_MIRAI.coupling_map or ()),
        gate_fidelity=0.992,
        readout_fidelity=0.96,
        t1_us=60.0,
        t2_us=40.0,
        gate_time_us=0.4,
    ),
}


def _indian_gate_matrix(name: str, params: tuple[float, ...]) -> tuple[complex, complex, complex, complex]:
    """Return 2x2 matrix (row-major) for a native Indian QPU gate."""
    if name == "x":
        return (0, 1, 1, 0)
    if name == "sx":
        return (0.5 + 0.5j, 0.5 - 0.5j, 0.5 - 0.5j, 0.5 + 0.5j)
    if name == "rz":
        theta = params[0] if params else 0.0
        return (cmath.exp(-1j * theta / 2), 0, 0, cmath.exp(1j * theta / 2))
    if name == "rx":
        theta = params[0] if params else 0.0
        c = math.cos(theta / 2)
        s = -1j * math.sin(theta / 2)
        return (c, s, s, c)
    if name == "ry":
        theta = params[0] if params else 0.0
        c = math.cos(theta / 2)
        s = math.sin(theta / 2)
        return (c, -s, s, c)
    if name == "h":
        inv = 1.0 / math.sqrt(2)
        return (inv, inv, inv, -inv)
    # default identity
    return (1, 0, 0, 1)


class IndianQPU(Backend):
    """Simulated Indian quantum processor — exercises the full stack.

    Runs circuits through the transpiler (decompose + route) then simulates
    with realistic Indian-hardware noise characteristics.
    """

    def __init__(
        self,
        config: IndianQPUConfig,
        seed: int | None = None,
        exact_noise: bool = False,
    ) -> None:
        self._config = config
        self.name = config.name
        self._seed = seed
        self._exact_noise = exact_noise
        self._rng = random.Random(seed)

    @classmethod
    def from_preset(
        cls, name: str, seed: int | None = None, exact_noise: bool = False
    ) -> "IndianQPU":
        if name not in INDIAN_QPU_CONFIGS:
            raise ValueError(
                f"unknown Indian QPU: {name!r}. "
                f"Available: {list(INDIAN_QPU_CONFIGS)}"
            )
        return cls(INDIAN_QPU_CONFIGS[name], seed=seed, exact_noise=exact_noise)

    def properties(self) -> BackendProperties:
        c = self._config
        return BackendProperties(
            num_qubits=c.num_qubits,
            basis_gates=c.gate_set.basis_gates,
            is_simulator=False,
            readout_fidelity=(c.readout_fidelity,) * c.num_qubits,
        )

    def calibration(self) -> DeviceCalibration:
        """Expose this QPU's T1/T2, per-edge error, topology and gate timing.

        Backend API v2 hook — the structured source a calibration-aware
        transpiler reads from. Surfaces the T1/T2/gate-time fields that the
        config has always carried but the old per-shot sampler ignored (A9).
        """
        c = self._config
        single_err = max(0.0, 1.0 - c.gate_fidelity)
        qubits = tuple(
            QubitCalibration(
                t1_us=c.t1_us,
                t2_us=c.t2_us,
                readout_fidelity=c.readout_fidelity,
                single_qubit_error=single_err,
            )
            for _ in range(c.num_qubits)
        )
        two_q_err = {edge: single_err for edge in c.coupling_map.edges}
        return DeviceCalibration(
            qubits=qubits,
            two_qubit_error=two_q_err,
            coupling_map=c.coupling_map.edges,
            basis_gates=c.gate_set.basis_gates,
            gate_durations_us={g: c.gate_time_us for g in c.gate_set.basis_gates},
        )

    def coupling_map(self) -> tuple[tuple[int, int], ...] | None:
        edges = self._config.coupling_map.edges
        return edges or None

    def run(self, circuit: Circuit, shots: int = 1024) -> JobResult:
        self.validate(circuit)

        # Step 1: Decompose to native gate set
        decomposed = decompose(circuit, self._config.gate_set)

        # Step 2: Route for coupling constraints
        routed = route(decomposed, self._config.coupling_map)

        # Step 3: Simulate with hardware noise.
        #   exact_noise=True ⇒ exact density-matrix sim driven by T1/T2 (A9);
        #   otherwise ⇒ fast per-shot depolarizing trajectory sampler.
        if self._exact_noise:
            from qorch.backends.density_simulator import DensitySimulator

            sim = DensitySimulator.from_calibration(self.calibration(), seed=self._seed)
            counts = sim.run(routed, shots=shots).counts
        else:
            counts = self._sample_noisy(routed, shots)

        return JobResult(
            counts=counts,
            shots=shots,
            backend_name=self.name,
            metadata={
                "native_gates": self._config.gate_set.basis_gates,
                "gate_fidelity": self._config.gate_fidelity,
                "readout_fidelity": self._config.readout_fidelity,
                "num_gates_after_transpile": len(routed.gates),
                "noise_model": "density-t1t2" if self._exact_noise else "depolarizing-trajectory",
            },
        )

    # --- noisy simulation --------------------------------------------------
    def _apply_1q(self, state: list[complex], n: int, m: tuple[complex, ...], q: int) -> None:
        stride = 1 << (n - 1 - q)
        for i in range(1 << n):
            if not (i & stride):
                j = i | stride
                a, b = state[i], state[j]
                state[i] = m[0] * a + m[1] * b
                state[j] = m[2] * a + m[3] * b

    def _apply_cx(self, state: list[complex], n: int, control: int, target: int) -> None:
        c_stride = 1 << (n - 1 - control)
        t_stride = 1 << (n - 1 - target)
        for i in range(1 << n):
            if (i & c_stride) and not (i & t_stride):
                j = i | t_stride
                state[i], state[j] = state[j], state[i]

    def _apply_swap(self, state: list[complex], n: int, q0: int, q1: int) -> None:
        self._apply_cx(state, n, q0, q1)
        self._apply_cx(state, n, q1, q0)
        self._apply_cx(state, n, q0, q1)

    def _apply_2q(self, state: list[complex], n: int, m: tuple[complex, ...],
                  q0: int, q1: int) -> None:
        """Apply an arbitrary 4×4 unitary (row-major, basis 00,01,10,11) to (q0, q1)."""
        s0 = 1 << (n - 1 - q0)
        s1 = 1 << (n - 1 - q1)
        for i in range(1 << n):
            if (i & s0) or (i & s1):
                continue
            i00, i01, i10, i11 = i, i | s1, i | s0, i | s0 | s1
            a, b, c, d = state[i00], state[i01], state[i10], state[i11]
            state[i00] = m[0] * a + m[1] * b + m[2] * c + m[3] * d
            state[i01] = m[4] * a + m[5] * b + m[6] * c + m[7] * d
            state[i10] = m[8] * a + m[9] * b + m[10] * c + m[11] * d
            state[i11] = m[12] * a + m[13] * b + m[14] * c + m[15] * d

    @staticmethod
    def _bit(index: int, n: int, qubit: int) -> int:
        return (index >> (n - 1 - qubit)) & 1

    def _apply_gate_noise(self, state: list[complex], n: int, qubits: tuple[int, ...]) -> None:
        """Apply gate error: with prob (1 - fidelity), a random Pauli."""
        fidelity = self._config.gate_fidelity
        if self._rng.random() < fidelity:
            return  # gate succeeded
        # Apply random Pauli to each qubit the gate touched
        paulis = [(0, 1, 1, 0), (0, -1j, 1j, 0), (1, 0, 0, -1)]
        for q in qubits:
            px, py, pz = 0.1, 0.1, 0.8  # Z-dominant (dephasing)
            r = self._rng.random()
            if r < px:
                self._apply_1q(state, n, paulis[0], q)
            elif r < px + py:
                self._apply_1q(state, n, paulis[1], q)
            elif r < px + py + pz:
                self._apply_1q(state, n, paulis[2], q)

    def _sample_noisy(self, circuit: Circuit, shots: int) -> dict[str, int]:
        n = circuit.num_qubits
        readout = circuit.readout_qubits

        counts: dict[str, int] = {}
        for _ in range(shots):
            state = [0j] * (1 << n)
            state[0] = 1 + 0j
            for g in circuit.gates:
                if g.name == "cx":
                    self._apply_cx(state, n, g.qubits[0], g.qubits[1])
                    self._apply_gate_noise(state, n, g.qubits)
                elif g.name == "swap":
                    self._apply_swap(state, n, g.qubits[0], g.qubits[1])
                    self._apply_gate_noise(state, n, g.qubits)
                elif g.name == "ms":
                    theta = float(g.params[0]) if g.params else 0.0
                    self._apply_2q(state, n, _xx_matrix(theta), g.qubits[0], g.qubits[1])
                    self._apply_gate_noise(state, n, g.qubits)
                else:
                    m = _indian_gate_matrix(g.name, bound_params(g.params))
                    self._apply_1q(state, n, m, g.qubits[0])
                    self._apply_gate_noise(state, n, (g.qubits[0],))

            probs = [abs(a) ** 2 for a in state]
            idx = self._rng.choices(range(len(probs)), weights=probs, k=1)[0]
            bits = [self._bit(idx, n, q) for q in readout]
            bits = [self._apply_readout_noise(b) for b in bits]
            key = "".join(str(b) for b in bits)
            counts[key] = counts.get(key, 0) + 1
        return counts

    def _apply_readout_noise(self, bit: int) -> int:
        fid = self._config.readout_fidelity
        if bit == 0:
            return 1 if self._rng.random() > fid else 0
        return 0 if self._rng.random() > fid else 1

