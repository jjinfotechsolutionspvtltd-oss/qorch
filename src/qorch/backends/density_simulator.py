"""Density-matrix simulator for exact mixed-state simulation.

Enables quantum channels, T1/T2 relaxation, and thermal noise — missing
from the pure-state trajectory Monte Carlo in ``simulator.py``.

Complexity O(4^n) per gate, practical up to ~8 qubits.
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass

from qorch.backends.base import Backend, BackendProperties, JobResult
from qorch.ir import Circuit


_INV_SQRT2 = 1.0 / math.sqrt(2.0)

_GATES_1Q: dict[str, tuple[complex, complex, complex, complex]] = {
    "h": (_INV_SQRT2, _INV_SQRT2, _INV_SQRT2, -_INV_SQRT2),
    "x": (0, 1, 1, 0),
    "y": (0, -1j, 1j, 0),
    "z": (1, 0, 0, -1),
    "sx": (0.5 + 0.5j, 0.5 - 0.5j, 0.5 - 0.5j, 0.5 + 0.5j),
    "id": (1, 0, 0, 1),
}


def _gate_matrix(name: str, params: tuple[float, ...] = ()) -> tuple[complex, complex, complex, complex]:
    if name in _GATES_1Q:
        return _GATES_1Q[name]
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
    if name == "rz":
        theta = params[0] if params else 0.0
        return (cmath.exp(-1j * theta / 2), 0, 0, cmath.exp(1j * theta / 2))
    raise ValueError(f"unknown gate: {name!r}")





def _cx_matrix() -> tuple[complex, ...]:
    """4×4 CNOT matrix as flat 16-tuple (row-major)."""
    return (1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 1,
            0, 0, 1, 0)


def _swap_matrix() -> tuple[complex, ...]:
    return (1, 0, 0, 0,
            0, 0, 1, 0,
            0, 1, 0, 0,
            0, 0, 0, 1)


@dataclass(frozen=True)
class NoiseChannel:
    """Per-qubit noise model for density-matrix simulation."""

    depolarizing_prob: float = 0.0
    amplitude_damping_gamma: float = 0.0   # T1 relaxation rate
    phase_damping_lambda: float = 0.0      # T2 dephasing rate

    @classmethod
    def from_gate_fidelity(
        cls,
        gate_fidelity: float,
        t1_us: float = 0.0,
        t2_us: float = 0.0,
        gate_time_us: float = 0.5,
    ) -> "NoiseChannel":
        """Build noise model from device calibration data.

        Args:
            gate_fidelity: per-gate average fidelity (0-1)
            t1_us: T1 relaxation time in microseconds (0 = ignore)
            t2_us: T2 dephasing time in microseconds (0 = ignore)
            gate_time_us: gate duration in microseconds
        """
        dep_prob = max(0.0, 1.0 - gate_fidelity)
        amp_gamma = 0.0
        phase_lam = 0.0
        if t1_us > 0 and gate_time_us > 0:
            amp_gamma = 1.0 - math.exp(-gate_time_us / t1_us)
        if t2_us > 0 and gate_time_us > 0:
            t_phi = 1.0 / (1.0 / t2_us - 1.0 / (2.0 * t1_us)) if t1_us > 0 and 1.0 / t2_us > 1.0 / (2.0 * t1_us) else 0.0
            if t_phi > 0:
                phase_lam = 1.0 - math.exp(-gate_time_us / t_phi)
        return cls(
            depolarizing_prob=dep_prob,
            amplitude_damping_gamma=amp_gamma,
            phase_damping_lambda=phase_lam,
        )


class DensitySimulator(Backend):
    """Density-matrix simulator with exact noise channels.

    Unlike ``LocalSimulator`` (Monte Carlo trajectory), this backend
    constructs the full density matrix, enabling exact simulation of:

    - Depolarizing noise (Pauli channel)
    - Amplitude damping (T1 relaxation)
    - Phase damping (T2 dephasing)

    Usage::

        sim = DensitySimulator(seed=0, noise=NoiseChannel(depolarizing_prob=0.01))
        result = sim.run(circuit, shots=4096)
    """

    name = "density-simulator"

    def __init__(
        self,
        seed: int | None = None,
        noise: NoiseChannel | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._noise = noise or NoiseChannel()

    def properties(self) -> BackendProperties:
        return BackendProperties(
            num_qubits=8,
            basis_gates=("h", "x", "y", "z", "sx", "rx", "ry", "rz", "cx", "swap", "id"),
            is_simulator=True,
        )

    def run(self, circuit: Circuit, shots: int = 1024) -> JobResult:
        self.validate(circuit)
        rho = self._evolve(circuit)
        counts = self._sample(rho, circuit, shots)
        return JobResult(counts=counts, shots=shots, backend_name=self.name, metadata={})

    # --- density matrix ---------------------------------------------------

    def _init_rho(self, n: int) -> list[complex]:
        """Return |0...0⟩⟨0...0| as a flat row-major density matrix."""
        N = 1 << n
        rho = [0j] * (N * N)
        rho[0] = 1.0 + 0j
        return rho

    def _evolve(self, circuit: Circuit) -> list[complex]:
        n = circuit.num_qubits
        rho = self._init_rho(n)
        for gate in circuit.gates:
            self._apply_gate(rho, n, gate)
            if self._noise.depolarizing_prob > 0:
                self._apply_depolarizing(rho, n, gate.qubits)
            if self._noise.amplitude_damping_gamma > 0:
                self._apply_amplitude_damping(rho, n, gate.qubits)
            if self._noise.phase_damping_lambda > 0:
                self._apply_phase_damping(rho, n, gate.qubits)
        return rho

    def _apply_gate(self, rho: list[complex], n: int, gate: object) -> None:
        from qorch.ir import Gate as QGate
        g: QGate = gate
        if g.name == "cx":
            self._apply_2q_gate(rho, n, _cx_matrix(), g.qubits[0], g.qubits[1])
        elif g.name == "swap":
            self._apply_2q_gate(rho, n, _swap_matrix(), g.qubits[0], g.qubits[1])
        else:
            m = _gate_matrix(g.name, g.params)
            self._apply_1q_gate(rho, n, m, g.qubits[0])

    def _apply_1q_gate(self, rho: list[complex], n: int,
                        m: tuple[complex, ...], q: int) -> None:
        """Apply single-qubit unitary: ρ → U ρ U†."""
        N = 1 << n
        new = [0j] * (N * N)
        m00, m01, m10, m11 = m
        shift = n - 1 - q

        for i in range(N):
            iq = (i >> shift) & 1
            i0 = i & ~(1 << shift)          # q-bit = 0
            i1 = i | (1 << shift)            # q-bit = 1
            row_coeff0 = m00 if iq == 0 else m10
            row_coeff1 = m01 if iq == 0 else m11
            for j in range(N):
                jq = (j >> shift) & 1
                j0 = j & ~(1 << shift)
                j1 = j | (1 << shift)
                col_coeff0 = (m00 if jq == 0 else m10).conjugate()
                col_coeff1 = (m01 if jq == 0 else m11).conjugate()

                val = (row_coeff0 * col_coeff0 * rho[i0 * N + j0]
                       + row_coeff0 * col_coeff1 * rho[i0 * N + j1]
                       + row_coeff1 * col_coeff0 * rho[i1 * N + j0]
                       + row_coeff1 * col_coeff1 * rho[i1 * N + j1])
                new[i * N + j] = val
        rho[:] = new

    def _apply_2q_gate(self, rho: list[complex], n: int,
                        m16: tuple[complex, ...], q0: int, q1: int) -> None:
        """Apply 2-qubit unitary: ρ → U ρ U†. m16 is row-major 4×4=16 elements."""
        N = 1 << n
        new = [0j] * (N * N)
        s0 = n - 1 - q0
        s1 = n - 1 - q1

        for i in range(N):
            iq0 = (i >> s0) & 1
            iq1 = (i >> s1) & 1
            for j in range(N):
                jq0 = (j >> s0) & 1
                jq1 = (j >> s1) & 1
                val = 0j
                for a in (0, 1):
                    for b in (0, 1):
                        ia = ((i & ~(1 << s0)) if a == 0 else (i | (1 << s0)))
                        ia = ((ia & ~(1 << s1)) if b == 0 else (ia | (1 << s1)))
                        u_ab = m16[(iq0 * 2 + iq1) * 4 + a * 2 + b]
                        if u_ab == 0j:
                            continue
                        for c in (0, 1):
                            for d in (0, 1):
                                jc = ((j & ~(1 << s0)) if c == 0 else (j | (1 << s0)))
                                jc = ((jc & ~(1 << s1)) if d == 0 else (jc | (1 << s1)))
                                u_cd = m16[(jq0 * 2 + jq1) * 4 + c * 2 + d]
                                if u_cd == 0j:
                                    continue
                                val += u_ab * u_cd.conjugate() * rho[ia * N + jc]
                new[i * N + j] = val
        rho[:] = new

    # --- noise channels ---------------------------------------------------

    def _apply_1q_gate_adj(self, rho: list[complex], n: int,
                            m: tuple[complex, ...], q: int) -> None:
        """Apply adjoint on the right: ρ → ρ U† (not full ρ → U ρ U†)."""
        N = 1 << n
        new = [0j] * (N * N)
        m00, m01, m10, m11 = m
        shift = n - 1 - q

        for i in range(N):
            for j in range(N):
                jq = (j >> shift) & 1
                j0 = j & ~(1 << shift)
                j1 = j | (1 << shift)
                col_coeff0 = m00.conjugate() if jq == 0 else m10.conjugate()
                col_coeff1 = m01.conjugate() if jq == 0 else m11.conjugate()
                new[i * N + j] = (col_coeff0 * rho[i * N + j0]
                                  + col_coeff1 * rho[i * N + j1])
        rho[:] = new

    def _apply_depolarizing(self, rho: list[complex], n: int,
                             qubits: tuple[int, ...]) -> None:
        """ρ → (1-p)ρ + p/3 (XρX + YρY + ZρZ) on each qubit."""
        p = self._noise.depolarizing_prob
        px = p / 3.0
        one_minus_p = 1.0 - p
        N = 1 << n
        pauli_mats = {
            "x": (0, 1, 1, 0),
            "y": (0, -1j, 1j, 0),
            "z": (1, 0, 0, -1),
        }
        for q in qubits:
            new = [0j] * (N * N)
            # (1-p) ρ contribution
            shift = n - 1 - q
            for i in range(N):
                iq = (i >> shift) & 1
                i0 = i & ~(1 << shift)
                i1 = i | (1 << shift)
                for j in range(N):
                    jq = (j >> shift) & 1
                    j0 = j & ~(1 << shift)
                    j1 = j | (1 << shift)
                    if iq == jq:
                        if iq == 0:
                            new[i * N + j] += one_minus_p * rho[i0 * N + j0]
                        else:
                            new[i * N + j] += one_minus_p * rho[i1 * N + j1]
            for pauli, mat in pauli_mats.items():
                tmp = rho[:]
                self._apply_1q_gate(tmp, n, mat, q)
                self._apply_1q_gate_adj(tmp, n, mat, q)
                for idx in range(N * N):
                    new[idx] += px * tmp[idx]
            rho[:] = new

    def _apply_amplitude_damping(self, rho: list[complex], n: int,
                                  qubits: tuple[int, ...]) -> None:
        gamma = self._noise.amplitude_damping_gamma
        sqrt_g = math.sqrt(gamma)
        sqrt_1mg = math.sqrt(1.0 - gamma)
        for q in qubits:
            new = [0j] * len(rho)
            N = 1 << n
            shift = n - 1 - q
            for i in range(N):
                iq = (i >> shift) & 1
                i0 = i & ~(1 << shift)
                i1 = i | (1 << shift)
                for j in range(N):
                    jq = (j >> shift) & 1
                    j0 = j & ~(1 << shift)
                    j1 = j | (1 << shift)
                    idx = i * N + j
                    # E₀ ρ E₀†
                    if iq == 0 and jq == 0:
                        new[idx] += (sqrt_1mg * sqrt_1mg * rho[i0 * N + j0]
                                     + sqrt_g * sqrt_g * rho[i1 * N + j1])
                    elif iq == 0 and jq == 1:
                        new[idx] += sqrt_1mg * sqrt_g * rho[i0 * N + j1]
                    elif iq == 1 and jq == 0:
                        new[idx] += sqrt_g * sqrt_1mg * rho[i1 * N + j0]
                    else:
                        pass  # E₁ ρ E₁† has no |1⟩⟨1| component
            rho[:] = new

    def _apply_phase_damping(self, rho: list[complex], n: int,
                              qubits: tuple[int, ...]) -> None:
        lam = self._noise.phase_damping_lambda
        sqrt_1ml = math.sqrt(1.0 - lam)
        N = 1 << n
        for q in qubits:
            shift = n - 1 - q
            for i in range(N):
                iq = (i >> shift) & 1
                for j in range(N):
                    jq = (j >> shift) & 1
                    if iq != jq:
                        rho[i * N + j] *= sqrt_1ml

    # --- measurement ------------------------------------------------------

    def _sample(self, rho: list[complex], circuit: Circuit, shots: int) -> dict[str, int]:
        n = circuit.num_qubits
        N = 1 << n
        readout = circuit.readout_qubits
        probs = [max(0.0, rho[i * N + i].real) for i in range(N)]
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        outcomes = self._rng.choices(range(N), weights=probs, k=shots)
        counts: dict[str, int] = {}
        for idx in outcomes:
            bits = [(idx >> (n - 1 - q)) & 1 for q in readout]
            key = "".join(str(b) for b in bits)
            counts[key] = counts.get(key, 0) + 1
        return counts
