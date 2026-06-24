"""Stabilizer (CHP tableau) simulator — polynomial-time Clifford + measurement.

Implements the Aaronson–Gottesman tableau algorithm ("Improved Simulation of
Stabilizer Circuits", PRA 70, 052328). A stabilizer state on ``n`` qubits is
tracked as a ``(2n+1) × (2n+1)`` binary tableau, so Clifford circuits with
mid-circuit measurement run in polynomial time and scale to hundreds of qubits —
far beyond the exponential statevector engine. This is the enabling tool for
quantum error correction (syndrome extraction is pure-Clifford).

Non-Clifford gates (t, rz, rx, ry, ms) are rejected with a clear error: this
backend is exact only for the Clifford group. Dynamic circuits (mid-circuit
measurement, reset, classical control) are fully supported.
"""

from __future__ import annotations

import random

from qorch.backends.base import Backend, BackendProperties, JobResult
from qorch.ir import Circuit, Measure, Reset

# Clifford gates this backend understands (others raise).
_CLIFFORD_GATES = frozenset({"h", "x", "y", "z", "cx", "swap", "sx", "id"})


class _Tableau:
    """Aaronson–Gottesman stabilizer tableau for ``n`` qubits.

    Rows 0..n-1 are destabilizers, n..2n-1 stabilizers, 2n a scratch row.
    Each row holds x bits, z bits, and a phase bit r.
    """

    def __init__(self, n: int, rng: random.Random) -> None:
        self.n = n
        self.rng = rng
        m = 2 * n + 1
        self.x = [bytearray(n) for _ in range(m)]
        self.z = [bytearray(n) for _ in range(m)]
        self.r = bytearray(m)
        for i in range(n):
            self.x[i][i] = 1        # destabilizer i = X_i
            self.z[n + i][i] = 1    # stabilizer i = Z_i

    # --- Clifford generators ---------------------------------------------
    def h(self, q: int) -> None:
        x, z, r = self.x, self.z, self.r
        for i in range(2 * self.n):
            r[i] ^= x[i][q] & z[i][q]
            x[i][q], z[i][q] = z[i][q], x[i][q]

    def s(self, q: int) -> None:
        x, z, r = self.x, self.z, self.r
        for i in range(2 * self.n):
            r[i] ^= x[i][q] & z[i][q]
            z[i][q] ^= x[i][q]

    def cnot(self, a: int, b: int) -> None:
        x, z, r = self.x, self.z, self.r
        for i in range(2 * self.n):
            r[i] ^= x[i][a] & z[i][b] & (x[i][b] ^ z[i][a] ^ 1)
            x[i][b] ^= x[i][a]
            z[i][a] ^= z[i][b]

    def x_gate(self, q: int) -> None:
        z, r = self.z, self.r
        for i in range(2 * self.n):
            r[i] ^= z[i][q]

    def z_gate(self, q: int) -> None:
        x, r = self.x, self.r
        for i in range(2 * self.n):
            r[i] ^= x[i][q]

    def y_gate(self, q: int) -> None:
        x, z, r = self.x, self.z, self.r
        for i in range(2 * self.n):
            r[i] ^= x[i][q] ^ z[i][q]

    def sx(self, q: int) -> None:
        # sqrt(X) = H S H (up to global phase, irrelevant to stabilizer state)
        self.h(q)
        self.s(q)
        self.h(q)

    def swap(self, a: int, b: int) -> None:
        self.cnot(a, b)
        self.cnot(b, a)
        self.cnot(a, b)

    # --- measurement ------------------------------------------------------
    @staticmethod
    def _g(x1: int, z1: int, x2: int, z2: int) -> int:
        """Phase exponent of Pauli product (CHP ``g`` function), in {-1,0,1}."""
        if x1 == 0 and z1 == 0:
            return 0
        if x1 == 1 and z1 == 1:
            return z2 - x2
        if x1 == 1 and z1 == 0:
            return z2 * (2 * x2 - 1)
        return x2 * (1 - 2 * z2)

    def _rowsum(self, h: int, i: int) -> None:
        """Left-multiply row ``h`` by row ``i`` (Pauli product), tracking phase."""
        total = 2 * self.r[h] + 2 * self.r[i]
        xi, zi, xh, zh = self.x[i], self.z[i], self.x[h], self.z[h]
        for j in range(self.n):
            total += self._g(xi[j], zi[j], xh[j], zh[j])
        self.r[h] = 1 if (total % 4) == 2 else 0
        for j in range(self.n):
            xh[j] ^= xi[j]
            zh[j] ^= zi[j]

    def measure(self, q: int) -> int:
        """Measure qubit ``q`` in the Z basis; collapse the tableau; return 0/1."""
        n = self.n
        x = self.x
        p = -1
        for i in range(n, 2 * n):
            if x[i][q]:
                p = i
                break
        if p >= 0:
            # random outcome
            for i in range(2 * n):
                if i != p and x[i][q]:
                    self._rowsum(i, p)
            self.x[p - n] = bytearray(self.x[p])
            self.z[p - n] = bytearray(self.z[p])
            self.r[p - n] = self.r[p]
            self.x[p] = bytearray(n)
            self.z[p] = bytearray(n)
            self.z[p][q] = 1
            self.r[p] = self.rng.randint(0, 1)
            return self.r[p]
        # deterministic outcome via scratch row
        self.x[2 * n] = bytearray(n)
        self.z[2 * n] = bytearray(n)
        self.r[2 * n] = 0
        for i in range(n):
            if x[i][q]:
                self._rowsum(2 * n, i + n)
        return self.r[2 * n]


class StabilizerSimulator(Backend):
    """Clifford-only backend using the CHP tableau; scales to many qubits.

    Supports the Clifford subset of the IR plus mid-circuit measurement, reset,
    and classical control. Raises on non-Clifford gates.
    """

    name = "stabilizer-simulator"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def properties(self) -> BackendProperties:
        return BackendProperties(
            num_qubits=1024,  # polynomial scaling; advisory ceiling
            basis_gates=tuple(sorted(_CLIFFORD_GATES)),
            is_simulator=True,
        )

    def run(self, circuit: Circuit, shots: int = 1024) -> JobResult:
        self.validate(circuit)
        counts: dict[str, int] = {}
        for _ in range(shots):
            key = self._one_shot(circuit)
            counts[key] = counts.get(key, 0) + 1
        return JobResult(
            counts=counts,
            shots=shots,
            backend_name=self.name,
            metadata={"method": "stabilizer-tableau", "dynamic": circuit.is_dynamic},
        )

    def _one_shot(self, circuit: Circuit) -> str:
        n = circuit.num_qubits
        tab = _Tableau(n, self._rng)
        creg = [0] * circuit.num_clbits
        for op in circuit.gates:
            if op.condition is not None and any(creg[c] != v for c, v in op.condition):
                continue
            if isinstance(op, Measure):
                creg[op.cbit] = tab.measure(op.qubits[0])
            elif isinstance(op, Reset):
                if tab.measure(op.qubits[0]) == 1:
                    tab.x_gate(op.qubits[0])
            else:
                self._apply_clifford(tab, op.name, op.qubits)
        if circuit.num_clbits > 0:
            return "".join(str(b) for b in creg)
        # static circuit: measure the terminal read-out qubits in logical order
        return "".join(str(tab.measure(q)) for q in circuit.readout_qubits)

    @staticmethod
    def _apply_clifford(tab: _Tableau, name: str, qubits: tuple[int, ...]) -> None:
        if name == "h":
            tab.h(qubits[0])
        elif name == "x":
            tab.x_gate(qubits[0])
        elif name == "y":
            tab.y_gate(qubits[0])
        elif name == "z":
            tab.z_gate(qubits[0])
        elif name == "sx":
            tab.sx(qubits[0])
        elif name == "cx":
            tab.cnot(qubits[0], qubits[1])
        elif name == "swap":
            tab.swap(qubits[0], qubits[1])
        elif name == "id":
            pass
        else:
            raise ValueError(
                f"gate {name!r} is not Clifford: the stabilizer simulator supports "
                f"only {sorted(_CLIFFORD_GATES)} (use LocalSimulator for universal gates)"
            )
