"""Circuit intermediate representation + OpenQASM-3 (subset) ingestion.

The IR is immutable: every builder method returns a *new* ``Circuit`` (see ADR-5).
Bit-ordering convention: in a result bitstring, the leftmost character is qubit 0
(documented in architecture.md §5 to avoid silent endianness bugs).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

# Gates the slice understands. Adapters/transpilers may lower these further.
# Extended for transpiler: sx, rz, rx, ry, swap, ms (Mølmer–Sørensen for ion trap)
SUPPORTED_GATES: frozenset[str] = frozenset({
    "h", "x", "y", "z", "cx", "sx", "rz", "rx", "ry", "swap", "ms", "id"
})


@dataclass(frozen=True)
class Gate:
    """A single operation on one or more qubits."""

    name: str
    qubits: tuple[int, ...]
    params: tuple[float, ...] = ()


@dataclass(frozen=True)
class Circuit:
    """An immutable quantum circuit value.

    ``measured`` lists the qubits to read out, in output-bitstring order. Empty means
    "measure all qubits, ascending".
    """

    num_qubits: int
    gates: tuple[Gate, ...] = ()
    measured: tuple[int, ...] = ()

    # --- validation -------------------------------------------------------
    def __post_init__(self) -> None:
        if self.num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        for g in self.gates:
            if g.name not in SUPPORTED_GATES:
                raise ValueError(f"unsupported gate: {g.name!r}")
            for q in g.qubits:
                if not 0 <= q < self.num_qubits:
                    raise ValueError(f"qubit {q} out of range for {self.num_qubits} qubits")
        for q in self.measured:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"measured qubit {q} out of range")

    @property
    def readout_qubits(self) -> tuple[int, ...]:
        """Qubits actually read out, resolving the 'measure all' default."""
        return self.measured if self.measured else tuple(range(self.num_qubits))

    # --- immutable builders ----------------------------------------------
    def _add(self, name: str, *qubits: int, params: tuple[float, ...] = ()) -> "Circuit":
        return replace(self, gates=self.gates + (Gate(name, qubits, params),))

    def h(self, q: int) -> "Circuit":
        return self._add("h", q)

    def x(self, q: int) -> "Circuit":
        return self._add("x", q)

    def y(self, q: int) -> "Circuit":
        return self._add("y", q)

    def z(self, q: int) -> "Circuit":
        return self._add("z", q)

    def cx(self, control: int, target: int) -> "Circuit":
        return self._add("cx", control, target)

    def sx(self, q: int) -> "Circuit":
        return self._add("sx", q)

    def id(self, q: int) -> "Circuit":
        return self._add("id", q)

    def rz(self, q: int, theta: float) -> "Circuit":
        return self._add("rz", q, params=(theta,))

    def rx(self, q: int, theta: float) -> "Circuit":
        return self._add("rx", q, params=(theta,))

    def ry(self, q: int, theta: float) -> "Circuit":
        return self._add("ry", q, params=(theta,))

    def swap(self, q0: int, q1: int) -> "Circuit":
        return self._add("swap", q0, q1)

    def ms(self, q0: int, q1: int, theta: float = 0.25) -> "Circuit":
        return self._add("ms", q0, q1, params=(theta,))

    def measure(self, *qubits: int) -> "Circuit":
        return replace(self, measured=self.measured + qubits)

    # --- JSON serialization ------------------------------------------------
    def to_json(self) -> str:
        """Serialize to a JSON string."""
        data = {
            "num_qubits": self.num_qubits,
            "gates": [
                {"name": g.name, "qubits": list(g.qubits), "params": list(g.params)}
                for g in self.gates
            ],
            "measured": list(self.measured),
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Circuit":
        """Deserialize from a JSON string."""
        data = json.loads(text)
        c = cls(num_qubits=data["num_qubits"], measured=tuple(data.get("measured", [])))
        for g in data["gates"]:
            c = c._add(g["name"], *g["qubits"], params=tuple(g.get("params", [])))
        return c


# --- OpenQASM 3 (subset) ingestion ---------------------------------------
_QUBIT_DECL = re.compile(r"qubit\[(\d+)\]\s+(\w+)\s*;")
_ONE_Q = re.compile(r"(h|x|z|sx)\s+\w+\[(\d+)\]\s*;")
_PARAM_Q = re.compile(r"(rx|ry|rz)\(\s*([\d.]+)\s*\)\s+\w+\[(\d+)\]\s*;")
_CX = re.compile(r"cx\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]\s*;")
_SWAP = re.compile(r"swap\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]\s*;")
_MS = re.compile(r"ms\(\s*([\d.]+)\s*\)\s+\w+\[(\d+)\]\s*,\s*\w+\[(\d+)\]\s*;")
_MEASURE = re.compile(r"(?:\w+\s*=\s*)?measure\s+\w+\[(\d+)\]\s*;")


def to_qasm3(circuit: Circuit) -> str:
    """Emit a ``Circuit`` as an OpenQASM 3 string.

    Produces standard QASM 3 with ``qubit[n] q;``, gate calls, and ``measure``.
    """
    n = circuit.num_qubits
    lines = [
        "OPENQASM 3.0;",
        f"qubit[{n}] q;",
    ]
    for g in circuit.gates:
        qstr = ", ".join(f"q[{q}]" for q in g.qubits)
        if g.params:
            pstr = ", ".join(f"{p:.10g}" for p in g.params)
            lines.append(f"{g.name}({pstr}) {qstr};")
        else:
            lines.append(f"{g.name} {qstr};")
    for q in circuit.readout_qubits:
        lines.append(f"measure q[{q}];")
    if not circuit.readout_qubits:
        for q in range(n):
            lines.append(f"measure q[{q}];")
    lines.append("")
    return "\n".join(lines)


def from_qasm3(text: str) -> Circuit:
    """Parse a documented subset of OpenQASM 3 into a ``Circuit``.

    Supported: ``qubit[n] q;``, ``h/x/z/sx q[i];``, ``rx(θ)/ry(θ)/rz(θ) q[i];``,
    ``cx q[i], q[j];``, ``swap q[i], q[j];``, ``ms(θ) q[i], q[j];``,
    and ``measure q[i];``.
    """
    decl = _QUBIT_DECL.search(text)
    if not decl:
        raise ValueError("no 'qubit[n] name;' declaration found")
    circuit = Circuit(num_qubits=int(decl.group(1)))

    body = re.sub(r"//.*", "", text)
    for stmt in body.split(";"):
        stmt = stmt.strip()
        if not stmt or stmt.startswith(("OPENQASM", "include", "qubit", "bit")):
            continue
        line = stmt + ";"
        if m := _CX.fullmatch(line):
            circuit = circuit.cx(int(m.group(1)), int(m.group(2)))
        elif m := _SWAP.fullmatch(line):
            circuit = circuit.swap(int(m.group(1)), int(m.group(2)))
        elif m := _MS.fullmatch(line):
            circuit = circuit.ms(int(m.group(2)), int(m.group(3)), float(m.group(1)))
        elif m := _PARAM_Q.fullmatch(line):
            circuit = circuit._add(m.group(1), int(m.group(3)), params=(float(m.group(2)),))
        elif m := _ONE_Q.fullmatch(line):
            circuit = circuit._add(m.group(1), int(m.group(2)))
        elif m := _MEASURE.fullmatch(line):
            circuit = circuit.measure(int(m.group(1)))
    return circuit
