"""Circuit intermediate representation + OpenQASM-3 (subset) ingestion.

The IR is immutable: every builder method returns a *new* ``Circuit`` (see ADR-5).
Bit-ordering convention: in a result bitstring, the leftmost character is qubit 0
(documented in ``docs/architecture.md`` §5 — endianness bugs are silent, so the
convention is stated once and obeyed everywhere).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

from qorch.gates import (
    ANGLE_INVERSE_GATES,
    SELF_INVERSE_GATES,
    SUPPORTED_GATE_NAMES,
)

# Gates the slice understands. Adapters/transpilers may lower these further.
# Derived from the gate registry so the set cannot drift from the definitions.
SUPPORTED_GATES: frozenset[str] = SUPPORTED_GATE_NAMES


@dataclass(frozen=True)
class Parameter:
    """A free symbolic parameter (e.g. a variational angle), bound later.

    Lets a circuit be built once and re-evaluated at many angles via
    :meth:`Circuit.bind` — the standard pattern for VQE/QAOA optimization loops,
    avoiding a full circuit rebuild per iteration.
    """

    name: str

    def __float__(self) -> float:
        raise TypeError(
            f"parameter {self.name!r} is unbound — call Circuit.bind(...) "
            f"before simulating, transpiling, or serializing"
        )


# A gate angle is either a concrete number or an unbound symbolic parameter.
ParamValue = float | Parameter


def bound_params(params: tuple["ParamValue", ...]) -> tuple[float, ...]:
    """Coerce gate params to concrete floats (raises on any unbound Parameter)."""
    return tuple(float(p) for p in params)


def static_gates(ops: "tuple[Operation, ...]") -> "tuple[Gate, ...]":
    """Return ``ops`` narrowed to unitary gates, rejecting dynamic operations.

    The transpiler, optimizer, mitigation passes, density simulator, and QMI
    format operate on static circuits only. Calling them on a dynamic circuit
    (mid-circuit measurement / reset / classical control) raises a clear error
    rather than silently dropping or mishandling those operations.
    """
    out: list["Gate"] = []
    for op in ops:
        if not isinstance(op, Gate):
            raise ValueError(
                f"operation {op.name!r} is not supported here: this stage requires a "
                f"static circuit (no mid-circuit measurement, reset, or classical control)"
            )
        out.append(op)
    return tuple(out)


def with_qubits(op: "Operation", qubits: tuple[int, ...]) -> "Operation":
    """Return ``op`` acting on ``qubits`` instead, preserving everything else.

    Routing permutes which *physical* wire each logical qubit lives on; every
    other attribute (classical condition, measurement target bit, params) must
    survive that rewrite untouched. Passes use this instead of reconstructing
    ops by hand, which is how conditions get silently dropped.
    """
    return replace(op, qubits=qubits)


@dataclass(frozen=True)
class Gate:
    """A single unitary operation on one or more qubits.

    ``condition`` enables classical control (feed-forward): when set to
    ``(clbit, value)`` the gate is applied only if classical bit ``clbit`` equals
    ``value`` at run time. ``None`` (the default) means unconditional — so every
    existing static-circuit construction is unchanged.
    """

    name: str
    qubits: tuple[int, ...]
    params: tuple[ParamValue, ...] = ()
    condition: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True)
class Measure:
    """Mid-circuit measurement of one qubit, written to a classical bit.

    Unlike the terminal ``Circuit.measured`` read-out, a ``Measure`` collapses the
    state *during* execution so later gates can be classically conditioned on it.
    """

    qubits: tuple[int, ...]   # exactly one qubit
    cbit: int
    name: str = "measure"
    params: tuple[ParamValue, ...] = ()
    condition: tuple[tuple[int, int], ...] | None = None


@dataclass(frozen=True)
class Reset:
    """Reset one qubit to |0⟩ mid-circuit (measure-and-correct semantics)."""

    qubits: tuple[int, ...]   # exactly one qubit
    name: str = "reset"
    params: tuple[ParamValue, ...] = ()
    condition: tuple[tuple[int, int], ...] | None = None


# Any element of a circuit's operation list.
Operation = Gate | Measure | Reset


@dataclass(frozen=True)
class Circuit:
    """An immutable quantum circuit value.

    ``measured`` lists the qubits to read out, in output-bitstring order. Empty means
    "measure all qubits, ascending". ``num_clbits`` declares classical bits for
    dynamic circuits (mid-circuit measurement + classical control); it defaults to
    0 so static circuits are unaffected.
    """

    num_qubits: int
    gates: tuple[Operation, ...] = ()
    measured: tuple[int, ...] = ()
    num_clbits: int = 0

    # --- validation -------------------------------------------------------
    def _validate_op(self, op: "Operation") -> None:
        """Check a single operation against this circuit's qubit/clbit widths."""
        if isinstance(op, Gate) and op.name not in SUPPORTED_GATES:
            raise ValueError(f"unsupported gate: {op.name!r}")
        for q in op.qubits:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"qubit {q} out of range for {self.num_qubits} qubits")
        if isinstance(op, Measure) and not 0 <= op.cbit < self.num_clbits:
            raise ValueError(f"measure target clbit {op.cbit} out of range")
        if op.condition is not None:
            for cbit, _value in op.condition:
                if not 0 <= cbit < self.num_clbits:
                    raise ValueError(f"condition clbit {cbit} out of range")

    def __post_init__(self) -> None:
        if self.num_qubits <= 0:
            raise ValueError("num_qubits must be positive")
        if self.num_clbits < 0:
            raise ValueError("num_clbits must be non-negative")
        for op in self.gates:
            self._validate_op(op)
        for q in self.measured:
            if not 0 <= q < self.num_qubits:
                raise ValueError(f"measured qubit {q} out of range")

    def _appending(self, op: "Operation") -> "Circuit":
        """This circuit plus ``op``, validating only ``op``.

        Every construction normally revalidates the whole gate list, which makes
        building an n-gate circuit O(n²) — the builder methods were the single
        largest cost in QEC threshold estimation, above the stabilizer simulator
        itself. Since ``self`` was validated when it was built and the operations
        are immutable, re-checking them cannot discover anything new; only ``op``
        is genuinely unchecked.

        The full check in ``__post_init__`` still guards every circuit built
        directly, deserialized, or produced by ``dataclasses.replace`` — this is
        strictly a fast path for growing an already-valid circuit by one.
        """
        self._validate_op(op)
        new = object.__new__(Circuit)
        object.__setattr__(new, "num_qubits", self.num_qubits)
        object.__setattr__(new, "gates", self.gates + (op,))
        object.__setattr__(new, "measured", self.measured)
        object.__setattr__(new, "num_clbits", self.num_clbits)
        return new

    @property
    def readout_qubits(self) -> tuple[int, ...]:
        """Qubits actually read out, resolving the 'measure all' default."""
        return self.measured if self.measured else tuple(range(self.num_qubits))

    @property
    def is_dynamic(self) -> bool:
        """True if the circuit needs the dynamic execution path.

        Triggers on any mid-circuit measurement, reset, classical condition, or
        declared classical bits — so the fast static path is used otherwise.
        """
        if self.num_clbits > 0:
            return True
        return any(
            isinstance(op, (Measure, Reset)) or op.condition is not None
            for op in self.gates
        )

    # --- immutable builders ----------------------------------------------
    def _add(self, name: str, *qubits: int, params: tuple[ParamValue, ...] = ()) -> "Circuit":
        return self._appending(Gate(name, qubits, params))

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

    def t(self, q: int) -> "Circuit":
        return self._add("t", q)

    def rz(self, q: int, theta: ParamValue) -> "Circuit":
        return self._add("rz", q, params=(theta,))

    def rx(self, q: int, theta: ParamValue) -> "Circuit":
        return self._add("rx", q, params=(theta,))

    def ry(self, q: int, theta: ParamValue) -> "Circuit":
        return self._add("ry", q, params=(theta,))

    def swap(self, q0: int, q1: int) -> "Circuit":
        return self._add("swap", q0, q1)

    def ms(self, q0: int, q1: int, theta: ParamValue = 0.25) -> "Circuit":
        return self._add("ms", q0, q1, params=(theta,))

    def measure(self, *qubits: int) -> "Circuit":
        return replace(self, measured=self.measured + qubits)

    # --- dynamic-circuit builders ----------------------------------------
    def _add_op(self, op: Operation) -> "Circuit":
        return self._appending(op)

    def measure_into(self, qubit: int, cbit: int) -> "Circuit":
        """Mid-circuit measurement of ``qubit`` into classical bit ``cbit``."""
        return self._add_op(Measure((qubit,), cbit))

    def reset(self, qubit: int) -> "Circuit":
        """Reset ``qubit`` to |0⟩ mid-circuit."""
        return self._add_op(Reset((qubit,)))

    def gate_if(
        self,
        name: str,
        qubits: tuple[int, ...],
        condition: tuple[tuple[int, int], ...],
        params: tuple[ParamValue, ...] = (),
    ) -> "Circuit":
        """Apply gate ``name`` on ``qubits`` only if every ``(cbit, value)`` in
        ``condition`` holds (a conjunction — e.g. a multi-bit syndrome)."""
        return self._add_op(Gate(name, qubits, params, condition=tuple(condition)))

    def x_if(self, q: int, cbit: int, value: int = 1) -> "Circuit":
        """Classically-controlled X (feed-forward correction) on a single bit."""
        return self.gate_if("x", (q,), ((cbit, value),))

    def z_if(self, q: int, cbit: int, value: int = 1) -> "Circuit":
        """Classically-controlled Z (feed-forward correction) on a single bit."""
        return self.gate_if("z", (q,), ((cbit, value),))

    def x_if_all(self, q: int, condition: tuple[tuple[int, int], ...]) -> "Circuit":
        """Classically-controlled X gated on a conjunction of classical bits."""
        return self.gate_if("x", (q,), condition)

    # --- parameters -------------------------------------------------------
    @property
    def parameters(self) -> tuple[Parameter, ...]:
        """Distinct unbound symbolic parameters, in first-seen order."""
        seen: dict[str, Parameter] = {}
        for g in self.gates:
            for p in g.params:
                if isinstance(p, Parameter) and p.name not in seen:
                    seen[p.name] = p
        return tuple(seen.values())

    def bind(self, values: dict[Parameter | str, float]) -> "Circuit":
        """Return a new circuit with symbolic parameters replaced by numbers.

        ``values`` may key on :class:`Parameter` objects or their names. Raises
        if any symbolic parameter in the circuit is left unbound.
        """
        by_name = {(k.name if isinstance(k, Parameter) else k): float(v)
                   for k, v in values.items()}

        def resolve(p: ParamValue) -> float:
            if isinstance(p, Parameter):
                if p.name not in by_name:
                    raise ValueError(f"unbound parameter: {p.name!r}")
                return by_name[p.name]
            return p

        def rebind(op: Operation) -> Operation:
            if isinstance(op, Gate) and op.params:
                return Gate(op.name, op.qubits,
                            tuple(resolve(p) for p in op.params), op.condition)
            return op

        return replace(self, gates=tuple(rebind(op) for op in self.gates))

    # --- JSON serialization ------------------------------------------------
    def to_json(self) -> str:
        """Serialize to a JSON string (supports dynamic-circuit ops)."""
        ops: list[dict[str, object]] = []
        for op in self.gates:
            d: dict[str, object] = {
                "name": op.name,
                "qubits": list(op.qubits),
                "params": list(op.params),
            }
            if isinstance(op, Measure):
                d["op"] = "measure"
                d["cbit"] = op.cbit
            elif isinstance(op, Reset):
                d["op"] = "reset"
            else:
                d["op"] = "gate"
            if op.condition is not None:
                d["condition"] = [list(c) for c in op.condition]
            ops.append(d)
        data = {
            "num_qubits": self.num_qubits,
            "gates": ops,
            "measured": list(self.measured),
            "num_clbits": self.num_clbits,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Circuit":
        """Deserialize from a JSON string (backward compatible with static circuits)."""
        data = json.loads(text)
        ops: list[Operation] = []
        for g in data["gates"]:
            kind = g.get("op", "gate")
            qubits = tuple(g["qubits"])
            params = tuple(g.get("params", []))
            cond = (tuple((int(c[0]), int(c[1])) for c in g["condition"])
                    if g.get("condition") else None)
            if kind == "measure":
                ops.append(Measure(qubits, int(g["cbit"]), condition=cond))
            elif kind == "reset":
                ops.append(Reset(qubits, condition=cond))
            else:
                ops.append(Gate(g["name"], qubits, params, cond))
        return cls(
            num_qubits=data["num_qubits"],
            gates=tuple(ops),
            measured=tuple(data.get("measured", [])),
            num_clbits=int(data.get("num_clbits", 0)),
        )


# --- gate algebra ---------------------------------------------------------
# Both derived from the gate registry (see qorch.gates), so "is this gate its
# own inverse?" has exactly one answer in the codebase.
_SELF_INVERSE_GATES: frozenset[str] = SELF_INVERSE_GATES
_ANGLE_GATES: frozenset[str] = ANGLE_INVERSE_GATES


def inverse_gates(gate: "Gate") -> list["Gate"]:
    """Return a gate sequence implementing ``gate``'s inverse, in native gates only.

    Used by ZNE unitary folding and any pass needing a true adjoint. Rotations
    negate their angle; ``sx`` (a fourth root of I via sx⁴=I) inverts to sx³;
    ``t`` (t⁸=I) inverts to t⁷; self-inverse gates return themselves.
    """
    name = gate.name
    if name in _SELF_INVERSE_GATES:
        return [gate]
    if name in _ANGLE_GATES:
        theta = float(gate.params[0]) if gate.params else 0.0
        return [Gate(name, gate.qubits, (-theta,))]
    if name == "sx":
        return [Gate("sx", gate.qubits)] * 3
    if name == "t":
        return [Gate("t", gate.qubits)] * 7
    raise ValueError(f"no inverse defined for gate {name!r}")  # pragma: no cover


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
    Dynamic circuits are out of scope for this subset emitter and raise clearly.
    """
    n = circuit.num_qubits
    lines = [
        "OPENQASM 3.0;",
        f"qubit[{n}] q;",
    ]
    for g in static_gates(circuit.gates):
        qstr = ", ".join(f"q[{q}]" for q in g.qubits)
        if g.params:
            pstr = ", ".join(f"{p:.10g}" for p in g.params)
            lines.append(f"{g.name}({pstr}) {qstr};")
        else:
            lines.append(f"{g.name} {qstr};")
    for q in circuit.readout_qubits:
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
