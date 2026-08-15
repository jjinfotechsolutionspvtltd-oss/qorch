"""qorch's capabilities as JSON-in / JSON-out tools.

Roadmap 6.1 asks for an MCP server exposing simulate / analyze / transpile /
certify / draw. The server is the easy half and the uninteresting one: what
actually has to be right is the *tool layer* — a set of operations with
serializable inputs and outputs, validated at the boundary, that never raise a
traceback at a caller who cannot see the stack.

So the tools live here as ordinary functions and the protocol binding is a thin
wrapper over them. That ordering matters for two reasons: the substance stays
testable without installing an SDK or standing up a server, and the MCP package
stays an optional extra rather than something the dependency-free core needs.

Every tool takes a plain dict and returns a plain dict. Errors come back as
``{"ok": False, "error": ...}`` rather than as exceptions, because the caller is
generally a program on the other side of a protocol boundary — often a language
model — and an exception there is an opaque failure, not a diagnostic.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from qorch.analysis import circuit_report
from qorch.ir import Circuit
from qorch.visual import draw_circuit

_MAX_QUBITS = 20
_MAX_SHOTS = 100_000


def _fail(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _ok(**payload: Any) -> dict[str, Any]:
    return {"ok": True, **payload}


def _parse_circuit(spec: Any) -> Circuit:
    """Build a Circuit from a JSON spec, rejecting anything unusable.

    Accepts either qorch's own JSON serialization or a compact
    ``{"num_qubits": n, "gates": [{"name": ..., "qubits": [...]}, ...]}`` form,
    since a caller composing a request by hand should not have to know the
    internal encoding.
    """
    if isinstance(spec, str):
        return Circuit.from_json(spec)
    if not isinstance(spec, dict):
        raise ValueError("circuit must be a JSON object or a serialized string")
    if "num_qubits" not in spec:
        raise ValueError("circuit needs a 'num_qubits' field")

    num_qubits = int(spec["num_qubits"])
    if not 0 < num_qubits <= _MAX_QUBITS:
        raise ValueError(f"num_qubits must be between 1 and {_MAX_QUBITS}")

    circuit = Circuit(num_qubits, num_clbits=int(spec.get("num_clbits", 0)))
    for entry in spec.get("gates", ()):
        name = entry["name"]
        qubits = tuple(int(q) for q in entry.get("qubits", ()))
        params = tuple(float(p) for p in entry.get("params", ()))
        circuit = circuit._add(name, *qubits, params=params)
    measured = spec.get("measured")
    if measured:
        circuit = circuit.measure(*(int(q) for q in measured))
    return circuit


def _guard(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Turn any failure into a structured error.

    A tool call crossing a protocol boundary cannot usefully raise: the caller
    sees a dead connection or an opaque stack trace it cannot act on. A named
    error it can read back is worth more than a correct exception it cannot.
    """
    def run(request: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            return fn(request or {})
        except (ValueError, KeyError, TypeError) as exc:
            return _fail(f"{type(exc).__name__}: {exc}")
    run.__name__ = fn.__name__
    run.__doc__ = fn.__doc__
    return run


@_guard
def simulate(request: dict[str, Any]) -> dict[str, Any]:
    """Run a circuit and return measurement counts.

    Shots are capped: a tool endpoint should not be a way to ask a shared
    process for an hour of work by typing a bigger number.
    """
    from qorch import LocalSimulator

    circuit = _parse_circuit(request.get("circuit"))
    shots = int(request.get("shots", 1024))
    if not 0 < shots <= _MAX_SHOTS:
        return _fail(f"shots must be between 1 and {_MAX_SHOTS}")

    seed = request.get("seed")
    result = LocalSimulator(seed=None if seed is None else int(seed)).run(circuit, shots)
    return _ok(
        counts=result.counts,
        shots=result.shots,
        backend=result.backend_name,
        probabilities=result.probabilities,
    )


@_guard
def analyze(request: dict[str, Any]) -> dict[str, Any]:
    """Report depth, gate counts and an estimated fidelity for a circuit."""
    circuit = _parse_circuit(request.get("circuit"))
    report = circuit_report(circuit, float(request.get("gate_error_rate", 0.01)))
    return _ok(**report)


@_guard
def transpile(request: dict[str, Any]) -> dict[str, Any]:
    """Compile a circuit for a named target, reporting layout and per-pass cost."""
    from qorch.transpiler import (
        CouplingMap,
        transpile_with_layout,
    )
    from qorch.transpiler.gateset import (
        CLIFFORD_T,
        DRDO_MIRAI,
        IIT_JODHPUR_ION_TRAP,
        TIFR_SUPERCONDUCTING,
    )

    targets = {
        "iit-jodhpur-ion-trap": IIT_JODHPUR_ION_TRAP,
        "tifr-superconducting": TIFR_SUPERCONDUCTING,
        "drdo-mirai": DRDO_MIRAI,
        "clifford-t": CLIFFORD_T,
    }
    name = request.get("target", "tifr-superconducting")
    if name not in targets:
        return _fail(f"unknown target {name!r}; options: {sorted(targets)}")
    target = targets[name]

    circuit = _parse_circuit(request.get("circuit"))
    coupling = CouplingMap(target.coupling_map) if target.coupling_map else None
    result = transpile_with_layout(
        circuit, target, coupling_map=coupling,
        layout_method=request.get("layout_method", "trivial"),
    )
    metrics = result.metrics
    return _ok(
        circuit=json.loads(result.circuit.to_json()),
        final_layout=list(result.final_layout),
        target=name,
        metrics={
            "input_gates": metrics.input_gate_count,
            "output_gates": metrics.output_gate_count,
            "two_qubit_gates": metrics.two_qubit_count,
            "swaps_inserted": metrics.swaps_inserted,
            "depth": metrics.depth,
            "passes": [
                {"name": p.name, "gates": p.gate_count, "depth": p.depth}
                for p in metrics.passes
            ],
        } if metrics else None,
    )


@_guard
def draw(request: dict[str, Any]) -> dict[str, Any]:
    """Render a circuit as ASCII art."""
    circuit = _parse_circuit(request.get("circuit"))
    return _ok(diagram=draw_circuit(circuit))


@_guard
def certify(request: dict[str, Any]) -> dict[str, Any]:
    """Run the certification suite against the local simulator."""
    from qorch import LocalSimulator
    from qorch.entanglement import bell_state_fidelity, chsh_s_value

    seed = request.get("seed")
    backend = LocalSimulator(seed=None if seed is None else int(seed))
    shots = int(request.get("shots", 2048))
    if not 0 < shots <= _MAX_SHOTS:
        return _fail(f"shots must be between 1 and {_MAX_SHOTS}")

    # Unpack to scalars: these helpers return dataclasses, and a tool that
    # promises JSON has to actually produce it rather than something that
    # happens to repr nicely.
    #
    # The fields are optional, and a missing measurement is reported as null
    # rather than coerced to a number. Inventing 0.0 for "no result" would be
    # indistinguishable from a genuinely terrible device.
    bell = bell_state_fidelity(backend, shots=shots)
    chsh = chsh_s_value(backend, shots=shots)
    return _ok(
        bell_fidelity=None if bell.fidelity is None else float(bell.fidelity),
        chsh_s=None if chsh.s_value is None else float(chsh.s_value),
        chsh_violation=None if chsh.violation is None else bool(chsh.violation),
        backend=backend.name,
        shots=shots,
    )


#: Every tool, by the name a protocol binding should expose it under.
TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "simulate": simulate,
    "analyze": analyze,
    "transpile": transpile,
    "draw": draw,
    "certify": certify,
}


def describe_tools() -> list[dict[str, str]]:
    """Name and one-line description of each tool, for a server's manifest."""
    return [
        {"name": name, "description": (fn.__doc__ or "").strip().split("\n")[0]}
        for name, fn in sorted(TOOLS.items())
    ]


def call_tool(name: str, request: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch by name, reporting an unknown tool the same way as any other error."""
    tool = TOOLS.get(name)
    if tool is None:
        return _fail(f"unknown tool {name!r}; available: {sorted(TOOLS)}")
    return tool(request or {})
