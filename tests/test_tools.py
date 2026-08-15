"""qorch's capabilities as JSON-in / JSON-out tools.

The MCP server is the easy half. What has to be right is the tool layer: every
result must actually serialize, every bad input must come back as a readable
error rather than a traceback, and the limits must hold — a tool endpoint should
not be a way to ask a shared process for an hour of work by typing a bigger
number.

The caller on the other side is usually a program, often a language model. It
cannot read a stack trace and cannot retry intelligently against an opaque
failure, so the error path gets as much attention here as the happy one.
"""

from __future__ import annotations

import json

import pytest

from qorch.tools import TOOLS, call_tool, describe_tools

_BELL = {
    "num_qubits": 2,
    "gates": [{"name": "h", "qubits": [0]}, {"name": "cx", "qubits": [0, 1]}],
    "measured": [0, 1],
}


# ── every tool works and returns JSON ────────────────────────────────────


@pytest.mark.parametrize("name,request_body", [
    ("simulate", {"circuit": _BELL, "shots": 200, "seed": 1}),
    ("analyze", {"circuit": _BELL}),
    ("transpile", {"circuit": _BELL}),
    ("draw", {"circuit": _BELL}),
    ("certify", {"shots": 256, "seed": 2}),
])
def test_every_tool_returns_serializable_success(name: str, request_body) -> None:
    """A tool that promises JSON must produce it, not something that reprs nicely."""
    result = call_tool(name, request_body)
    assert result["ok"] is True
    json.dumps(result)


def test_every_registered_tool_is_described() -> None:
    described = {entry["name"] for entry in describe_tools()}
    assert described == set(TOOLS)
    assert all(entry["description"] for entry in describe_tools())


# ── the tools do the right thing ─────────────────────────────────────────


def test_simulate_returns_bell_counts() -> None:
    result = call_tool("simulate", {"circuit": _BELL, "shots": 500, "seed": 1})
    assert sum(result["counts"].values()) == 500
    assert set(result["counts"]) == {"00", "11"}
    assert pytest.approx(sum(result["probabilities"].values()), abs=1e-9) == 1.0


def test_analyze_reports_structure() -> None:
    result = call_tool("analyze", {"circuit": _BELL})
    assert result["num_qubits"] == 2
    assert result["num_gates"] == 2
    assert result["num_2q_gates"] == 1


def test_transpile_reports_layout_and_per_pass_metrics() -> None:
    result = call_tool("transpile", {"circuit": _BELL, "target": "tifr-superconducting"})
    assert sorted(result["final_layout"]) == [0, 1]
    assert result["metrics"]["input_gates"] == 2
    assert [p["name"] for p in result["metrics"]["passes"]]


def test_transpile_output_is_a_reloadable_circuit() -> None:
    """The returned circuit has to be usable, not merely descriptive."""
    from qorch.ir import Circuit

    result = call_tool("transpile", {"circuit": _BELL})
    reloaded = Circuit.from_json(json.dumps(result["circuit"]))
    assert reloaded.num_qubits == 2


def test_draw_produces_a_diagram() -> None:
    assert call_tool("draw", {"circuit": _BELL})["diagram"].strip()


def test_certify_reports_scalars_not_objects() -> None:
    result = call_tool("certify", {"shots": 512, "seed": 3})
    assert isinstance(result["bell_fidelity"], float)
    assert isinstance(result["chsh_s"], float)
    assert isinstance(result["chsh_violation"], bool)
    assert result["chsh_s"] > 2.0, "an ideal simulator should violate the CHSH bound"


# ── bad input comes back readable ────────────────────────────────────────


def test_an_unknown_tool_lists_the_real_ones() -> None:
    result = call_tool("teleport-me")
    assert result["ok"] is False
    assert "simulate" in result["error"]


@pytest.mark.parametrize("bad,expected", [
    ({}, "num_qubits"),
    ({"gates": []}, "num_qubits"),
    ({"num_qubits": 0}, "between 1"),
    ({"num_qubits": 999}, "between 1"),
    ({"num_qubits": 1, "gates": [{"name": "zzz", "qubits": [0]}]}, "unsupported gate"),
    ({"num_qubits": 1, "gates": [{"name": "h", "qubits": [7]}]}, "out of range"),
])
def test_a_bad_circuit_is_reported_not_raised(bad, expected: str) -> None:
    result = call_tool("simulate", {"circuit": bad})
    assert result["ok"] is False
    assert expected in result["error"]


def test_a_non_object_circuit_is_rejected() -> None:
    assert call_tool("analyze", {"circuit": 42})["ok"] is False


def test_an_unknown_transpile_target_lists_the_options() -> None:
    result = call_tool("transpile", {"circuit": _BELL, "target": "nope"})
    assert result["ok"] is False
    assert "tifr-superconducting" in result["error"]


@pytest.mark.parametrize("shots", [0, -5, 10 ** 9])
def test_shot_counts_are_bounded(shots: int) -> None:
    """Not a validation nicety: an unbounded shot count is a denial of service."""
    result = call_tool("simulate", {"circuit": _BELL, "shots": shots})
    assert result["ok"] is False
    assert "shots" in result["error"]


def test_certify_shot_count_is_bounded_too() -> None:
    assert call_tool("certify", {"shots": 10 ** 9})["ok"] is False


def test_a_missing_request_body_does_not_crash() -> None:
    for name in TOOLS:
        result = call_tool(name)
        assert isinstance(result, dict)
        assert "ok" in result


# ── serialized circuits are accepted too ─────────────────────────────────


def test_a_circuit_can_be_passed_as_qorch_json() -> None:
    """A caller holding a serialized circuit should not have to unpack it."""
    from qorch import Circuit

    circuit = Circuit(2).h(0).cx(0, 1).measure(0, 1)
    result = call_tool("simulate", {"circuit": circuit.to_json(), "shots": 100, "seed": 1})
    assert result["ok"] is True
    assert sum(result["counts"].values()) == 100


def test_seeding_makes_a_tool_call_reproducible() -> None:
    a = call_tool("simulate", {"circuit": _BELL, "shots": 300, "seed": 42})
    b = call_tool("simulate", {"circuit": _BELL, "shots": 300, "seed": 42})
    assert a["counts"] == b["counts"]
