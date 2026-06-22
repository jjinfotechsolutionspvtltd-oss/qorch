"""Tests for the CLI interface."""

from qorch.cli import _parse_gates, _build_backend, cmd_list


def test_parse_gates_h_cx():
    circuit = _parse_gates("h0,cx01")
    assert len(circuit.gates) == 2
    assert circuit.gates[0].name == "h"
    assert circuit.gates[1].name == "cx"


def test_build_backend_local_simulator():
    backend = _build_backend("local-simulator")
    assert backend.name == "local-simulator"


def test_build_backend_indian_qpu():
    backend = _build_backend("tifr-superconducting")
    assert "TIFR" in backend.name or "Indian" in backend.name


def test_build_backend_unknown_raises():
    try:
        _build_backend("nonexistent")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_cmd_list_output(capsys):
    cmd_list(None)
    captured = capsys.readouterr()
    assert "local-simulator" in captured.out
    assert "tifr-superconducting" in captured.out
    assert "readout" in captured.out
    assert "zne" in captured.out

