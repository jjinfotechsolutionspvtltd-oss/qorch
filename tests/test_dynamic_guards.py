"""Static-only stages must reject dynamic circuits with a clear error."""

from __future__ import annotations

import pytest

from qorch import Circuit, DensitySimulator, to_qasm3
from qorch.qmi import to_qmi
from qorch.transpiler import TIFR_SUPERCONDUCTING, decompose, optimize, transpile


def _dynamic() -> Circuit:
    return Circuit(2, num_clbits=1).h(0).measure_into(0, 0).x_if(1, 0, 1)


@pytest.mark.parametrize("fn", [
    lambda c: decompose(c, TIFR_SUPERCONDUCTING),
    optimize,
    lambda c: transpile(c, TIFR_SUPERCONDUCTING),
    to_qmi,
    to_qasm3,
    lambda c: DensitySimulator().run(c, shots=8),
])
def test_static_stage_rejects_dynamic_circuit(fn):
    with pytest.raises(ValueError, match="static circuit"):
        fn(_dynamic())


def test_static_circuit_still_works_everywhere():
    """Sanity: the guards don't affect ordinary static circuits."""
    c = Circuit(2).h(0).cx(0, 1).measure(0, 1)
    assert len(decompose(c, TIFR_SUPERCONDUCTING).gates) >= 1
    assert "OPENQASM" in to_qasm3(c)
    assert isinstance(to_qmi(c), bytes)
