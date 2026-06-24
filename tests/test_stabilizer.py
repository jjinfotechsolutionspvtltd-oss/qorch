"""CHP stabilizer simulator: correctness vs statevector, scaling, dynamic support."""

from __future__ import annotations

import pytest

from qorch import Circuit, LocalSimulator
from qorch.backends.stabilizer import StabilizerSimulator


def _tvd(a, b, shots):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / shots


@pytest.mark.parametrize("circuit", [
    Circuit(2).h(0).cx(0, 1).measure(0, 1),
    Circuit(3).h(0).cx(0, 1).cx(1, 2).measure(0, 1, 2),
    Circuit(2).x(0).h(1).z(1).cx(0, 1).measure(0, 1),
    Circuit(2).x(0).swap(0, 1).measure(0, 1),
    Circuit(1).sx(0).sx(0).measure(0),
])
def test_matches_statevector_on_clifford(circuit):
    a = LocalSimulator(seed=1).run(circuit, 4000).counts
    b = StabilizerSimulator(seed=2).run(circuit, 4000).counts
    assert _tvd(a, b, 4000) < 0.05


def test_rejects_non_clifford():
    sim = StabilizerSimulator(seed=0)
    with pytest.raises(ValueError, match="not Clifford"):
        sim.run(Circuit(1).t(0).measure(0), shots=8)
    with pytest.raises(ValueError, match="not Clifford"):
        sim.run(Circuit(1).rz(0, 0.3).measure(0), shots=8)


def test_scales_beyond_statevector():
    """A 40-qubit GHZ is intractable for statevector but trivial for the tableau."""
    n = 40
    c = Circuit(n).h(0)
    for i in range(n - 1):
        c = c.cx(i, i + 1)
    c = c.measure(*range(n))
    counts = StabilizerSimulator(seed=1).run(c, shots=100).counts
    assert set(counts) <= {"0" * n, "1" * n}  # perfect GHZ correlation


def test_supports_dynamic_feed_forward():
    """Mid-circuit measurement + classical control works on the tableau sim."""
    c = (Circuit(2, num_clbits=2)
         .x(0).measure_into(0, 0)
         .x_if(1, 0, 1)
         .measure_into(1, 1))
    counts = StabilizerSimulator(seed=3).run(c, shots=500).counts
    assert counts == {"11": 500}


def test_reset_on_tableau():
    c = Circuit(1, num_clbits=1).x(0).reset(0).measure_into(0, 0)
    assert StabilizerSimulator(seed=1).run(c, shots=200).counts == {"0": 200}
