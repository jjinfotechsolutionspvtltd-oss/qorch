"""Symbolic parameters: build once, bind many times (variational loops)."""

from __future__ import annotations

import math

from qorch import Circuit, LocalSimulator, Parameter


def test_circuit_reports_parameters():
    theta = Parameter("theta")
    phi = Parameter("phi")
    c = Circuit(2).rx(0, theta).rz(1, phi).rx(0, theta)
    names = {p.name for p in c.parameters}
    assert names == {"theta", "phi"}  # distinct, deduplicated


def test_bind_by_object_and_name():
    theta = Parameter("theta")
    c = Circuit(1).rx(0, theta).measure(0)
    bound_obj = c.bind({theta: math.pi})
    bound_name = c.bind({"theta": math.pi})
    for bound in (bound_obj, bound_name):
        assert bound.parameters == ()  # fully bound
        counts = LocalSimulator(seed=1).run(bound, shots=400).counts
        assert counts.get("1", 0) > 380  # rx(pi) ~ X


def test_bind_unbound_raises():
    theta = Parameter("theta")
    c = Circuit(1).rx(0, theta)
    try:
        c.bind({})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "theta" in str(e)


def test_parametrized_sweep_matches_direct():
    """Binding a template equals building each angle directly (the VQE pattern)."""
    theta = Parameter("t")
    template = Circuit(1).ry(0, theta).measure(0)
    for angle in (0.0, math.pi / 2, math.pi):
        bound = template.bind({theta: angle})
        direct = Circuit(1).ry(0, angle).measure(0)
        b = LocalSimulator(seed=2).run(bound, shots=2000).counts
        d = LocalSimulator(seed=2).run(direct, shots=2000).counts
        assert b == d
