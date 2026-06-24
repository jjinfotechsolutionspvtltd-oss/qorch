"""Dynamic circuits: mid-circuit measurement, classical control, reset.

These exercise the Phase-2 IR + LocalSimulator dynamic execution path.
"""

from __future__ import annotations

from qorch import Circuit, LocalSimulator


def _tvd(a, b, shots):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys) / shots


def test_is_dynamic_flag():
    assert not Circuit(2).h(0).cx(0, 1).is_dynamic
    assert Circuit(2, num_clbits=1).h(0).measure_into(0, 0).is_dynamic
    assert Circuit(1, num_clbits=1).measure_into(0, 0).x_if(0, 0).is_dynamic


def test_mid_circuit_measure_collapses_and_records():
    """Measuring |+> yields ~50/50 in the classical register."""
    c = Circuit(1, num_clbits=1).h(0).measure_into(0, 0)
    counts = LocalSimulator(seed=1).run(c, shots=4000).counts
    assert set(counts) == {"0", "1"}
    assert abs(counts["0"] - counts["1"]) < 400  # ~balanced


def test_deferred_measurement_equivalence():
    """Mid-circuit measurement of a Bell pair == terminal measurement distribution."""
    dynamic = Circuit(2, num_clbits=2).h(0).cx(0, 1).measure_into(0, 0).measure_into(1, 1)
    static = Circuit(2).h(0).cx(0, 1).measure(0, 1)
    d = LocalSimulator(seed=7).run(dynamic, shots=4000).counts
    s = LocalSimulator(seed=7).run(static, shots=4000).counts
    assert set(d) <= {"00", "11"}
    assert _tvd(d, s, 4000) < 0.05


def test_classical_conditioned_gate_feed_forward():
    """If the measured bit is 1, a conditioned X flips the target deterministically."""
    # qubit0 prepared |1>, measured into c0; conditioned X on qubit1 if c0==1.
    c = (Circuit(2, num_clbits=2)
         .x(0)
         .measure_into(0, 0)
         .x_if(1, 0, 1)       # apply X to qubit 1 iff c0 == 1
         .measure_into(1, 1))
    counts = LocalSimulator(seed=2).run(c, shots=1000).counts
    # c0 always 1 (qubit0 was |1>), so qubit1 always flipped → c1 always 1
    assert counts == {"11": 1000}


def test_conditioned_gate_not_applied_when_bit_zero():
    c = (Circuit(2, num_clbits=2)
         .measure_into(0, 0)   # qubit0 is |0> → c0 = 0
         .x_if(1, 0, 1)        # condition false → no flip
         .measure_into(1, 1))
    counts = LocalSimulator(seed=2).run(c, shots=1000).counts
    assert counts == {"00": 1000}


def test_reset_returns_qubit_to_zero():
    c = Circuit(1, num_clbits=1).x(0).reset(0).measure_into(0, 0)
    counts = LocalSimulator(seed=3).run(c, shots=1000).counts
    assert counts == {"0": 1000}


def test_repeated_measure_is_deterministic_after_collapse():
    """Measuring the same qubit twice gives identical bits (state collapsed)."""
    c = Circuit(1, num_clbits=2).h(0).measure_into(0, 0).measure_into(0, 1)
    counts = LocalSimulator(seed=4).run(c, shots=2000).counts
    # second measurement must equal the first → only "00" and "11"
    assert set(counts) <= {"00", "11"}
