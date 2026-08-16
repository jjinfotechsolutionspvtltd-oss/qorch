"""Device evaluation as a product: reproducible, vendor-neutral, honest.

The individual benchmarks already existed. What makes them an *evaluation* is
the surrounding contract — explicit thresholds, recorded provenance, and a
report that distinguishes "passed" from "could not be measured".

That last distinction carries the most weight here. A device publishing no
calibration has not passed a coherence check; it has not taken one, and a suite
that blurs those two is worse than no suite.
"""

from __future__ import annotations

import json

import pytest

from qorch import IndianQPU, LocalSimulator
from qorch.backends.simulator import GateNoise
from qorch.benchmarking import _heavy_outputs, quantum_volume
from qorch.certification import (
    Outcome,
    Thresholds,
    certify_backend,
    compare_reports,
)

_WIDE = Thresholds(quantum_volume_width=3)


# ── the quantum-volume fix ───────────────────────────────────────────────


def test_heavy_outputs_are_the_above_median_half() -> None:
    """The QV definition: heavy means high *probability*, not high bit value.

    The previous implementation returned every bitstring whose integer value was
    at least 2^(n-1). That samples an arbitrary half of the outcomes, which is
    why an ideal device scored ~0.35-0.5 and could never pass.
    """
    probabilities = {0: 0.10, 1: 0.40, 2: 0.05, 3: 0.45}
    assert _heavy_outputs(probabilities) == {1, 3}


def test_heavy_outputs_ignore_bitstring_ordering() -> None:
    """Index 0 can be heavy and index 3 light — the old rule had it backwards.

    Under the previous "integer value >= 2^(n-1)" rule this would have returned
    {2, 3}: exactly the two *lightest* outcomes here.
    """
    probabilities = {0: 0.50, 1: 0.40, 2: 0.06, 3: 0.04}
    assert _heavy_outputs(probabilities) == {0, 1}


def test_heavy_outputs_of_nothing_is_empty() -> None:
    assert _heavy_outputs(None) == set()
    assert _heavy_outputs({}) == set()


def test_an_ideal_device_reaches_the_theoretical_heavy_output_probability() -> None:
    """Porter-Thomas gives HOP → (1+ln2)/2 ≈ 0.847 for an ideal device.

    This is the check that the model circuits are actually random enough: a
    fixed circuit shape with a couple of free angles does not produce that
    distribution however many trials you run.
    """
    result = quantum_volume(LocalSimulator(seed=1), width=3, shots=1024,
                            trials=30, seed=3)
    assert 0.78 < result.heavy_output_probability < 0.90


def test_noise_degrades_heavy_output_probability() -> None:
    clean = quantum_volume(LocalSimulator(seed=1), width=3, shots=512,
                           trials=15, seed=3).heavy_output_probability
    noisy = quantum_volume(
        LocalSimulator(seed=1, gate_noise=GateNoise(depolarizing_prob=0.15)),
        width=3, shots=512, trials=15, seed=3,
    ).heavy_output_probability
    assert noisy < clean


def test_an_ideal_device_passes_quantum_volume() -> None:
    """It could not before the fix, at any trial count."""
    result = quantum_volume(LocalSimulator(seed=1), width=3, shots=1024,
                            trials=40, seed=3)
    assert result.success


# ── the report ───────────────────────────────────────────────────────────


def test_an_ideal_simulator_passes_the_suite() -> None:
    report = certify_backend(LocalSimulator(seed=1), shots=1024, seed=3,
                             thresholds=_WIDE)
    assert report.ok
    assert report.failed == 0
    assert report.get("bell_fidelity").outcome is Outcome.PASS
    assert report.get("chsh_s").outcome is Outcome.PASS


def test_a_noisy_device_fails_and_says_which_checks() -> None:
    noisy = LocalSimulator(seed=1, gate_noise=GateNoise(depolarizing_prob=0.25))
    report = certify_backend(noisy, shots=512, seed=3)

    assert not report.ok
    assert report.failed > 0
    assert any(c.outcome is Outcome.FAIL for c in report.checks)


def test_unmeasurable_checks_are_not_passes() -> None:
    """The distinction the whole report rests on.

    A simulator publishes no calibration, so it has not taken the coherence
    check. Recording that as a pass would let an unmeasured device look good.
    """
    report = certify_backend(LocalSimulator(seed=1), shots=256, seed=3)
    coherence = report.get("coherence")

    assert coherence.outcome is Outcome.NOT_APPLICABLE
    assert coherence.outcome is not Outcome.PASS
    assert report.not_applicable >= 1


def test_a_calibrated_device_does_take_the_coherence_check() -> None:
    report = certify_backend(IndianQPU.from_preset("tifr-superconducting", seed=1),
                             shots=256, seed=3)
    assert report.get("coherence").outcome is Outcome.PASS
    assert report.get("coherence").value == pytest.approx(50.0)


def test_sampled_values_carry_their_uncertainty() -> None:
    """A fidelity from 100 shots and one from 100,000 are different claims."""
    few = certify_backend(LocalSimulator(seed=1), shots=128, seed=3)
    many = certify_backend(LocalSimulator(seed=1), shots=4096, seed=3)
    assert few.get("chsh_s").uncertainty > many.get("chsh_s").uncertainty


# ── reproducibility ──────────────────────────────────────────────────────


def test_the_report_records_what_it_was_run_under() -> None:
    """A number without its provenance is not evidence — nobody else can check it."""
    report = certify_backend(LocalSimulator(seed=1), shots=256, seed=11)
    provenance = report.provenance

    assert provenance["shots"] == 256
    assert provenance["seed"] == 11
    assert provenance["qorch_version"]
    assert provenance["timestamp_utc"].endswith("Z")
    assert provenance["thresholds"]["bell_fidelity"] == 0.80


def test_thresholds_are_recorded_so_pass_means_something() -> None:
    custom = Thresholds(bell_fidelity=0.99)
    report = certify_backend(LocalSimulator(seed=1), shots=256, seed=3,
                             thresholds=custom)
    assert report.provenance["thresholds"]["bell_fidelity"] == 0.99
    assert report.get("bell_fidelity").threshold == 0.99


def test_a_seeded_run_is_reproducible() -> None:
    a = certify_backend(LocalSimulator(seed=1), shots=256, seed=5)
    b = certify_backend(LocalSimulator(seed=1), shots=256, seed=5)
    assert [c.value for c in a.checks] == [c.value for c in b.checks]


def test_the_report_serializes() -> None:
    report = certify_backend(LocalSimulator(seed=1), shots=256, seed=3)
    revived = json.loads(report.to_json())

    assert revived["backend_name"] == "local-simulator"
    assert len(revived["checks"]) == len(report.checks)
    assert revived["provenance"]["shots"] == 256


def test_the_report_formats_readably() -> None:
    text = certify_backend(LocalSimulator(seed=1), shots=256, seed=3).format()
    for fragment in ("Certification report", "bell_fidelity", "verdict", "provenance"):
        assert fragment in text


# ── resilience ───────────────────────────────────────────────────────────


def test_one_broken_check_does_not_lose_the_others() -> None:
    """A partial evaluation of a partly-working device is when this matters most."""
    class Flaky(LocalSimulator):
        def run(self, circuit, shots=1024):
            if circuit.num_qubits >= 3:
                raise RuntimeError("device fell over")
            return super().run(circuit, shots)

    report = certify_backend(Flaky(seed=1), shots=256, seed=3, thresholds=_WIDE)
    assert any(c.outcome is Outcome.ERROR for c in report.checks)
    assert any(c.outcome is Outcome.PASS for c in report.checks)
    assert not report.ok


def test_a_device_too_small_for_a_check_reports_not_applicable() -> None:
    class Tiny(LocalSimulator):
        def properties(self):
            base = super().properties()
            return type(base)(num_qubits=1, basis_gates=base.basis_gates,
                              is_simulator=True, readout_fidelity=())

    report = certify_backend(Tiny(seed=1), shots=128, seed=3, thresholds=_WIDE)
    assert report.get("quantum_volume").outcome is Outcome.NOT_APPLICABLE


# ── comparison ───────────────────────────────────────────────────────────


def test_two_devices_compare_check_by_check() -> None:
    """Meaningful only because identical code evaluated both sides."""
    ideal = certify_backend(LocalSimulator(seed=1), shots=512, seed=3)
    noisy = certify_backend(
        LocalSimulator(seed=1, gate_noise=GateNoise(depolarizing_prob=0.2)),
        shots=512, seed=3,
    )
    comparison = compare_reports(ideal, noisy)
    by_name = {row[0]: row for row in comparison.rows}

    assert by_name["bell_fidelity"][3] == ideal.backend_name


def test_comparison_knows_which_metrics_improve_downward() -> None:
    """Lower error is better; higher fidelity is better. Confusing them inverts it."""
    good = certify_backend(LocalSimulator(seed=1), shots=512, seed=3)
    bad = certify_backend(
        LocalSimulator(seed=1, gate_noise=GateNoise(depolarizing_prob=0.2)),
        shots=512, seed=3,
    )
    row = {r[0]: r for r in compare_reports(good, bad).rows}["rb_error_rate"]
    assert row[3] == good.backend_name


def test_a_missing_check_compares_as_unknown() -> None:
    simulator = certify_backend(LocalSimulator(seed=1), shots=256, seed=3)
    qpu = certify_backend(IndianQPU.from_preset("tifr-superconducting", seed=1),
                          shots=256, seed=3)
    row = {r[0]: r for r in compare_reports(simulator, qpu).rows}["coherence"]
    assert row[3] == "—"


def test_comparison_formats_readably() -> None:
    report = certify_backend(LocalSimulator(seed=1), shots=256, seed=3)
    text = compare_reports(report, report).format()
    assert "bell_fidelity" in text
    assert "tie" in text
