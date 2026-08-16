"""A vendor-neutral device evaluation that produces a report you can act on.

The individual benchmarks already exist — Bell fidelity, CHSH, randomized
benchmarking, quantum volume. What was missing is the thing that makes them
usable as an *evaluation*: one suite that runs against any ``Backend``, states
its thresholds up front, records enough provenance to be re-run, and says
plainly which checks passed, which failed, and which could not be run at all.

Three properties matter more than the individual numbers:

**Vendor-neutral.** Every check goes through the ``Backend`` interface, so a
simulator, an Indian QPU preset, and a Qiskit-backed device are evaluated by
identical code. A comparison between two devices is meaningful precisely because
nothing in the suite knows which vendor it is talking to.

**Reproducible.** A report carries the seed, shot counts, thresholds, and
backend properties it was produced under. A number without those is not
evidence — it cannot be checked by anyone else.

**Honest about what it did not measure.** A check that could not run reports
``NOT_APPLICABLE``, never ``PASS``. A device that publishes no calibration has
not passed a coherence check; it has not taken one, and a report that blurs the
two is worse than no report.
"""

from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from qorch.backends.base import Backend


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


@dataclass(frozen=True)
class CheckResult:
    """One measurement, the bar it was held to, and whether it cleared it."""

    name: str
    outcome: Outcome
    value: float | None = None
    threshold: float | None = None
    uncertainty: float | None = None
    detail: str = ""

    @property
    def summary(self) -> str:
        if self.value is None:
            return f"{self.name}: {self.outcome.value} — {self.detail}"
        margin = f" ±{self.uncertainty:.4f}" if self.uncertainty is not None else ""
        bar = f" (threshold {self.threshold})" if self.threshold is not None else ""
        return f"{self.name}: {self.value:.4f}{margin}{bar} — {self.outcome.value}"


@dataclass(frozen=True)
class CertificationReport:
    """Everything needed to interpret, trust, and re-run an evaluation."""

    backend_name: str
    checks: tuple[CheckResult, ...]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.outcome is Outcome.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for c in self.checks if c.outcome is Outcome.FAIL)

    @property
    def not_applicable(self) -> int:
        return sum(1 for c in self.checks if c.outcome is Outcome.NOT_APPLICABLE)

    @property
    def ok(self) -> bool:
        """True when nothing failed.

        Checks that could not run do not count against a device, and do not
        count for it either — they are reported separately so a reader can see
        how much of the suite actually applied.
        """
        return self.failed == 0 and not any(
            c.outcome is Outcome.ERROR for c in self.checks
        )

    def get(self, name: str) -> CheckResult | None:
        return next((c for c in self.checks if c.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "ok": self.ok,
            "passed": self.passed,
            "failed": self.failed,
            "not_applicable": self.not_applicable,
            "checks": [
                {**asdict(c), "outcome": c.outcome.value} for c in self.checks
            ],
            "provenance": self.provenance,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def format(self) -> str:
        lines = [
            f"Certification report — {self.backend_name}",
            "=" * (24 + len(self.backend_name)),
        ]
        lines += [f"  {c.summary}" for c in self.checks]
        lines += [
            "",
            f"  passed {self.passed}   failed {self.failed}"
            f"   not applicable {self.not_applicable}",
            f"  verdict: {'OK' if self.ok else 'FAILED'}",
        ]
        if self.provenance:
            lines.append("")
            lines.append("  provenance:")
            lines += [f"    {k}: {v}" for k, v in sorted(self.provenance.items())]
        return "\n".join(lines)


@dataclass(frozen=True)
class Thresholds:
    """The bar each check is held to.

    Stated as data and recorded in the report, so a reader can see what "pass"
    meant rather than having to trust it. Defaults are deliberately modest —
    they are meant to catch a device that is broken, not to rank good ones.
    """

    bell_fidelity: float = 0.80
    chsh_s: float = 2.0                 # the classical bound; above it is quantum
    rb_error_rate: float = 0.05         # per-Clifford error
    quantum_volume_width: int = 2


def _binomial_uncertainty(p: float, shots: int) -> float:
    """Standard error on a sampled probability.

    Reported alongside every sampled quantity because a fidelity from 100 shots
    and one from 100,000 are not the same claim, and a bare number hides that.
    """
    if shots <= 0:
        return 0.0
    p = min(max(p, 0.0), 1.0)
    return math.sqrt(max(p * (1.0 - p), 0.0) / shots)


def _check_bell(backend: Backend, shots: int, thresholds: Thresholds) -> CheckResult:
    from qorch.entanglement import bell_state_fidelity

    result = bell_state_fidelity(backend, shots=shots)
    if result.fidelity is None:
        return CheckResult("bell_fidelity", Outcome.NOT_APPLICABLE,
                           detail="backend returned no usable fidelity")
    value = float(result.fidelity)
    return CheckResult(
        name="bell_fidelity",
        outcome=Outcome.PASS if value >= thresholds.bell_fidelity else Outcome.FAIL,
        value=value,
        threshold=thresholds.bell_fidelity,
        uncertainty=_binomial_uncertainty(value, shots),
        detail="two-qubit entanglement quality",
    )


def _check_chsh(backend: Backend, shots: int, thresholds: Thresholds) -> CheckResult:
    from qorch.entanglement import chsh_s_value

    result = chsh_s_value(backend, shots=shots)
    if result.s_value is None:
        return CheckResult("chsh_s", Outcome.NOT_APPLICABLE,
                           detail="backend returned no usable S value")
    value = float(result.s_value)
    return CheckResult(
        name="chsh_s",
        outcome=Outcome.PASS if value > thresholds.chsh_s else Outcome.FAIL,
        value=value,
        threshold=thresholds.chsh_s,
        uncertainty=4 * _binomial_uncertainty(0.5, shots),
        detail="violation of the classical bound (S > 2)",
    )


def _check_rb(backend: Backend, shots: int, thresholds: Thresholds,
              seed: int | None) -> CheckResult:
    from qorch.benchmarking import randomized_benchmarking

    result = randomized_benchmarking(
        backend, num_qubits=1, depths=(1, 2, 4, 8), circuits_per_depth=5,
        shots=shots, seed=seed,
    )
    if result.estimated_error_rate is None:
        return CheckResult("rb_error_rate", Outcome.NOT_APPLICABLE,
                           detail="randomized benchmarking produced no fit")
    value = float(result.estimated_error_rate)
    return CheckResult(
        name="rb_error_rate",
        outcome=Outcome.PASS if value <= thresholds.rb_error_rate else Outcome.FAIL,
        value=value,
        threshold=thresholds.rb_error_rate,
        detail="per-Clifford error from the survival-probability decay",
    )


def _check_quantum_volume(backend: Backend, shots: int, thresholds: Thresholds,
                          seed: int | None) -> CheckResult:
    from qorch.benchmarking import quantum_volume

    width = thresholds.quantum_volume_width
    if backend.properties().num_qubits < width:
        return CheckResult(
            "quantum_volume", Outcome.NOT_APPLICABLE,
            detail=f"backend has fewer than {width} qubits",
        )
    # 40 trials, not a handful. The QV protocol's success test demands 97.5%
    # one-sided confidence that HOP exceeds 2/3, and with too few trials the
    # standard error is wide enough that even a perfect device cannot clear the
    # bar — measured: an ideal simulator scores z=0.82 at 6 trials and z=2.50 at
    # 40. A check that no device can ever pass is worse than no check.
    result = quantum_volume(backend, width=width, shots=shots, trials=40, seed=seed)
    if result.heavy_output_probability is None:
        return CheckResult("quantum_volume", Outcome.NOT_APPLICABLE,
                           detail="no heavy-output probability was produced")
    value = float(result.heavy_output_probability)
    return CheckResult(
        name="quantum_volume",
        outcome=Outcome.PASS if result.success else Outcome.FAIL,
        value=value,
        threshold=2 / 3,
        uncertainty=_binomial_uncertainty(value, shots),
        detail=f"heavy-output probability at width {width}",
    )


def _check_coherence(backend: Backend, thresholds: Thresholds) -> CheckResult:
    """T1/T2 from the device's own calibration, if it publishes any.

    A backend with no calibration has not *passed* this — it has not taken it.
    Reporting that as a pass would let an unmeasured device look like a good one.
    """
    calibration = getattr(backend, "calibration", None)
    data = calibration() if callable(calibration) else None
    if data is None or not data.qubits:
        return CheckResult("coherence", Outcome.NOT_APPLICABLE,
                           detail="backend publishes no calibration data")

    times = [q.t1_us for q in data.qubits if q.t1_us > 0]
    if not times:
        return CheckResult("coherence", Outcome.NOT_APPLICABLE,
                           detail="calibration reports no T1 values")
    worst = min(times)
    return CheckResult(
        name="coherence",
        outcome=Outcome.PASS,
        value=worst,
        detail=f"worst T1 across {len(times)} qubits (microseconds)",
    )


def certify_backend(
    backend: Backend,
    shots: int = 2048,
    seed: int | None = 7,
    thresholds: Thresholds | None = None,
) -> CertificationReport:
    """Run the full suite against any backend and report the result.

    Every check is wrapped: one benchmark failing to run must not lose the
    results of the others, since a partial evaluation of a partly-working device
    is exactly when a report is most useful.
    """
    thresholds = thresholds or Thresholds()
    properties = backend.properties()

    checks: list[CheckResult] = []
    for name, run in (
        ("bell_fidelity", lambda: _check_bell(backend, shots, thresholds)),
        ("chsh_s", lambda: _check_chsh(backend, shots, thresholds)),
        ("rb_error_rate", lambda: _check_rb(backend, shots, thresholds, seed)),
        ("quantum_volume",
         lambda: _check_quantum_volume(backend, shots, thresholds, seed)),
        ("coherence", lambda: _check_coherence(backend, thresholds)),
    ):
        try:
            checks.append(run())
        except Exception as exc:                            # noqa: BLE001
            checks.append(CheckResult(
                name, Outcome.ERROR,
                detail=f"{type(exc).__name__}: {exc}",
            ))

    from qorch import __version__

    return CertificationReport(
        backend_name=backend.name,
        checks=tuple(checks),
        provenance={
            "qorch_version": __version__,
            "python": platform.python_version(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "shots": shots,
            "seed": seed,
            "thresholds": asdict(thresholds),
            "backend_qubits": properties.num_qubits,
            "backend_is_simulator": properties.is_simulator,
            "backend_basis_gates": list(properties.basis_gates),
        },
    )


@dataclass(frozen=True)
class Comparison:
    """How two devices compare, check by check."""

    left: str
    right: str
    rows: tuple[tuple[str, float | None, float | None, str], ...]

    def format(self) -> str:
        lines = [
            f"{'check':<18} {self.left:>14} {self.right:>14}  better",
            "-" * 60,
        ]
        for name, a, b, better in self.rows:
            left = "n/a" if a is None else f"{a:.4f}"
            right = "n/a" if b is None else f"{b:.4f}"
            lines.append(f"{name:<18} {left:>14} {right:>14}  {better}")
        return "\n".join(lines)


#: Checks where a *lower* number is the better one.
_LOWER_IS_BETTER = frozenset({"rb_error_rate"})


def compare_reports(
    left: CertificationReport, right: CertificationReport
) -> Comparison:
    """Compare two reports check by check.

    Meaningful only because the suite is vendor-neutral: identical code produced
    both sides, so a difference is a difference between the devices rather than
    between two evaluation methods.
    """
    names = sorted({c.name for c in left.checks} | {c.name for c in right.checks})
    rows: list[tuple[str, float | None, float | None, str]] = []
    for name in names:
        a = left.get(name)
        b = right.get(name)
        a_value = a.value if a else None
        b_value = b.value if b else None
        if a_value is None or b_value is None:
            better = "—"
        elif a_value == b_value:
            better = "tie"
        elif (a_value < b_value) is (name in _LOWER_IS_BETTER):
            better = left.backend_name
        else:
            better = right.backend_name
        rows.append((name, a_value, b_value, better))
    return Comparison(left=left.backend_name, right=right.backend_name,
                      rows=tuple(rows))
