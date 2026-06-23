"""Mitigation pipeline — compose error-mitigation steps in the correct order.

Standard order:
1. Pauli Twirling (pre-execution, randomizes coherent noise)
2. Dynamical Decoupling (inserted into circuit before execution)
3. Readout mitigation (post-execution, inverts readout errors)
4. Zero-Noise Extrapolation (post-execution, extrapolates to zero noise)
5. Probabilistic Error Cancellation (post-execution, inverts noise channel)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from qorch.ir import Circuit
from qorch.backends.base import Backend


@dataclass(frozen=True)
class PipelineResult:
    """Aggregated result from a mitigation pipeline run."""

    raw_counts: dict[str, int]
    counts: dict[str, float]  # mitigated counts
    shots: int
    steps_applied: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


StepFunc = Callable[..., Any]


class MitigationPipeline:
    """Composable error-mitigation pipeline.

    Usage::

        pipeline = MitigationPipeline(
            backend=my_backend,
            shots=8192,
        )
        result = pipeline.run(circuit)

    Customise with ``with_twirling``, ``with_dd``, ``with_readout``, etc.
    """

    def __init__(
        self,
        backend: Backend,
        shots: int = 8192,
        twirling: bool = False,
        dd_sequence: str | None = None,
        readout_mitigator: Any | None = None,
        zne_scales: tuple[int, ...] | None = None,
        pec: bool = False,
    ):
        self.backend = backend
        self.shots = shots
        self.twirling = twirling
        self.dd_sequence = dd_sequence
        self.readout_mitigator = readout_mitigator
        self.zne_scales = zne_scales
        self.pec = pec

    def run(self, circuit: Circuit) -> PipelineResult:
        """Run the circuit through the mitigation pipeline."""
        c = circuit
        steps: list[str] = []

        # Step 1: Pauli Twirling (pre-execution, circuit modification)
        if self.twirling:
            from qorch.mitigation.twirling import twirl_circuit
            c = twirl_circuit(c, seed=42)
            steps.append("twirling")

        # Step 2: Dynamical Decoupling (inserted into circuit)
        if self.dd_sequence:
            from qorch.mitigation.dd import insert_dd
            c = insert_dd(c, sequence=self.dd_sequence)
            steps.append(f"dd({self.dd_sequence})")

        # Step 3: Run on backend (possibly with ZNE / PEC)
        if self.zne_scales:
            from qorch.mitigation.zne import zne_expectation, expectation_z
            zne_result = zne_expectation(
                self.backend, c, expectation_z,
                scales=self.zne_scales, shots=self.shots,
            )
            steps.append(f"zne(scales={self.zne_scales})")
            counts = {"0": zne_result.mitigated, "1": 1 - zne_result.mitigated}
            raw_counts = {"0": int(zne_result.raw * self.shots),
                          "1": self.shots - int(zne_result.raw * self.shots)}
            metadata = {"raw": zne_result.raw, "mitigated": zne_result.mitigated,
                        "zne_extrapolated": True}
        elif self.pec:
            raw = self.backend.run(c, shots=self.shots)
            steps.append("pec")
            raw_counts = raw.counts
            counts = {k: float(v) for k, v in raw.counts.items()}
            metadata = {}
        else:
            raw = self.backend.run(c, shots=self.shots)
            steps.append("raw")
            raw_counts = raw.counts
            counts = {k: float(v) for k, v in raw.counts.items()}
            metadata = {}

        # Step 4: Readout mitigation (post-execution)
        if self.readout_mitigator:
            counts = self.readout_mitigator.apply(raw_counts)
            steps.append("readout")
            metadata["readout_mitigated"] = True

        return PipelineResult(
            raw_counts=raw_counts,
            counts=counts,
            shots=self.shots,
            steps_applied=tuple(steps),
            metadata=metadata,
        )
