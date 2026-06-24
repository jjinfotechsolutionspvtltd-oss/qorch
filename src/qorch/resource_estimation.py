"""Fault-tolerant resource estimation from a circuit's Clifford+T cost.

Converts a logical circuit into a first-order estimate of the physical resources
a surface-code machine would need: code distance, physical qubit count, magic-state
distillation overhead, and wall-clock runtime. Builds directly on the Clifford+T
decomposition pass (T-count / T-depth), which already computes the expensive part.

Formulas follow the standard surface-code model (Fowler et al. 2012):
  - logical error per qubit per cycle  p_L(d) ≈ 0.1 (p_phys / p_th)^((d+1)/2)
  - physical qubits per logical qubit  ≈ 2 d²   (rotated patch + routing overhead)
  - one logical cycle                  ≈ d surface-code rounds
These are order-of-magnitude estimates, not a substitute for a full resource compiler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qorch.ir import Circuit

# Surface-code threshold (below this physical error, increasing d helps).
_THRESHOLD = 0.01
# Prefactor in the logical-error formula.
_PREFACTOR = 0.1


@dataclass(frozen=True)
class ResourceEstimate:
    """First-order fault-tolerant resource estimate for a circuit."""

    algorithm_qubits: int
    t_count: int
    t_depth: int
    code_distance: int
    logical_error_per_cycle: float
    physical_qubits: int
    runtime_seconds: float
    target_logical_error: float
    physical_error_rate: float


def _logical_error(distance: int, physical_error_rate: float) -> float:
    """Logical error per logical qubit per cycle for a surface code of given distance."""
    if physical_error_rate >= _THRESHOLD:
        return 1.0  # at/above threshold the code does not suppress errors
    ratio = physical_error_rate / _THRESHOLD
    return _PREFACTOR * ratio ** ((distance + 1) / 2.0)


def _choose_distance(
    physical_error_rate: float,
    target_per_cycle: float,
    max_distance: int = 51,
) -> int:
    """Smallest odd distance whose per-cycle logical error is below the target."""
    d = 3
    while d <= max_distance:
        if _logical_error(d, physical_error_rate) <= target_per_cycle:
            return d
        d += 2
    return max_distance


def estimate_resources(
    circuit: Circuit,
    physical_error_rate: float = 1e-3,
    target_logical_error: float = 1e-2,
    cycle_time_us: float = 1.0,
    distillation_overhead: float = 1.5,
) -> ResourceEstimate:
    """Estimate the fault-tolerant cost of running ``circuit``.

    Args:
        circuit: The logical circuit.
        physical_error_rate: Per-operation physical error of the hardware.
        target_logical_error: Acceptable total logical failure probability.
        cycle_time_us: Duration of one surface-code cycle (microseconds).
        distillation_overhead: Multiplier on qubit count for magic-state factories.

    Returns:
        A :class:`ResourceEstimate`. Requires ``physical_error_rate`` below the
        ~1% surface-code threshold, else no finite distance suffices.
    """
    from qorch.transpiler.decompose import decompose_to_clifford_t

    _, t_count, t_depth = decompose_to_clifford_t(circuit)
    n = circuit.num_qubits

    # Total "logical operations" that each must stay coherent: spread the error
    # budget across (qubits × T-depth) logical-cycle slots.
    logical_slots = max(1, n * max(1, t_depth))
    target_per_cycle = target_logical_error / logical_slots

    distance = _choose_distance(physical_error_rate, target_per_cycle)
    p_l = _logical_error(distance, physical_error_rate)

    # Physical qubits: ~2 d² per logical qubit, plus distillation overhead.
    phys_per_logical = 2 * distance * distance
    physical_qubits = int(math.ceil(n * phys_per_logical * distillation_overhead))

    # Runtime: T-depth sets the sequential magic-state consumption; each logical
    # cycle is ~d rounds of cycle_time. Clifford layers are comparatively cheap.
    rounds = max(1, t_depth) * distance
    runtime_seconds = rounds * cycle_time_us * 1e-6

    return ResourceEstimate(
        algorithm_qubits=n,
        t_count=t_count,
        t_depth=t_depth,
        code_distance=distance,
        logical_error_per_cycle=p_l,
        physical_qubits=physical_qubits,
        runtime_seconds=runtime_seconds,
        target_logical_error=target_logical_error,
        physical_error_rate=physical_error_rate,
    )


def format_estimate(est: ResourceEstimate) -> str:
    """Human-readable resource-estimate report."""
    return "\n".join([
        "Fault-Tolerant Resource Estimate",
        "================================",
        f"  Algorithm qubits:     {est.algorithm_qubits}",
        f"  T-count:              {est.t_count}",
        f"  T-depth:              {est.t_depth}",
        f"  Physical error rate:  {est.physical_error_rate:.2e}",
        f"  Target logical error: {est.target_logical_error:.2e}",
        f"  Code distance d:      {est.code_distance}",
        f"  Logical err/cycle:    {est.logical_error_per_cycle:.2e}",
        f"  Physical qubits:      {est.physical_qubits:,}",
        f"  Est. runtime:         {est.runtime_seconds * 1e3:.3f} ms",
    ])
