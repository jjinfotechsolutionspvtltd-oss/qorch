"""Zero-noise extrapolation (ZNE) via unitary circuit folding.

ZNE amplifies the circuit's noise by known factors, measures an observable at each, and
extrapolates back to the zero-noise limit. We amplify noise *digitally* by folding the
circuit ``C -> C (C^dagger C)^k``: logically the identity-padded circuit equals ``C``, but a
noisy device accumulates ``(2k+1)x`` the gate errors — so noise scale ``lambda = 2k+1``.

All gates in the QS-004 IR (h, x, z, cx) are self-inverse, so the adjoint ``C^dagger`` is just
the gate list reversed. When non-self-inverse gates are added, ``_dagger`` must invert each.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

from qorch.backends.base import Backend
from qorch.ir import Circuit, Gate

# An observable maps measurement counts to a real expectation value.
Observable = Callable[[dict[str, int]], float]


def _dagger(gates: tuple[Gate, ...]) -> tuple[Gate, ...]:
    """Adjoint of a gate sequence. Valid while every gate is self-inverse (see module docstring)."""
    return tuple(reversed(gates))


def fold_circuit(circuit: Circuit, scale: int) -> Circuit:
    """Fold to noise scale ``scale`` (an odd integer >= 1). Gate count becomes ``scale x``."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError(f"scale must be an odd integer >= 1, got {scale}")
    base = circuit.gates
    folded = list(base)
    for _ in range((scale - 1) // 2):
        folded.extend(_dagger(base))
        folded.extend(base)
    return replace(circuit, gates=tuple(folded))


def expectation_z(counts: dict[str, int], qubit: int = 0) -> float:
    """<Z> on ``qubit`` = P(0) - P(1). Assumes readout order maps position i -> qubit i."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    signed = sum((1 if key[qubit] == "0" else -1) * c for key, c in counts.items())
    return signed / total


def _linear_extrapolate(xs: list[float], ys: list[float]) -> float:
    """Least-squares line through (xs, ys); return its value at x = 0 (the zero-noise limit)."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var = sum((x - mean_x) ** 2 for x in xs)
    if var == 0:
        return mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / var
    return mean_y - slope * mean_x  # intercept = value at x = 0


@dataclass(frozen=True)
class ZNEResult:
    scales: tuple[int, ...]
    values: tuple[float, ...]   # observed expectation at each noise scale
    mitigated: float            # extrapolated zero-noise value

    @property
    def raw(self) -> float:
        """The unmitigated value (noise scale 1)."""
        return self.values[0]


def zne_expectation(
    backend: Backend,
    circuit: Circuit,
    observable: Observable,
    scales: tuple[int, ...] = (1, 3, 5),
    shots: int = 8192,
) -> ZNEResult:
    """Run ``circuit`` at each folding scale, then extrapolate the observable to zero noise."""
    xs: list[float] = []
    ys: list[float] = []
    for scale in scales:
        result = backend.run(fold_circuit(circuit, scale), shots)
        xs.append(float(scale))
        ys.append(observable(result.counts))
    return ZNEResult(
        scales=tuple(scales),
        values=tuple(ys),
        mitigated=_linear_extrapolate(xs, ys),
    )
