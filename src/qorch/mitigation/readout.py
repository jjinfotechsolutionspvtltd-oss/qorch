"""Readout-error mitigation by calibration-matrix inversion.

The simplest *meaningful* mitigation technique and the M1 differentiator. Real readout is
noisy: a true |0> is sometimes reported as 1 and vice-versa. If we characterise that with a
calibration matrix ``A`` (column j = measured distribution when the true state is basis state
j), then for observed probabilities ``p_obs`` the corrected estimate solves ``A @ p_true =
p_obs``. We invert ``A`` with pure-Python Gauss-Jordan (no numpy), clip negatives, and
renormalise — keeping the slice dependency-free (ADR-6).
"""

from __future__ import annotations

from dataclasses import dataclass


def _invert(matrix: list[list[float]]) -> list[list[float]]:
    """Gauss-Jordan inverse of a square matrix. Raises on singular input."""
    n = len(matrix)
    # augment [A | I]
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]
    for col in range(n):
        # partial pivot for numerical stability
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("calibration matrix is singular; cannot invert")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        aug[col] = [v / piv for v in aug[col]]
        for r in range(n):
            if r != col and aug[r][col]:
                factor = aug[r][col]
                aug[r] = [v - factor * aug[col][k] for k, v in enumerate(aug[r])]
    return [row[n:] for row in aug]


@dataclass(frozen=True)
class ReadoutMitigator:
    """Holds an inverted calibration matrix and applies it to raw counts."""

    labels: tuple[str, ...]            # basis-state bitstrings, matrix column/row order
    inverse: tuple[tuple[float, ...], ...]

    @classmethod
    def from_calibration_matrix(
        cls, labels: list[str], matrix: list[list[float]]
    ) -> "ReadoutMitigator":
        """Build from a calibration matrix ``A`` where ``A[i][j] = P(measure i | prepared j)``."""
        inv = _invert(matrix)
        return cls(labels=tuple(labels), inverse=tuple(tuple(row) for row in inv))

    def apply(self, counts: dict[str, int]) -> dict[str, float]:
        """Return mitigated (corrected) counts. Values are floats; sum is preserved."""
        total = sum(counts.values())
        if total == 0:
            return {label: 0.0 for label in self.labels}
        p_obs = [counts.get(label, 0) / total for label in self.labels]
        p_true = [sum(self.inverse[i][j] * p_obs[j] for j in range(len(p_obs)))
                  for i in range(len(p_obs))]
        # physicality: clip negatives, renormalise to a probability distribution
        p_true = [max(0.0, p) for p in p_true]
        norm = sum(p_true) or 1.0
        return {label: (p / norm) * total for label, p in zip(self.labels, p_true)}
