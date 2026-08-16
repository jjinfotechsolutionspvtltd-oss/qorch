"""Linear-optical photonics as a *separate* IR family.

Photonics does not fit the qubit IR, and forcing it in would be a category
error rather than an inconvenience. A qubit circuit describes unitaries on a
2^n-dimensional state space of distinguishable two-level systems. Linear optics
describes a transformation of **modes** — an n×n unitary on the mode space, not
2^n — with the interesting physics living in how indistinguishable photons
*populate* those modes. Two photons in two modes is not a two-qubit state, and
``Circuit`` has no honest way to say what it is.

So this is its own IR, sharing the orchestration around it rather than the type
underneath it: build a :class:`PhotonicCircuit`, get a transfer matrix, get
photon statistics.

The output distribution comes from **permanents** of submatrices of the transfer
matrix, which is what makes boson sampling hard — the permanent has no
determinant-style fast algorithm. That is a physical fact rather than an
implementation shortcoming, and it is why the helpers here are honest about only
being usable for small mode and photon counts.
"""

from __future__ import annotations

import cmath
import itertools
import math
from dataclasses import dataclass, replace

Matrix = tuple[tuple[complex, ...], ...]


@dataclass(frozen=True)
class BeamSplitter:
    """Mixes two modes. ``theta`` sets reflectivity; ``phi`` a relative phase.

    ``theta = π/4`` is the 50:50 splitter that everything interesting uses.
    """

    mode_a: int
    mode_b: int
    theta: float = math.pi / 4
    phi: float = 0.0
    name: str = "bs"


@dataclass(frozen=True)
class PhaseShifter:
    """Advances the phase of one mode. The other half of universal linear optics."""

    mode: int
    phi: float
    name: str = "ps"


PhotonicOp = BeamSplitter | PhaseShifter


@dataclass(frozen=True)
class PhotonicCircuit:
    """An interferometer: modes, and the optical elements acting on them."""

    num_modes: int
    ops: tuple[PhotonicOp, ...] = ()

    def __post_init__(self) -> None:
        if self.num_modes <= 0:
            raise ValueError("num_modes must be positive")
        for op in self.ops:
            modes = (
                (op.mode_a, op.mode_b) if isinstance(op, BeamSplitter) else (op.mode,)
            )
            for mode in modes:
                if not 0 <= mode < self.num_modes:
                    raise ValueError(
                        f"mode {mode} out of range for {self.num_modes} modes"
                    )
            if isinstance(op, BeamSplitter) and op.mode_a == op.mode_b:
                raise ValueError("a beam splitter needs two distinct modes")

    def beam_splitter(
        self, a: int, b: int, theta: float = math.pi / 4, phi: float = 0.0
    ) -> "PhotonicCircuit":
        return replace(self, ops=self.ops + (BeamSplitter(a, b, theta, phi),))

    def phase_shifter(self, mode: int, phi: float) -> "PhotonicCircuit":
        return replace(self, ops=self.ops + (PhaseShifter(mode, phi),))


def _identity(n: int) -> Matrix:
    return tuple(
        tuple(1 + 0j if i == j else 0j for j in range(n)) for i in range(n)
    )


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    n = len(left)
    return tuple(
        tuple(
            sum(left[i][k] * right[k][j] for k in range(n)) for j in range(n)
        )
        for i in range(n)
    )


def _op_matrix(op: PhotonicOp, n: int) -> Matrix:
    rows = [list(row) for row in _identity(n)]
    if isinstance(op, PhaseShifter):
        rows[op.mode][op.mode] = cmath.exp(1j * op.phi)
    else:
        a, b = op.mode_a, op.mode_b
        cos, sin = math.cos(op.theta), math.sin(op.theta)
        phase = cmath.exp(1j * op.phi)
        rows[a][a] = complex(cos)
        rows[a][b] = 1j * sin * phase.conjugate()
        rows[b][a] = 1j * sin * phase
        rows[b][b] = complex(cos)
    return tuple(tuple(row) for row in rows)


def transfer_matrix(circuit: PhotonicCircuit) -> Matrix:
    """The n×n unitary the interferometer applies to its modes.

    Note the size: n, not 2^n. That is the whole reason this is a separate IR —
    linear optics acts on modes, and the exponential cost appears only when you
    ask how photons distribute across them.
    """
    matrix = _identity(circuit.num_modes)
    for op in circuit.ops:
        matrix = _multiply(_op_matrix(op, circuit.num_modes), matrix)
    return matrix


def is_unitary(matrix: Matrix, tol: float = 1e-9) -> bool:
    """Check U†U = I — the property every passive optical network must have."""
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            total = sum(matrix[k][i].conjugate() * matrix[k][j] for k in range(n))
            expected = 1.0 if i == j else 0.0
            if abs(total - expected) > tol:
                return False
    return True


def permanent(matrix: Matrix) -> complex:
    """Permanent of a small square matrix, by Ryser's formula.

    Like the determinant but without the alternating signs — which removes the
    cancellation that makes determinants cheap, and is precisely why sampling
    photon statistics is believed to be classically hard. Exponential in the
    matrix size by nature, not by neglect, so this is for small systems only.
    """
    n = len(matrix)
    if n == 0:
        return 1 + 0j
    total = 0j
    for r in range(1, 1 << n):
        subset = [i for i in range(n) if r & (1 << i)]
        product = 1 + 0j
        for row in range(n):
            product *= sum(matrix[row][col] for col in subset)
        if product:
            total += product * (-1) ** (n - len(subset))
    return total


def _submatrix(matrix: Matrix, inputs: tuple[int, ...],
               outputs: tuple[int, ...]) -> Matrix:
    """Rows repeated per input photon, columns per output photon."""
    return tuple(tuple(matrix[out][inp] for inp in inputs) for out in outputs)


def output_amplitude(
    circuit: PhotonicCircuit,
    input_modes: tuple[int, ...],
    output_modes: tuple[int, ...],
) -> complex:
    """Amplitude for photons entering ``input_modes`` to leave in ``output_modes``.

    Both are given as one mode per photon, so ``(0, 0)`` means two photons in
    mode 0. The amplitude is the permanent of the corresponding submatrix,
    normalized by the multiplicities — the standard boson-sampling expression.
    """
    if len(input_modes) != len(output_modes):
        raise ValueError("photon number must be the same on input and output")
    matrix = transfer_matrix(circuit)
    for mode in input_modes + output_modes:
        if not 0 <= mode < circuit.num_modes:
            raise ValueError(f"mode {mode} out of range")

    norm = 1.0
    for modes in (input_modes, output_modes):
        for mode in set(modes):
            norm *= math.factorial(modes.count(mode))
    return permanent(_submatrix(matrix, input_modes, output_modes)) / math.sqrt(norm)


def output_distribution(
    circuit: PhotonicCircuit, input_modes: tuple[int, ...]
) -> dict[tuple[int, ...], float]:
    """Probability of every photon-number outcome, keyed by sorted output modes.

    Enumerates all ways the photons can be distributed, so it grows as
    C(modes + photons - 1, photons). Fine for the small interferometers this is
    meant for and hopeless beyond them, which is the honest situation.
    """
    photons = len(input_modes)
    distribution: dict[tuple[int, ...], float] = {}
    for outputs in itertools.combinations_with_replacement(
        range(circuit.num_modes), photons
    ):
        amplitude = output_amplitude(circuit, input_modes, outputs)
        probability = abs(amplitude) ** 2
        if probability > 1e-15:
            distribution[outputs] = probability
    return distribution


def hong_ou_mandel_coincidence(theta: float = math.pi / 4) -> float:
    """Probability two photons leave a beam splitter in *different* modes.

    The canonical test of a linear-optics implementation, and a definitive one:
    at 50:50 the two paths to a coincidence interfere destructively and the
    probability is exactly zero. Photons bunch. A simulator that reports
    anything else at ``theta = π/4`` has the interference wrong, and no amount
    of plausible-looking output elsewhere compensates for that.
    """
    circuit = PhotonicCircuit(2).beam_splitter(0, 1, theta=theta)
    return abs(output_amplitude(circuit, (0, 1), (0, 1))) ** 2
