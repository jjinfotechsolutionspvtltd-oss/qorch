"""Single-qubit Clifford+T synthesis: approximate Rz(θ) with an H/T word.

Clifford+T is a *discrete* gate set. It can express rotations by multiples of
π/4 exactly (those are powers of T) and nothing else — every other angle must be
approximated, and the interesting question is how well, at what T-cost, and
whether the caller is told.

The method here is **meet-in-the-middle**. Enumerate every distinct SU(2) element
reachable by an H/T word up to some length, index them in a spatial hash on the
3-sphere, then for each candidate left factor ``U`` look up the nearest stored
``V`` to ``target·U†``. That searches |left| × |table| products while paying only
|left| lookups, which is what makes the accuracy affordable.

Why not Solovay–Kitaev, the textbook answer? SK reaches any precision but its
sequences grow as O(log^3.97(1/ε)) — thousands of T gates for a single rotation.
T-count is precisely what this library's fault-tolerant resource estimator
consumes, so an SK-based synthesis would make its output *worse* while looking
more rigorous. Meet-in-the-middle finds near-minimal words instead: ~24 T gates
for ε ≈ 1e-3, against the ~3·log₂(1/ε) ≈ 30 an optimal synthesizer would use.

Every result carries the error it actually achieved. Nothing here silently
returns a bad approximation.
"""

from __future__ import annotations

import cmath
import itertools
import math
from dataclasses import dataclass

# A 2x2 complex matrix, row-major.
_Mat = tuple[complex, complex, complex, complex]

_INV_SQRT2 = 1 / math.sqrt(2)
_H: _Mat = (_INV_SQRT2 + 0j, _INV_SQRT2 + 0j, _INV_SQRT2 + 0j, -_INV_SQRT2 + 0j)
_T: _Mat = (1 + 0j, 0j, 0j, cmath.exp(1j * math.pi / 4))
_IDENTITY: _Mat = (1 + 0j, 0j, 0j, 1 + 0j)

# Resolution for deduplicating genuinely-distinct group elements.
_DEDUP = 10 ** 9

# Default accuracy target. Word length grows roughly as log(1/ε), so tightening
# this is affordable; see _DEPTH_FOR_PRECISION for the measured trade-off.
DEFAULT_PRECISION = 1e-3

# Enumeration depth needed to reach a given worst-case error, measured over 150
# random angles. Deeper costs more memory (~7 MB at depth 26) and build time.
_DEPTH_FOR_PRECISION: tuple[tuple[float, int, int], ...] = (
    # (worst-case error achieved, enumeration depth, max length of a left factor)
    (2.0e-3, 22, 12),
    (1.0e-3, 26, 14),
    (7.0e-4, 28, 16),
    (3.0e-4, 30, 18),
)


@dataclass(frozen=True)
class SynthesisResult:
    """An H/T word approximating a rotation, with the error it actually achieved."""

    gates: tuple[str, ...]
    error: float
    exact: bool

    @property
    def t_count(self) -> int:
        return sum(1 for g in self.gates if g == "t")


def _mul(u: _Mat, v: _Mat) -> _Mat:
    a, b, c, d = u
    e, f, g, h = v
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def _dagger(u: _Mat) -> _Mat:
    a, b, c, d = u
    return (a.conjugate(), c.conjugate(), b.conjugate(), d.conjugate())


def _su2(u: _Mat) -> tuple[float, float, float, float]:
    """Coordinates of ``u`` on the 3-sphere, as the SU(2) representative.

    Any 2x2 unitary is a phase times an SU(2) element; dividing by √det removes
    the phase. The ±U ambiguity that remains is handled at the call sites (the
    index stores both signs, and fidelity is phase-blind).
    """
    det = u[0] * u[3] - u[1] * u[2]
    s = cmath.sqrt(det)
    a, b = u[0] / s, u[1] / s
    return (a.real, a.imag, b.real, b.imag)


def _canonical(v: tuple[float, float, float, float]) -> tuple[float, ...]:
    """Pick one of ±v, so that U and -U dedup to the same group element."""
    for x in v:
        if abs(x) > 1e-12:
            return v if x > 0 else tuple(-y for y in v)
    return v


def _dedup_key(u: _Mat) -> tuple[int, ...]:
    return tuple(round(x * _DEDUP) for x in _canonical(_su2(u)))


def fidelity(u: _Mat, v: _Mat) -> float:
    """Process fidelity |tr(U†V)|/2 — blind to global phase, as it must be."""
    return 0.5 * abs(sum(x.conjugate() * y for x, y in zip(u, v)))


def _enumerate(max_length: int) -> list[tuple[str, _Mat]]:
    """Every distinct SU(2) element reachable by an H/T word of bounded length.

    Breadth-first, so the first word reaching an element is a shortest one.
    ``HH = I`` is pruned explicitly; every other relation (notably ``T⁸ = I``)
    collapses on its own via the dedup key.
    """
    seen: dict[tuple[int, ...], tuple[str, _Mat]] = {
        _dedup_key(_IDENTITY): ("", _IDENTITY)
    }
    frontier: list[tuple[_Mat, str]] = [(_IDENTITY, "")]
    for _ in range(max_length):
        nxt: list[tuple[_Mat, str]] = []
        for mat, word in frontier:
            for name, gate in (("h", _H), ("t", _T)):
                if name == "h" and word.endswith("h"):
                    continue
                new_mat = _mul(gate, mat)
                key = _dedup_key(new_mat)
                if key in seen:
                    continue
                seen[key] = (word + name, new_mat)
                nxt.append((new_mat, word + name))
        frontier = nxt
        if not frontier:
            break
    return list(seen.values())


class _Index:
    """Spatial hash of group elements on S³, for nearest-neighbour lookup.

    Cell size is matched to the mean spacing of the stored points: too fine and
    the neighbourhood is empty, too coarse and every query degenerates to a scan.
    Both +v and -v are stored so a lookup never has to reason about which
    representative of ±U it holds.
    """

    def __init__(self, items: list[tuple[str, _Mat]]) -> None:
        # Points are spread over a 3-sphere, so mean spacing ~ N^(-1/3).
        self.resolution = max(4, int(len(items) ** (1 / 3.0)))
        self.cells: dict[tuple[int, ...], list[tuple[str, _Mat]]] = {}
        for word, mat in items:
            v = _su2(mat)
            for signed in (v, tuple(-x for x in v)):
                cell = tuple(round(x * self.resolution) for x in signed)
                self.cells.setdefault(cell, []).append((word, mat))

    def near(self, u: _Mat):
        cell = tuple(round(x * self.resolution) for x in _su2(u))
        for offset in itertools.product((-1, 0, 1), repeat=4):
            hit = self.cells.get(tuple(a + b for a, b in zip(cell, offset)))
            if hit:
                yield from hit


# Enumeration is the expensive part, so it is built once per depth on first use.
_TABLES: dict[int, tuple[list[tuple[str, _Mat]], _Index]] = {}


def _table(depth: int) -> tuple[list[tuple[str, _Mat]], _Index]:
    cached = _TABLES.get(depth)
    if cached is None:
        items = _enumerate(depth)
        cached = (items, _Index(items))
        _TABLES[depth] = cached
    return cached


def _rz_matrix(theta: float) -> _Mat:
    return (cmath.exp(-1j * theta / 2), 0j, 0j, cmath.exp(1j * theta / 2))


def _t_power(theta: float) -> tuple[tuple[str, ...], float]:
    """Nearest T^k to Rz(θ), with the error that costs. T = Rz(π/4) up to phase.

    Returns the *candidate*, not a verdict — the caller decides whether the error
    is acceptable. Snapping to T^k whenever it meets the requested precision is
    not a shortcut but the right answer: an angle a millionth of a radian from
    π/4 is served by one T gate, and spending a twenty-gate word to express it
    more "honestly" would be pure waste. Rz(δ) has fidelity |cos(δ/2)|.
    """
    k = round(theta * 4 / math.pi)
    residual = theta - k * math.pi / 4
    error = 1.0 - abs(math.cos(residual / 2))
    return ("t",) * (k % 8), error


def _search(theta: float, depth: int, left_max: int) -> tuple[float, str]:
    """Best H/T word for Rz(θ), as (fidelity, word). Word is in circuit order."""
    items, index = _table(depth)
    target = _rz_matrix(theta)
    best_fidelity, best_word = -1.0, ""
    for word_u, mat_u in items:
        if len(word_u) > left_max:
            continue
        # Want V·U ≈ target, so V should be close to target·U†.
        need = _mul(target, _dagger(mat_u))
        for word_v, mat_v in index.near(need):
            f = fidelity(_mul(mat_v, mat_u), target)
            word = word_u + word_v
            if f > best_fidelity or (f == best_fidelity and len(word) < len(best_word)):
                best_fidelity, best_word = f, word
    return best_fidelity, best_word


_CACHE: dict[tuple[float, float], SynthesisResult] = {}


def synthesize_rz(theta: float, precision: float = DEFAULT_PRECISION) -> SynthesisResult:
    """Approximate ``Rz(theta)`` by a Clifford+T (H/T) word.

    Returns the word in circuit order together with the error it achieved —
    ``1 - fidelity``, and exactly ``0.0`` when θ is a multiple of π/4 and the
    rotation is representable outright.

    Search depth escalates until the requested ``precision`` is met or the
    tabulated maximum is reached. If the target cannot be met, the result still
    comes back with its true error rather than a silent near-miss; callers that
    care (see :mod:`qorch.resource_estimation`) can inspect and report it.
    """
    theta = math.remainder(theta, 2 * math.pi)   # Rz(θ+2π) = -Rz(θ), same up to phase

    # A power of T is the cheapest word there is; take it whenever it suffices.
    t_word, t_error = _t_power(theta)
    if t_error <= max(precision, 0.0):
        return SynthesisResult(
            gates=t_word,
            error=0.0 if t_error < 1e-12 else t_error,
            exact=t_error < 1e-12,
        )

    cache_key = (theta, precision)
    hit = _CACHE.get(cache_key)
    if hit is not None:
        return hit

    best: SynthesisResult | None = None
    for achievable, depth, left_max in _DEPTH_FOR_PRECISION:
        f, word = _search(theta, depth, left_max)
        best = SynthesisResult(gates=tuple(word), error=1.0 - f, exact=False)
        if best.error <= precision:
            break
        if achievable > precision:
            continue          # this depth was never going to be enough — go deeper

    assert best is not None
    _CACHE[cache_key] = best
    return best
