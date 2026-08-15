"""Toric surface code with a minimum-weight perfect-matching (MWPM) decoder.

The toric code is the canonical topological stabilizer code: qubits on the edges
of an L×L torus, ``Z``-type plaquette and ``X``-type vertex stabilizers, distance
``L``, and two logical qubits. Its periodic geometry keeps the decoder clean —
syndrome defects always come in pairs and match by toroidal distance, with no
boundary special-casing.

This module provides the geometry, an exact bitmask-DP MWPM decoder, a homology
(winding-number) logical-failure test, and a code-capacity Monte-Carlo estimator
for the **error-correction threshold** — the defining quantitative result of QEC.
Everything is pure-Python and dependency-free.

Edge indexing (2L² qubits): horizontal ``h(r,c) = r·L + c``; vertical
``v(r,c) = L² + r·L + c``. ``X`` errors are detected by the Z-plaquette syndrome.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from qorch.backends.base import Backend
from qorch.ir import Circuit


class ToricCode:
    """L×L toric code geometry, syndrome, decoder, and logical-failure test."""

    def __init__(self, distance: int) -> None:
        if distance < 2:
            raise ValueError("toric code distance must be >= 2")
        self.L = distance

    @property
    def num_edges(self) -> int:
        return 2 * self.L * self.L

    # --- edge indexing ----------------------------------------------------
    def h(self, r: int, c: int) -> int:
        L = self.L
        return (r % L) * L + (c % L)

    def v(self, r: int, c: int) -> int:
        L = self.L
        return L * L + (r % L) * L + (c % L)

    def plaquettes_of_edge(self, e: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """The two Z-plaquettes whose syndrome an X error on edge ``e`` toggles."""
        L = self.L
        if e < L * L:                      # horizontal edge h(r,c)
            r, c = divmod(e, L)
            return (r, c), ((r - 1) % L, c)
        r, c = divmod(e - L * L, L)        # vertical edge v(r,c)
        return (r, c), (r, (c - 1) % L)

    # --- syndrome ---------------------------------------------------------
    def syndrome(self, error_edges: set[int]) -> list[tuple[int, int]]:
        """Plaquettes with odd error parity (the detected defects)."""
        parity: dict[tuple[int, int], int] = {}
        for e in error_edges:
            for p in self.plaquettes_of_edge(e):
                parity[p] = parity.get(p, 0) ^ 1
        return [p for p, val in parity.items() if val]

    def plaq_distance(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        L = self.L
        dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
        return min(dr, L - dr) + min(dc, L - dc)

    def path_edges(self, a: tuple[int, int], b: tuple[int, int]) -> list[int]:
        """Edges of a shortest plaquette-lattice path from ``a`` to ``b``.

        Flipping these edges toggles exactly the syndrome at ``a`` and ``b``.
        """
        L = self.L
        r, c = a
        tr, tc = b
        edges: list[int] = []
        while r != tr:
            if (tr - r) % L <= (r - tr) % L:   # step r -> r+1 crosses h(r+1, c)
                edges.append(self.h(r + 1, c))
                r = (r + 1) % L
            else:                               # step r -> r-1 crosses h(r, c)
                edges.append(self.h(r, c))
                r = (r - 1) % L
        while c != tc:
            if (tc - c) % L <= (c - tc) % L:   # step c -> c+1 crosses v(r, c+1)
                edges.append(self.v(r, c + 1))
                c = (c + 1) % L
            else:                               # step c -> c-1 crosses v(r, c)
                edges.append(self.v(r, c))
                c = (c - 1) % L
        return edges

    # --- MWPM decoder -----------------------------------------------------
    def decode(self, defects: list[tuple[int, int]], exact_cap: int = 16) -> set[int]:
        """Return a correction edge set matching all defects (MWPM)."""
        if not defects:
            return set()
        pairs = (self._dp_match(defects) if len(defects) <= exact_cap
                 else self._greedy_match(defects))
        corr: set[int] = set()
        for i, j in pairs:
            for e in self.path_edges(defects[i], defects[j]):
                corr ^= {e}   # overlapping path edges cancel
        return corr

    def _dp_match(self, defects: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Exact minimum-weight perfect matching via bitmask DP over defects.

        Explored top-down. That matters more than it looks: because each step
        always pairs off the *lowest* remaining defect, only a small fraction of
        the 2^k masks is ever reachable — on the order of 700 rather than 32768
        at k=15. A bottom-up sweep over every mask is the obvious-looking
        rewrite and is dramatically slower for exactly that reason.

        What the recursion should not do is rebuild the pair list at every node.
        Memoizing cost and partner separately, then reconstructing the matching
        once at the end, removes an O(k) list allocation per subproblem.

        Tie-breaking is unchanged (first minimum wins, scanning j ascending), so
        decoded corrections are identical and Monte-Carlo results reproducible.
        """
        k = len(defects)
        if k == 0:
            return []

        # Flat distance table: one index instead of a method call per probe.
        dist = [0] * (k * k)
        for i in range(k):
            for j in range(i + 1, k):
                d = self.plaq_distance(defects[i], defects[j])
                dist[i * k + j] = d
                dist[j * k + i] = d

        cost_memo: dict[int, float] = {0: 0.0}
        partner: dict[int, int] = {}

        def solve(mask: int) -> float:
            cached = cost_memo.get(mask)
            if cached is not None:
                return cached
            i = (mask & -mask).bit_length() - 1
            rest = mask & ~(1 << i)
            row = i * k
            best = float("inf")
            best_j = -1
            jj = rest
            while jj:
                j = (jj & -jj).bit_length() - 1
                c = dist[row + j] + solve(rest & ~(1 << j))
                if c < best:                   # strict: first minimum wins
                    best = c
                    best_j = j
                jj &= jj - 1
            cost_memo[mask] = best
            partner[mask] = best_j
            return best

        full = (1 << k) - 1
        solve(full)

        pairs: list[tuple[int, int]] = []
        mask = full
        while mask:
            i = (mask & -mask).bit_length() - 1
            j = partner.get(mask, -1)
            if j < 0:                          # odd defect count: nothing to pair
                break
            pairs.append((i, j))
            mask &= ~((1 << i) | (1 << j))
        return pairs

    def _greedy_match(self, defects: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Greedy nearest-neighbour matching (fallback for many defects)."""
        remaining = list(range(len(defects)))
        pairs: list[tuple[int, int]] = []
        while remaining:
            i = remaining.pop(0)
            best_j, best_d = None, float("inf")
            for j in remaining:
                d = self.plaq_distance(defects[i], defects[j])
                if d < best_d:
                    best_d, best_j = d, j
            if best_j is not None:
                remaining.remove(best_j)
                pairs.append((i, best_j))
        return pairs

    # --- logical failure (homology) ---------------------------------------
    def logical_failure(self, net_edges: set[int]) -> bool:
        """True if the residual X chain is a non-contractible (logical) loop.

        X errors form cycles on the *dual* lattice (plaquettes are dual vertices).
        The residual's homology class is its intersection parity with two seam
        cuts: ``{h(0,c)}`` (edges wrapping the dual rows) and ``{v(r,0)}`` (edges
        wrapping the dual columns). Both cuts have even intersection with every
        dual-face boundary, so the parity is homology-invariant; an odd crossing of
        either seam is a logical X error.
        """
        L = self.L
        vert_cut = {self.h(0, c) for c in range(L)}   # wraps dual rows
        horiz_cut = {self.v(r, 0) for r in range(L)}  # wraps dual columns
        return (len(net_edges & vert_cut) % 2 == 1) or (len(net_edges & horiz_cut) % 2 == 1)


def toric_syndrome_circuit(distance: int, error_edges: set[int]) -> Circuit:
    """Build the Z-plaquette syndrome-extraction circuit for the toric code.

    Data qubits are the 2L² edges; one ancilla per plaquette accumulates the
    parity of its four edges (CNOT(data → ancilla)) and is measured. Run on a
    Clifford backend to read the syndrome quantum-mechanically. ``|0…0⟩`` is a +1
    eigenstate of every Z-plaquette, so the measured defects equal the classical
    syndrome of ``error_edges`` — the quantum/classical consistency check.
    """
    code = ToricCode(distance)
    L = distance
    n_data = code.num_edges
    n = n_data + L * L
    circ = Circuit(n, num_clbits=L * L)
    for e in error_edges:
        circ = circ.x(e)
    for r in range(L):
        for c in range(L):
            anc = n_data + r * L + c
            for e in (code.h(r, c), code.h(r + 1, c), code.v(r, c), code.v(r, c + 1)):
                circ = circ.cx(e, anc)
            circ = circ.measure_into(anc, r * L + c)
    return circ


def measured_toric_syndrome(
    distance: int,
    error_edges: set[int],
    backend: Backend | None = None,
) -> set[tuple[int, int]]:
    """Run the syndrome circuit and return the defect plaquettes it reports."""
    from qorch.backends.stabilizer import StabilizerSimulator

    backend = backend or StabilizerSimulator(seed=0)
    counts = backend.run(toric_syndrome_circuit(distance, error_edges), shots=1).counts
    key = next(iter(counts))
    L = distance
    return {(r, c) for r in range(L) for c in range(L) if key[r * L + c] == "1"}


@dataclass(frozen=True)
class ToricThresholdSweep:
    distances: tuple[int, ...]
    physical_errors: tuple[float, ...]
    logical_error: dict[int, dict[float, float]]


def toric_logical_error_rate(
    distance: int,
    physical_error: float,
    trials: int = 2000,
    seed: int | None = None,
) -> float:
    """Code-capacity logical error rate of the toric code under i.i.d. bit-flips.

    Each trial flips every edge qubit independently with probability
    ``physical_error``, extracts the plaquette syndrome, decodes with MWPM, and
    checks whether the residual chain is a logical error.
    """
    code = ToricCode(distance)
    rng = random.Random(seed)
    edges = range(code.num_edges)
    failures = 0
    for _ in range(trials):
        error = {e for e in edges if rng.random() < physical_error}
        defects = code.syndrome(error)
        correction = code.decode(defects)
        net = error ^ correction
        if code.logical_failure(net):
            failures += 1
    return failures / trials


def toric_threshold_sweep(
    distances: tuple[int, ...] = (3, 5),
    physical_errors: tuple[float, ...] = (0.05, 0.08, 0.10, 0.13, 0.16),
    trials: int = 2000,
    seed: int | None = None,
) -> ToricThresholdSweep:
    """Sweep toric-code logical error over distance × physical error.

    Below the ~10% code-capacity threshold the curves separate with larger
    distance giving lower logical error; they cross near the threshold.
    """
    table: dict[int, dict[float, float]] = {}
    for d in distances:
        table[d] = {
            p: toric_logical_error_rate(d, p, trials=trials, seed=seed)
            for p in physical_errors
        }
    return ToricThresholdSweep(tuple(distances), tuple(physical_errors), table)
