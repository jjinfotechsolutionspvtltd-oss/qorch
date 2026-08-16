"""Neutral atoms: connectivity as geometry rather than as a fixed lattice.

Neutral-atom processors fit the qubit IR — atoms are two-level systems and
lasers drive ordinary rotations — so unlike photonics this needs no separate IR.
What it needs is a different notion of *connectivity*.

Superconducting devices have a coupling map fixed at fabrication. Neutral atoms
do not: the atoms are held in optical tweezers at positions the operator
chooses, and two atoms interact when they are within the **Rydberg blockade
radius** of each other. Connectivity is therefore a function of an arrangement,
and the arrangement can be changed between circuits — or, on some machines,
during one.

That makes the coupling map an *output* of the layout problem rather than an
input to it, which is the genuinely different thing here. This module produces
coupling maps from geometry so the existing router and layout passes can work
against them unchanged.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace

from qorch.transpiler.gateset import IndianGateSet
from qorch.transpiler.routing import CouplingMap

Position = tuple[float, float]


@dataclass(frozen=True)
class AtomArray:
    """Atoms at positions in a plane, coupled by the Rydberg blockade radius.

    Distances are in micrometres, the natural unit for tweezer arrays; the
    blockade radius on real machines is a handful of them.
    """

    positions: tuple[Position, ...]
    blockade_radius_um: float = 8.0

    def __post_init__(self) -> None:
        if self.blockade_radius_um <= 0:
            raise ValueError("blockade radius must be positive")

    @property
    def num_atoms(self) -> int:
        return len(self.positions)

    def distance(self, a: int, b: int) -> float:
        (x1, y1), (x2, y2) = self.positions[a], self.positions[b]
        return math.hypot(x2 - x1, y2 - y1)

    def coupling_map(self) -> CouplingMap:
        """Pairs within the blockade radius, as a symmetric coupling map.

        Both directions are emitted: the blockade is a mutual interaction, so
        there is no preferred control, unlike a superconducting CX that is
        calibrated one way round.
        """
        edges: list[tuple[int, int]] = []
        for a, b in itertools.combinations(range(self.num_atoms), 2):
            if self.distance(a, b) <= self.blockade_radius_um:
                edges.append((a, b))
                edges.append((b, a))
        return CouplingMap(edges=tuple(edges))

    def rearranged(self, positions: tuple[Position, ...]) -> "AtomArray":
        """The same array with atoms moved — the operation a lattice cannot do.

        Rearranging is how a neutral-atom machine adapts its connectivity to a
        circuit instead of forcing the circuit to adapt to its connectivity.
        """
        if len(positions) != self.num_atoms:
            raise ValueError(
                f"expected {self.num_atoms} positions, got {len(positions)}"
            )
        return replace(self, positions=positions)

    def is_connected(self) -> bool:
        """Whether the blockade graph is connected.

        A disconnected arrangement cannot run an entangling circuit across the
        split however cleverly it is routed, so this is worth knowing before
        compiling rather than after routing fails.
        """
        if self.num_atoms <= 1:
            return True
        adjacency: dict[int, set[int]] = {i: set() for i in range(self.num_atoms)}
        for a, b in self.coupling_map().edges:
            adjacency[a].add(b)
        seen = {0}
        stack = [0]
        while stack:
            for neighbour in adjacency[stack.pop()]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return len(seen) == self.num_atoms


def line_array(
    num_atoms: int, spacing_um: float = 5.0, blockade_radius_um: float = 8.0
) -> AtomArray:
    """Atoms in a row.

    Which neighbours couple follows from the numbers rather than from the shape:
    at the defaults, adjacent atoms sit 5 µm apart and couple, next-nearest sit
    10 µm apart and do not. Widening the blockade past ``2 × spacing`` brings the
    next-nearest pairs in.
    """
    return AtomArray(
        positions=tuple((i * spacing_um, 0.0) for i in range(num_atoms)),
        blockade_radius_um=blockade_radius_um,
    )


def grid_array(
    rows: int, cols: int, spacing_um: float = 5.0, blockade_radius_um: float = 8.0
) -> AtomArray:
    """Atoms on a rectangular grid — the common tweezer arrangement."""
    return AtomArray(
        positions=tuple(
            (c * spacing_um, r * spacing_um) for r in range(rows) for c in range(cols)
        ),
        blockade_radius_um=blockade_radius_um,
    )


def ring_array(
    num_atoms: int, spacing_um: float = 5.0, blockade_radius_um: float = 8.0
) -> AtomArray:
    """Atoms on a circle, given the spacing *between neighbours*.

    Parameterized by spacing rather than by the circle's radius, so that "each
    atom couples to its two neighbours" is true by construction. Specifying the
    radius instead makes neighbour spacing ``2·R·sin(π/n)`` — which silently
    exceeds the blockade for a modest ring and produces an array with no edges
    at all, looking like a bug in the coupling code rather than a choice of
    geometry. The radius is derived here instead.
    """
    if num_atoms < 2:
        return AtomArray(positions=((0.0, 0.0),) * num_atoms,
                         blockade_radius_um=blockade_radius_um)
    radius_um = spacing_um / (2.0 * math.sin(math.pi / num_atoms))
    return AtomArray(
        positions=tuple(
            (
                radius_um * math.cos(2 * math.pi * i / num_atoms),
                radius_um * math.sin(2 * math.pi * i / num_atoms),
            )
            for i in range(num_atoms)
        ),
        blockade_radius_um=blockade_radius_um,
    )


# Laser-driven rotations plus a blockade-mediated entangler. ``cx`` stands in
# for the native Rydberg CZ, which differs from it by a Hadamard on the target —
# the decomposer already handles that, and inventing a `cz` gate solely to
# rename an existing one would add a gate to every table in the library for no
# capability gained.
NEUTRAL_ATOM = IndianGateSet(
    name="neutral-atom",
    description="Rydberg-blockade neutral-atom array: rx/ry/rz + blockade entangler",
    basis_gates=("rx", "ry", "rz", "cx"),
    coupling_map=None,      # geometry-derived; see AtomArray.coupling_map()
    num_qubits=0,           # set by the array being used
)


def gate_set_for(array: AtomArray) -> IndianGateSet:
    """The neutral-atom gate set specialized to one arrangement."""
    return replace(
        NEUTRAL_ATOM,
        coupling_map=array.coupling_map().edges,
        num_qubits=array.num_atoms,
    )
