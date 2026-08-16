"""Neutral atoms: connectivity derived from geometry, and rearrangeable.

Unlike photonics this needs no separate IR — atoms are two-level systems and
lasers drive ordinary rotations. What differs is that the coupling map is not
fixed at fabrication: atoms sit where the operator puts them, couple within the
Rydberg blockade radius, and can be moved between circuits.

So the coupling map is an *output* of the arrangement rather than an input to
compilation, and these tests pin that it is derived correctly and that the
existing router works against it unchanged.
"""

from __future__ import annotations


import pytest

from qorch import Circuit, LocalSimulator
from qorch.neutral_atom import (
    NEUTRAL_ATOM,
    AtomArray,
    gate_set_for,
    grid_array,
    line_array,
    ring_array,
)
from qorch.transpiler import transpile_with_layout


def _undirected(array: AtomArray) -> set[tuple[int, int]]:
    return {tuple(sorted(edge)) for edge in array.coupling_map().edges}


# ── connectivity follows from geometry ───────────────────────────────────


def test_only_atoms_inside_the_blockade_radius_couple() -> None:
    """5 µm apart couples at an 8 µm blockade; 10 µm apart does not."""
    assert _undirected(line_array(5)) == {(0, 1), (1, 2), (2, 3), (3, 4)}


def test_widening_the_blockade_adds_couplings() -> None:
    """Nothing physical changed but the radius, and the graph densifies."""
    narrow = _undirected(line_array(5, spacing_um=5.0, blockade_radius_um=8.0))
    wide = _undirected(line_array(5, spacing_um=5.0, blockade_radius_um=11.0))
    assert narrow < wide
    assert (0, 2) in wide and (0, 2) not in narrow


def test_atoms_beyond_the_blockade_do_not_couple_at_all() -> None:
    isolated = line_array(5, spacing_um=5.0, blockade_radius_um=4.0)
    assert _undirected(isolated) == set()
    assert not isolated.is_connected()


def test_coupling_is_symmetric() -> None:
    """The blockade is mutual — unlike a CX calibrated in one direction."""
    edges = set(line_array(4).coupling_map().edges)
    for a, b in edges:
        assert (b, a) in edges


def test_a_ring_couples_each_atom_to_its_two_neighbours() -> None:
    """Parameterized by neighbour spacing so this is true by construction.

    Specifying a radius instead makes the spacing 2·R·sin(π/n), which silently
    exceeds the blockade for a modest ring and yields an array with no edges —
    looking like a coupling bug rather than a choice of geometry.
    """
    ring = ring_array(6)
    assert len(_undirected(ring)) == 6
    assert ring.is_connected()


def test_a_grid_is_connected_and_denser_than_a_line() -> None:
    grid = grid_array(2, 3)
    assert grid.is_connected()
    assert len(_undirected(grid)) > len(_undirected(line_array(6)))


def test_distances_are_euclidean() -> None:
    array = AtomArray(positions=((0.0, 0.0), (3.0, 4.0)))
    assert array.distance(0, 1) == pytest.approx(5.0)


@pytest.mark.parametrize("build", [
    lambda: line_array(4), lambda: grid_array(2, 2), lambda: ring_array(5),
])
def test_standard_arrangements_are_connected(build) -> None:
    assert build().is_connected()


def test_a_single_atom_is_trivially_connected() -> None:
    assert line_array(1).is_connected()


# ── rearrangement: what a fixed lattice cannot do ────────────────────────


def test_moving_atoms_changes_the_coupling_map() -> None:
    """The distinguishing capability: connectivity adapts to the circuit."""
    array = line_array(3, spacing_um=5.0, blockade_radius_um=6.0)
    assert (0, 2) not in _undirected(array)

    huddled = array.rearranged(((0.0, 0.0), (5.0, 0.0), (2.5, 4.0)))
    assert (0, 2) in _undirected(huddled)


def test_rearranging_preserves_the_atom_count() -> None:
    with pytest.raises(ValueError, match="expected 3 positions"):
        line_array(3).rearranged(((0.0, 0.0), (1.0, 1.0)))


def test_rearranging_leaves_the_original_untouched() -> None:
    original = line_array(3)
    original.rearranged(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    assert original.positions == line_array(3).positions


def test_a_blockade_radius_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        AtomArray(positions=((0.0, 0.0),), blockade_radius_um=0.0)


# ── it compiles through the existing pipeline ────────────────────────────


def test_a_circuit_compiles_onto_an_atom_array() -> None:
    """The point of deriving a CouplingMap: the router needs no special case."""
    array = line_array(5)
    gate_set = gate_set_for(array)
    circuit = Circuit(5).h(0).cx(0, 4).measure(*range(5))

    result = transpile_with_layout(circuit, gate_set,
                                   coupling_map=array.coupling_map())
    names = {g.name for g in result.circuit.gates}
    assert names <= set(gate_set.basis_gates)
    assert result.metrics.swaps_inserted > 0      # 0 and 4 are far apart on a line


def test_compiling_onto_atoms_preserves_the_distribution() -> None:
    array = line_array(5)
    circuit = Circuit(5).h(0).cx(0, 4).measure(*range(5))
    compiled = transpile_with_layout(
        circuit, gate_set_for(array), coupling_map=array.coupling_map()
    ).circuit

    before = LocalSimulator(seed=3).run(circuit, shots=2000).counts
    after = LocalSimulator(seed=3).run(compiled, shots=2000).counts
    for key in set(before) | set(after):
        assert abs(before.get(key, 0) - after.get(key, 0)) / 2000 < 0.05


def test_a_better_arrangement_needs_fewer_swaps() -> None:
    """Rearranging to suit the circuit is the neutral-atom advantage, measured."""
    circuit = Circuit(3).h(0).cx(0, 2).cx(0, 2).measure(0, 1, 2)

    spread = line_array(3, spacing_um=5.0, blockade_radius_um=6.0)
    huddled = spread.rearranged(((0.0, 0.0), (5.0, 0.0), (2.5, 4.0)))

    spread_swaps = transpile_with_layout(
        circuit, gate_set_for(spread), coupling_map=spread.coupling_map()
    ).metrics.swaps_inserted
    huddled_swaps = transpile_with_layout(
        circuit, gate_set_for(huddled), coupling_map=huddled.coupling_map()
    ).metrics.swaps_inserted

    assert huddled_swaps < spread_swaps


def test_the_gate_set_specializes_to_an_arrangement() -> None:
    array = grid_array(2, 3)
    gate_set = gate_set_for(array)

    assert gate_set.num_qubits == 6
    assert gate_set.coupling_map == array.coupling_map().edges
    assert NEUTRAL_ATOM.coupling_map is None      # the template stays geometry-free
