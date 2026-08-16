"""Linear optics as a separate IR family, checked against known physics.

The qubit IR cannot represent this: linear optics transforms **modes** — an n×n
unitary, not 2^n — with the physics living in how indistinguishable photons
populate them. Two photons in two modes is not a two-qubit state.

Hong-Ou-Mandel is the test that matters. Two identical photons entering a 50:50
beam splitter *always* leave together; the two paths to a coincidence interfere
destructively and the probability is exactly zero. A simulator that reports
anything else has the interference wrong, and plausible output elsewhere does
not compensate.
"""

from __future__ import annotations

import math

import pytest

from qorch.photonic import (
    BeamSplitter,
    PhotonicCircuit,
    hong_ou_mandel_coincidence,
    is_unitary,
    output_amplitude,
    output_distribution,
    permanent,
    transfer_matrix,
)


# ── Hong-Ou-Mandel ───────────────────────────────────────────────────────


def test_two_photons_bunch_at_a_fifty_fifty_splitter() -> None:
    """Exactly zero, not approximately — this is destructive interference."""
    assert hong_ou_mandel_coincidence() == pytest.approx(0.0, abs=1e-12)


def test_a_fully_transmitting_splitter_never_bunches() -> None:
    """theta=0 is a piece of glass: the photons pass straight through."""
    assert hong_ou_mandel_coincidence(0.0) == pytest.approx(1.0)


def test_the_hom_dip_has_the_right_shape() -> None:
    """Coincidence follows cos²(2θ), dipping to zero only at 50:50."""
    for theta in (0.0, math.pi / 8, math.pi / 4, math.pi / 3, math.pi / 2):
        expected = math.cos(2 * theta) ** 2
        assert hong_ou_mandel_coincidence(theta) == pytest.approx(expected, abs=1e-9)


def test_bunched_photons_leave_together_with_equal_probability() -> None:
    circuit = PhotonicCircuit(2).beam_splitter(0, 1)
    distribution = output_distribution(circuit, (0, 1))

    assert set(distribution) == {(0, 0), (1, 1)}
    assert distribution[(0, 0)] == pytest.approx(0.5)
    assert distribution[(1, 1)] == pytest.approx(0.5)


# ── the transfer matrix ──────────────────────────────────────────────────


def test_an_empty_interferometer_is_the_identity() -> None:
    matrix = transfer_matrix(PhotonicCircuit(3))
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[0][1] == pytest.approx(0.0)


@pytest.mark.parametrize("build", [
    lambda: PhotonicCircuit(2).beam_splitter(0, 1),
    lambda: PhotonicCircuit(3).beam_splitter(0, 1).phase_shifter(1, 0.7),
    lambda: PhotonicCircuit(4).beam_splitter(0, 1).beam_splitter(2, 3)
                              .phase_shifter(0, 1.1).beam_splitter(1, 2),
])
def test_every_optical_network_is_unitary(build) -> None:
    """Passive optics conserves photons; a non-unitary transfer matrix cannot."""
    assert is_unitary(transfer_matrix(build()))


def test_the_transfer_matrix_is_mode_sized_not_exponential() -> None:
    """The reason this is a separate IR: n modes, not 2^n amplitudes."""
    assert len(transfer_matrix(PhotonicCircuit(6))) == 6


def test_a_phase_shifter_only_touches_its_own_mode() -> None:
    matrix = transfer_matrix(PhotonicCircuit(3).phase_shifter(1, math.pi))
    assert matrix[0][0] == pytest.approx(1.0)
    assert matrix[1][1] == pytest.approx(-1.0)
    assert matrix[2][2] == pytest.approx(1.0)


# ── permanents ───────────────────────────────────────────────────────────


def test_permanent_of_a_small_matrix() -> None:
    assert permanent(((1 + 0j, 1 + 0j), (1 + 0j, 1 + 0j))) == pytest.approx(2 + 0j)


def test_permanent_has_no_alternating_signs() -> None:
    """perm[[1,2],[3,4]] = 1·4 + 2·3 = 10, where the determinant is -2."""
    assert permanent(((1 + 0j, 2 + 0j), (3 + 0j, 4 + 0j))) == pytest.approx(10 + 0j)


def test_permanent_of_the_empty_matrix_is_one() -> None:
    assert permanent(()) == pytest.approx(1 + 0j)


# ── distributions ────────────────────────────────────────────────────────


@pytest.mark.parametrize("build,inputs", [
    (lambda: PhotonicCircuit(2).beam_splitter(0, 1), (0, 1)),
    (lambda: PhotonicCircuit(3).beam_splitter(0, 1).phase_shifter(1, 0.7)
                               .beam_splitter(1, 2), (0, 1)),
    (lambda: PhotonicCircuit(3).beam_splitter(0, 1).beam_splitter(1, 2), (0, 1, 2)),
])
def test_output_distributions_are_normalized(build, inputs) -> None:
    """Photons are neither created nor lost by passive optics."""
    assert sum(output_distribution(build(), inputs).values()) == pytest.approx(1.0)


def test_a_single_photon_splits_evenly() -> None:
    distribution = output_distribution(PhotonicCircuit(2).beam_splitter(0, 1), (0,))
    assert distribution[(0,)] == pytest.approx(0.5)
    assert distribution[(1,)] == pytest.approx(0.5)


def test_photon_number_must_match() -> None:
    with pytest.raises(ValueError, match="same on input and output"):
        output_amplitude(PhotonicCircuit(2).beam_splitter(0, 1), (0, 1), (0,))


# ── validation ───────────────────────────────────────────────────────────


def test_a_mode_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError, match="out of range"):
        PhotonicCircuit(2, ops=(BeamSplitter(0, 5),))


def test_a_beam_splitter_needs_two_distinct_modes() -> None:
    with pytest.raises(ValueError, match="two distinct modes"):
        PhotonicCircuit(2, ops=(BeamSplitter(1, 1),))


def test_a_circuit_needs_at_least_one_mode() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        PhotonicCircuit(0)


def test_circuits_are_immutable_like_the_qubit_ir() -> None:
    """Same value semantics as Circuit, so passes compose the same way."""
    base = PhotonicCircuit(2)
    extended = base.beam_splitter(0, 1)
    assert base.ops == ()
    assert len(extended.ops) == 1
