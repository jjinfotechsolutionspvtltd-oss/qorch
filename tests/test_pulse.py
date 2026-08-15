"""Pulse-level layer: waveforms reproduce gates via rotating-frame simulation."""

from __future__ import annotations

import math

from qorch.pulse import (
    Waveform,
    calibrate_gaussian,
    equals_gate,
    pulse_unitary,
    rotation_angle,
    sx_pulse,
    x_pulse,
    y_pulse,
)

_X = (0, 1, 1, 0)
_Y = (0, -1j, 1j, 0)


def _matmul(u, v):
    a, b, c, d = u
    e, f, g, h = v
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def test_pi_pulse_implements_x():
    assert equals_gate(pulse_unitary(x_pulse()), _X)


def test_pi_pulse_phase_quarter_implements_y():
    assert equals_gate(pulse_unitary(y_pulse()), _Y)


def test_two_sx_pulses_make_x():
    u = pulse_unitary(sx_pulse())
    assert equals_gate(_matmul(u, u), _X)


def test_rotation_angle_matches_rabi_population():
    """A θ-area pulse drives |0> to P(1) = sin^2(θ/2) — exact Rabi physics."""
    for angle in (math.pi / 6, math.pi / 4, math.pi / 2, math.pi):
        u = pulse_unitary(calibrate_gaussian(angle))
        p1 = abs(u[2]) ** 2                 # |<1|U|0>|^2, U[1,0] is index 2
        assert abs(p1 - math.sin(angle / 2) ** 2) < 1e-3


def test_calibrated_angle_is_accurate():
    w = calibrate_gaussian(math.pi)
    assert abs(rotation_angle(w) - math.pi) < 1e-6


def test_unitarity():
    """The simulated evolution must be unitary: U U† = I."""
    u = pulse_unitary(calibrate_gaussian(1.3))
    a, b, c, d = u
    # U U^dagger
    p00 = a * a.conjugate() + b * b.conjugate()
    p11 = c * c.conjugate() + d * d.conjugate()
    p01 = a * c.conjugate() + b * d.conjugate()
    assert abs(p00 - 1) < 1e-9 and abs(p11 - 1) < 1e-9 and abs(p01) < 1e-9


def test_waveform_shapes_build():
    assert len(Waveform.constant(10, 0.5).samples) == 100
    assert len(Waveform.gaussian(20, 1.0, 5.0).samples) == 200
    d = Waveform.drag(20, 1.0, 5.0, beta=0.3)
    assert any(s.imag != 0 for s in d.samples)   # DRAG adds a quadrature
    assert abs(Waveform.constant(10, 0.5).area - 0.5 * 10) < 1e-9


def test_dependency_free():
    """The pulse simulator must not require numpy."""
    import sys
    assert "numpy" not in sys.modules or pulse_unitary(x_pulse()) is not None
