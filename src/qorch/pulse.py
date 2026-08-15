"""Pulse-level control: waveforms and a rotating-frame single-qubit simulator.

The gate-level IR abstracts away *how* a gate is physically realized. Real
hardware (superconducting, ion-trap) drives qubits with shaped microwave/laser
**pulses**; calibration and pulse-aware compilation bottom out here. This module
adds that missing layer:

  - ``Waveform`` — a complex baseband envelope (I/Q) sampled at a time step, with
    ``constant``, ``gaussian``, and ``drag`` shapes.
  - ``pulse_unitary`` — the exact rotating-frame evolution a resonant drive
    produces, computed in closed form per time slice (pure Python, dependency-free).
  - calibration helpers and ``x_pulse`` / ``y_pulse`` / ``sx_pulse`` that reproduce
    the corresponding gates from physical pulses.

Model: on resonance and in the rotating frame, a drive with complex envelope
``Ω(t) = I(t) + iQ(t)`` gives ``H(t) = ½(I(t) σx + Q(t) σy)``. The rotation angle
of a pulse is the envelope area ``∫Ω dt``; a π-area real pulse implements X
(an ``Rx(π)`` up to global phase), a π/2-area pulse implements √X, and a π/2
phase offset rotates about the Y axis.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

# A 2x2 complex matrix as a row-major 4-tuple (a, b, c, d) = [[a, b], [c, d]].
Matrix2 = tuple[complex, complex, complex, complex]

_I2: Matrix2 = (1, 0, 0, 1)


def _matmul(u: Matrix2, v: Matrix2) -> Matrix2:
    a, b, c, d = u
    e, f, g, h = v
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


@dataclass(frozen=True)
class Waveform:
    """A complex baseband envelope sampled at fixed time step ``dt`` (ns).

    ``samples[k] = I_k + i·Q_k`` is the drive amplitude (rad/ns) during slice k.
    """

    samples: tuple[complex, ...]
    dt: float = 0.1

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")

    @property
    def duration(self) -> float:
        return len(self.samples) * self.dt

    @property
    def area(self) -> complex:
        """Envelope area ∫Ω dt — the net rotation angle (complex: I→X, Q→Y)."""
        return sum(self.samples) * self.dt

    def scaled(self, factor: float) -> "Waveform":
        return Waveform(tuple(s * factor for s in self.samples), self.dt)

    def with_phase(self, phi: float) -> "Waveform":
        """Rotate the drive phase, moving the rotation axis in the X–Y plane."""
        ph = cmath.exp(1j * phi)
        return Waveform(tuple(s * ph for s in self.samples), self.dt)

    # --- shapes -----------------------------------------------------------
    @classmethod
    def constant(cls, duration: float, amp: float, dt: float = 0.1) -> "Waveform":
        n = max(1, round(duration / dt))
        return cls((complex(amp),) * n, dt)

    @classmethod
    def gaussian(cls, duration: float, amp: float, sigma: float, dt: float = 0.1) -> "Waveform":
        n = max(1, round(duration / dt))
        t0 = (n - 1) / 2.0
        samples = tuple(
            complex(amp * math.exp(-0.5 * ((k - t0) * dt / sigma) ** 2))
            for k in range(n)
        )
        return cls(samples, dt)

    @classmethod
    def drag(
        cls, duration: float, amp: float, sigma: float, beta: float, dt: float = 0.1
    ) -> "Waveform":
        """Gaussian with a DRAG quadrature ``i·β·dG/dt`` (leakage suppression)."""
        n = max(1, round(duration / dt))
        t0 = (n - 1) / 2.0
        samples: list[complex] = []
        for k in range(n):
            x = (k - t0) * dt
            g = amp * math.exp(-0.5 * (x / sigma) ** 2)
            dg = -(x / sigma**2) * g          # derivative of the Gaussian
            samples.append(complex(g, beta * dg))
        return cls(tuple(samples), dt)


def _slice_unitary(omega: complex, dt: float) -> Matrix2:
    """Exact exp(-i H dt) for H = ½(Re Ω·σx + Im Ω·σy) over one time slice."""
    a = omega.real / 2.0      # σx coefficient
    b = omega.imag / 2.0      # σy coefficient
    r = math.hypot(a, b)
    cos = math.cos(r * dt)
    if r == 0.0:
        return (complex(cos), 0, 0, complex(cos))
    s = math.sin(r * dt) / r
    # U = cos·I - i·sin(r dt)·(a σx + b σy)/r, with (aσx+bσy) = [[0, a-ib],[a+ib, 0]]
    off_upper = -1j * s * (a - 1j * b)
    off_lower = -1j * s * (a + 1j * b)
    return (complex(cos), off_upper, off_lower, complex(cos))


def pulse_unitary(waveform: Waveform) -> Matrix2:
    """Return the 2x2 unitary a resonant drive ``waveform`` produces (rotating frame)."""
    u = _I2
    for omega in waveform.samples:
        u = _matmul(_slice_unitary(omega, waveform.dt), u)
    return u


def rotation_angle(waveform: Waveform) -> float:
    """Net rotation angle of a (single-axis) pulse = |envelope area|."""
    return abs(waveform.area)


def calibrate_gaussian(
    target_angle: float,
    duration: float = 20.0,
    sigma: float = 5.0,
    dt: float = 0.1,
    phase: float = 0.0,
) -> Waveform:
    """A Gaussian pulse whose rotation angle is exactly ``target_angle``.

    Builds a unit-amplitude Gaussian, then scales it so its area equals the target
    angle (the rotation a resonant drive imparts). ``phase`` sets the rotation axis
    (0 → X, π/2 → Y).
    """
    base = Waveform.gaussian(duration, amp=1.0, sigma=sigma, dt=dt)
    unit_area = base.area.real
    if unit_area == 0:
        raise ValueError("degenerate pulse (zero area)")
    return base.scaled(target_angle / unit_area).with_phase(phase)


def x_pulse(duration: float = 20.0, sigma: float = 5.0, dt: float = 0.1) -> Waveform:
    """A π pulse about X — implements the X gate (as Rx(π), up to global phase)."""
    return calibrate_gaussian(math.pi, duration, sigma, dt, phase=0.0)


def y_pulse(duration: float = 20.0, sigma: float = 5.0, dt: float = 0.1) -> Waveform:
    """A π pulse about Y — implements the Y gate (as Ry(π), up to global phase)."""
    return calibrate_gaussian(math.pi, duration, sigma, dt, phase=math.pi / 2)


def sx_pulse(duration: float = 20.0, sigma: float = 5.0, dt: float = 0.1) -> Waveform:
    """A π/2 pulse about X — implements √X (SX), up to global phase."""
    return calibrate_gaussian(math.pi / 2, duration, sigma, dt, phase=0.0)


def equals_gate(u: Matrix2, target: Matrix2, tol: float = 1e-3) -> bool:
    """True if ``u`` equals ``target`` up to a global phase, within ``tol``."""
    ref = None
    for x, y in zip(u, target):
        if abs(y) > 1e-9:
            ref = x / y
            break
    if ref is None:
        return False
    if abs(abs(ref) - 1.0) > tol:
        return False
    return all(abs(x - ref * y) < tol for x, y in zip(u, target))
