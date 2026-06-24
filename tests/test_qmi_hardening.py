"""QMI decoding must reject malformed input with clear errors (security)."""

from __future__ import annotations

import pytest

from qorch import Circuit
from qorch.qmi import from_qmi, to_qmi


def test_roundtrip_still_works():
    c = Circuit(2).h(0).cx(0, 1).rz(0, 0.5).measure(0, 1)
    assert len(from_qmi(to_qmi(c)).gates) == len(c.gates)


def test_empty_buffer_raises_valueerror():
    with pytest.raises(ValueError, match="too short"):
        from_qmi(b"")


def test_bad_magic_raises():
    # full 9-byte header length, but wrong magic
    with pytest.raises(ValueError, match="magic"):
        from_qmi(b"XXXX" + b"\x00" * 5)


def test_truncated_gate_body_raises_valueerror_not_structerror():
    good = to_qmi(Circuit(2).rz(0, 0.5).measure(0, 1))
    # chop off the trailing bytes → must be a clean ValueError, never a crash
    with pytest.raises(ValueError):
        from_qmi(good[:-3])
