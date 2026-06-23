"""QMI — Quantum Machine Instruction binary format.

Compact binary encoding for quantum circuits suitable for QPU microcode,
firmware transfer, and low-latency job submission.

Format (v1):
  [4 bytes]  magic: ``QMI1``
  [1 byte ]  num_qubits
  [2 bytes]  num_gates  (little-endian)
  [1 byte ]  num_measured
  gate records × num_gates:
    [1 byte ]  opcode
    [1 byte ]  num_qubits
    [num_qubits bytes]  qubit indices
    [1 byte ]  num_params  (0--3)
    [num_params × 4 bytes]  params as float32 (little-endian)
  [num_measured bytes]  measured qubit indices
"""

from __future__ import annotations

import struct
from dataclasses import replace
from typing import ClassVar

from qorch.ir import Circuit, Gate

OPCODES: dict[str, int] = {
    "h": 0,
    "x": 1,
    "y": 2,
    "z": 3,
    "cx": 4,
    "sx": 5,
    "rz": 6,
    "rx": 7,
    "ry": 8,
    "swap": 9,
    "ms": 10,
    "id": 11,
}

NAMES: dict[int, str] = {v: k for k, v in OPCODES.items()}

_MAGIC = b"QMI1"
_HEADER_FMT = "<4sBHBB"  # magic(4s), reserved(1B), num_qubits(B), num_gates(H), num_measured(B)
_GATE_HEADER_FMT = "<BBB"  # opcode(B), nqubits(B), nparams(B)


def _float32_le(x: float) -> bytes:
    return struct.pack("<f", x)


def _unpack_float32_le(data: bytes, offset: int) -> tuple[float, int]:
    return struct.unpack_from("<f", data, offset)[0], offset + 4


def to_qmi(circuit: Circuit) -> bytes:
    """Encode a ``Circuit`` as QMI binary bytes."""
    gates_data = bytearray()
    for g in circuit.gates:
        op = OPCODES[g.name]
        nq = len(g.qubits)
        np = len(g.params)
        gates_data.extend(struct.pack("<BBB", op, nq, np))
        gates_data.extend(bytes(g.qubits))
        for p in g.params:
            gates_data.extend(_float32_le(p))

    measured = bytes(circuit.measured)
    header = struct.pack(
        _HEADER_FMT,
        _MAGIC,
        0,  # reserved
        circuit.num_qubits,
        len(circuit.gates),
        len(measured),
    )
    return bytes(header + gates_data + measured)


def from_qmi(data: bytes) -> Circuit:
    """Decode QMI binary bytes into a ``Circuit``."""
    magic, _reserved, num_qubits, num_gates, num_measured = struct.unpack_from(
        _HEADER_FMT, data, 0
    )
    if magic != _MAGIC:
        raise ValueError(f"bad magic: {magic!r}, expected {_MAGIC!r}")

    offset = struct.calcsize(_HEADER_FMT)
    gates: list[Gate] = []
    for _ in range(num_gates):
        op, nq, np = struct.unpack_from(_GATE_HEADER_FMT, data, offset)
        offset += struct.calcsize(_GATE_HEADER_FMT)
        qubits = tuple(data[offset : offset + nq])
        offset += nq
        params: list[float] = []
        for _ in range(np):
            p, offset = _unpack_float32_le(data, offset)
            params.append(p)
        gates.append(Gate(name=NAMES[op], qubits=qubits, params=tuple(params)))

    measured = tuple(data[offset : offset + num_measured])
    offset += num_measured

    c = Circuit(num_qubits=num_qubits)
    c = replace(c, gates=tuple(gates), measured=measured)
    return c


class QMIEncoder:
    """Encoder/decoder pair with human-readable helpers."""

    SUPPORTED_GATES: ClassVar[frozenset[str]] = frozenset(OPCODES)

    @staticmethod
    def encode(circuit: Circuit) -> bytes:
        return to_qmi(circuit)

    @staticmethod
    def decode(data: bytes) -> Circuit:
        return from_qmi(data)

    @staticmethod
    def hexdump(data: bytes, max_bytes: int = 256) -> str:
        """Return a hex dump of the first ``max_bytes`` of QMI data."""
        preview = data[:max_bytes]
        lines = []
        for i in range(0, len(preview), 16):
            chunk = preview[i : i + 16]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{i:04x}  {hex_part:<48s}  {ascii_part}")
        if len(data) > max_bytes:
            lines.append(f"... ({len(data)} total bytes)")  # pragma: no cover
        return "\n".join(lines)
