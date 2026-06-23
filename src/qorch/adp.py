"""ADP — Application Development Platform for quantum algorithms.

High-level algorithm templates that compose on top of qorch's IR, backends,
and mitigation layers. Every algorithm returns typed result objects and
accepts any ``Backend`` for execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qorch.ir import Circuit, Gate
from qorch.backends.base import Backend


# ── Shared result types ──────────────────────────────────────────────────


@dataclass
class AlgorithmResult:
    """Base result for ADP algorithms."""


@dataclass
class QFTResult(AlgorithmResult):
    """Result of Quantum Fourier Transform execution."""

    num_qubits: int
    counts: dict[str, int]
    shots: int


@dataclass
class GroverResult(AlgorithmResult):
    """Result of Grover's search algorithm."""

    num_qubits: int
    marked_states: tuple[str, ...]
    top_outcomes: list[tuple[str, int]]
    shots: int
    iterations: int


@dataclass
class QAOResult(AlgorithmResult):
    """Result of QAOA execution."""

    num_qubits: int
    optimal_angles: tuple[float, float]
    best_solution: str
    best_value: float
    counts: dict[str, int]
    shots: int


@dataclass
class VQEResult(AlgorithmResult):
    """Result of VQE execution."""

    num_qubits: int
    optimal_params: tuple[float, ...]
    optimal_value: float
    nfev: int


@dataclass
class QPEResult(AlgorithmResult):
    """Result of Quantum Phase Estimation."""

    num_qubits: int
    phase: float | None
    counts: dict[str, int]


# ── 1. Quantum Fourier Transform ─────────────────────────────────────────


def qft_circuit(num_qubits: int, swap: bool = True) -> Circuit:
    """Build a Quantum Fourier Transform circuit.

    Args:
        num_qubits: Number of qubits.
        swap: Whether to include final SWAP gates for bit-reversal.

    Returns:
        ``Circuit`` implementing the QFT.
    """
    c = Circuit(num_qubits)
    for i in range(num_qubits):
        c = c.h(i)
        for j in range(i + 1, num_qubits):
            angle = math.pi / (1 << (j - i))
            c = c.cx(j, i)
            c = c.rz(i, angle)
            c = c.cx(j, i)
    if swap:
        for i in range(num_qubits // 2):
            c = c.swap(i, num_qubits - 1 - i)
    return c


def run_qft(
    backend: Backend,
    num_qubits: int,
    input_state: Circuit | None = None,
    shots: int = 1024,
) -> QFTResult:
    """Run Quantum Fourier Transform on a backend.

    Args:
        backend: Target backend.
        num_qubits: Number of qubits.
        input_state: Optional circuit to prepare input state (appended before QFT).
        shots: Number of shots.

    Returns:
        ``QFTResult`` with measurement counts.
    """
    c = qft_circuit(num_qubits)
    if input_state is not None:
        c = _combine_circuits(input_state, c)
    c = c.measure(*range(num_qubits))
    result = backend.run(c, shots=shots)
    return QFTResult(num_qubits=num_qubits, counts=result.counts, shots=result.shots)


# ── 2. Grover's Search ───────────────────────────────────────────────────


def grover_diffusion(num_qubits: int) -> Circuit:
    """Build Grover diffusion operator (inversion about the mean)."""
    c = Circuit(num_qubits)
    for q in range(num_qubits):
        c = c.h(q)
    for q in range(num_qubits):
        c = c.x(q)
    c = c.h(num_qubits - 1)
    c = c.cx(0, num_qubits - 1) if num_qubits >= 2 else c
    for q in range(1, num_qubits - 1):
        c = c.cx(0, q) if q == 1 else c
        if q > 1:
            c = c.cx(q, num_qubits - 1)
    c = c.h(num_qubits - 1)
    for q in range(num_qubits):
        c = c.x(q)
    for q in range(num_qubits):
        c = c.h(q)
    return c


def oracle_by_bitstring(num_qubits: int, target: str) -> Circuit:
    """Build an oracle that marks a target bitstring via phase flip."""
    c = Circuit(num_qubits)
    for i, bit in enumerate(target):
        if bit == "0":
            c = c.x(i)
    if num_qubits >= 2:
        c = c.h(num_qubits - 1)
        c = c.cx(0, num_qubits - 1) if num_qubits >= 2 else c
        for q in range(1, num_qubits - 1):
            c = c.cx(q, num_qubits - 1)
        c = c.h(num_qubits - 1)
    else:
        c = c.z(0)
    for i, bit in enumerate(target):
        if bit == "0":
            c = c.x(i)
    return c


def _combine_circuits(a: Circuit, b: Circuit) -> Circuit:
    """Append circuit ``b`` after circuit ``a`` (same qubit count)."""
    from dataclasses import replace
    return replace(a, gates=a.gates + b.gates)


def run_grover(
    backend: Backend,
    num_qubits: int,
    marked: str | tuple[str, ...],
    shots: int = 1024,
    iterations: int | None = None,
) -> GroverResult:
    """Run Grover's search algorithm.

    Args:
        backend: Target backend.
        num_qubits: Number of qubits.
        marked: Bitstring to search for (or tuple of bitstrings).
        shots: Number of shots.
        iterations: Number of Grover iterations (auto if None).

    Returns:
        ``GroverResult`` with measurement outcomes.
    """
    if isinstance(marked, str):
        marked = (marked,)

    N = 1 << num_qubits
    M = len(marked)
    if iterations is None:
        theta = math.asin(math.sqrt(M / N)) if M <= N else math.pi / 2
        iterations = max(1, round(math.pi / (4 * theta) - 0.5))

    c = Circuit(num_qubits)
    for q in range(num_qubits):
        c = c.h(q)

    for _ in range(iterations):
        for m in marked:
            c = _combine_circuits(c, oracle_by_bitstring(num_qubits, m))
        c = _combine_circuits(c, grover_diffusion(num_qubits))

    c = c.measure(*range(num_qubits))
    result = backend.run(c, shots=shots)
    sorted_outcomes = sorted(result.counts.items(), key=lambda x: -x[1])
    return GroverResult(
        num_qubits=num_qubits,
        marked_states=marked,
        top_outcomes=[(k, v) for k, v in sorted_outcomes[:5]],
        shots=shots,
        iterations=iterations,
    )


# ── 3. QAOA ──────────────────────────────────────────────────────────────


def qaoa_cost_circuit(num_qubits: int, gamma: float, edges: list[tuple[int, int]]) -> Circuit:
    """Build QAOA cost layer: exp(-i * gamma * H_C) for MaxCut."""
    c = Circuit(num_qubits)
    for u, v in edges:
        c = c.cx(u, v)
        c = c.rz(v, 2.0 * gamma)
        c = c.cx(u, v)
    return c


def qaoa_mixer_circuit(num_qubits: int, beta: float) -> Circuit:
    """Build QAOA mixer layer: exp(-i * beta * H_B) where H_B = sum X_i."""
    c = Circuit(num_qubits)
    for q in range(num_qubits):
        c = c.rx(q, 2.0 * beta)
    return c


def run_qaoa(
    backend: Backend,
    num_qubits: int,
    edges: list[tuple[int, int]],
    gamma: float = 0.5,
    beta: float = 0.5,
    layers: int = 1,
    shots: int = 4096,
) -> QAOResult:
    """Run QAOA for MaxCut with fixed angles.

    Args:
        backend: Target backend.
        num_qubits: Number of qubits.
        edges: Graph edges as (u, v) pairs.
        gamma, beta: QAOA angles.
        layers: Number of QAOA layers (p).
        shots: Number of shots.

    Returns:
        ``QAOResult`` with solutions and their values.
    """
    c = Circuit(num_qubits)
    for q in range(num_qubits):
        c = c.h(q)

    for _ in range(layers):
        c = _combine_circuits(c, qaoa_cost_circuit(num_qubits, gamma, edges))
        c = _combine_circuits(c, qaoa_mixer_circuit(num_qubits, beta))

    c = c.measure(*range(num_qubits))
    result = backend.run(c, shots=shots)

    best_solution = ""
    best_value = float("-inf")
    for bits, count in result.counts.items():
        val = _maxcut_value(bits, edges)
        if val > best_value:
            best_value = val
            best_solution = bits

    return QAOResult(
        num_qubits=num_qubits,
        optimal_angles=(gamma, beta),
        best_solution=best_solution,
        best_value=best_value,
        counts=result.counts,
        shots=shots,
    )


def _maxcut_value(bitstring: str, edges: list[tuple[int, int]]) -> float:
    """Compute MaxCut value for a bitstring."""
    val = 0.0
    for u, v in edges:
        if u < len(bitstring) and v < len(bitstring):
            if bitstring[u] != bitstring[v]:
                val += 1.0
    return val


# ── 4. VQE ────────────────────────────────────────────────────────────────


def _pauli_expectation(
    backend: Backend,
    circuit: Circuit,
    qubit: int,
    basis: str,
    shots: int,
) -> float:
    """Measure expectation of a Pauli operator on a qubit."""
    from qorch.tomography import _rotate_to_basis  # noqa: PLC2701

    c = circuit
    c = _rotate_to_basis(c, qubit, basis)
    c = c.measure(qubit)
    result = backend.run(c, shots=shots)
    p0 = result.counts.get("0", 0) / shots
    p1 = result.counts.get("1", 0) / shots
    return p0 - p1


def _energy_h2(params: tuple[float, ...], backend: Backend, shots: int) -> float:
    """H₂ Hamiltonian energy: E = ⟨ZZ⟩ - ⟨XX⟩ - ⟨YY⟩ + ⟨Z⟩_0 + ⟨Z⟩_1."""
    c = Circuit(2)
    theta = params[0]
    c = c.rx(0, math.pi / 2).rz(0, theta).rx(0, -math.pi / 2)
    c = c.cx(0, 1)
    zz = _pauli_expectation(backend, c, 0, "Z", shots)
    xx = _pauli_expectation(backend, c, 0, "X", shots)
    yy = _pauli_expectation(backend, c, 0, "Y", shots)
    z0 = _pauli_expectation(backend, Circuit(2).rx(0, math.pi / 2).rz(0, theta).rx(0, -math.pi / 2), 0, "Z", shots)
    z1 = _pauli_expectation(backend, Circuit(2).rx(0, math.pi / 2).rz(0, theta).rx(0, -math.pi / 2), 1, "Z", shots)
    return zz - xx - yy + z0 + z1


def run_vqe(
    backend: Backend,
    shots: int = 4096,
    max_iterations: int = 50,
    initial_guess: tuple[float, ...] = (0.1,),
) -> VQEResult:
    """Run VQE for the H₂ molecule (2-qubit Hamiltonian).

    Uses a simple parameter-shift optimizer.

    Args:
        backend: Target backend.
        shots: Shots per expectation value measurement.
        max_iterations: Maximum optimizer steps.
        initial_guess: Initial ansatz parameters.

    Returns:
        ``VQEResult`` with optimal parameters and energy.
    """
    params = list(initial_guess)
    best_val = float("inf")
    best_params = tuple(params)
    nfev = 0
    step = 0.1

    for _ in range(max_iterations):
        val = _energy_h2(tuple(params), backend, shots)
        nfev += 1
        if val < best_val:
            best_val = val
            best_params = tuple(params)

        grad: list[float] = []
        for i in range(len(params)):
            params[i] += step
            v_plus = _energy_h2(tuple(params), backend, shots)
            nfev += 1
            params[i] -= 2 * step
            v_minus = _energy_h2(tuple(params), backend, shots)
            nfev += 1
            params[i] += step
            grad.append((v_plus - v_minus) / (2 * step))

        for i in range(len(params)):
            params[i] -= 0.01 * grad[i]

    return VQEResult(
        num_qubits=2,
        optimal_params=best_params,
        optimal_value=best_val,
        nfev=nfev,
    )


# ── 5. Quantum Phase Estimation ───────────────────────────────────────────


def qpe_circuit(num_qubits: int, unitary: Circuit) -> Circuit:
    """Build a Quantum Phase Estimation circuit.

    Args:
        num_qubits: Number of phase estimation qubits.
        unitary: Circuit implementing the controlled unitary (acts on 1 data qubit).

    Returns:
        ``Circuit`` implementing QPE without final measurement.
    """
    total = num_qubits + 1
    c = Circuit(total)
    for q in range(num_qubits):
        c = c.h(q)

    for i in range(num_qubits):
        for _ in range(1 << i):
            c = _combine_circuits(c, _shift_qubits(unitary, num_qubits - 1 - i, num_qubits))

    # Inverse QFT on phase qubits
    for i in range(num_qubits // 2):
        c = c.swap(i, num_qubits - 1 - i)
    for i in range(num_qubits - 1, -1, -1):
        c = c.h(i)
        for j in range(i):
            angle = -math.pi / (1 << (i - j))
            c = c.cx(j, i)
            c = c.rz(i, angle)
            c = c.cx(j, i)

    return c


def _controlled_gate(gate: Gate, control: int, data_offset: int) -> list[Gate]:
    """Create a controlled version of a gate using CX + RZ decomposition."""
    target = gate.qubits[0] + data_offset
    if gate.name == "rz":
        theta = gate.params[0] if gate.params else 0.0
        return [
            Gate("cx", (control, target)),
            Gate("rz", (target,), (theta / 2,)),
            Gate("cx", (control, target)),
            Gate("rz", (target,), (-theta / 2,)),
        ]
    return [Gate(gate.name, (control, target), gate.params)]


def _shift_qubits(circuit: Circuit, control: int, offset: int) -> Circuit:
    """Shift qubit indices in a circuit by an offset, adding a controlled qubit."""
    from dataclasses import replace
    new_gates: list[Gate] = []
    for g in circuit.gates:
        if len(g.qubits) >= 1:
            new_gates.extend(_controlled_gate(g, control, offset))
        else:
            new_gates.append(g)
    new_num = max(control + 1, offset + circuit.num_qubits)
    return replace(circuit, num_qubits=new_num, gates=tuple(new_gates))


def run_qpe(
    backend: Backend,
    phase_qubits: int,
    unitary: Circuit,
    shots: int = 4096,
) -> QPEResult:
    """Run Quantum Phase Estimation.

    Args:
        backend: Target backend.
        phase_qubits: Number of phase estimation qubits.
        unitary: Circuit implementing the unitary (acts on 1 data qubit).
        shots: Number of shots.

    Returns:
        ``QPEResult`` with estimated phase and measurement counts.
    """
    c = qpe_circuit(phase_qubits, unitary)
    c = c.measure(*range(phase_qubits))
    result = backend.run(c, shots=shots)

    phase = None
    if result.counts:
        top = max(result.counts, key=result.counts.get)
        phase = int(top, 2) / (1 << phase_qubits) if phase_qubits > 0 else 0.0

    return QPEResult(
        num_qubits=phase_qubits,
        phase=phase,
        counts=result.counts,
    )
