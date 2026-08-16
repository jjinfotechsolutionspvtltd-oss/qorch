"""Quantum benchmarking suite: Randomized Benchmarking, Quantum Volume, XEB.

All benchmarks run against any ``Backend`` — local simulator or real QPU —
enabling standardised quality metrics for Indian quantum hardware.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from dataclasses import replace

from qorch.ir import Circuit, Gate, inverse_gates
from qorch.backends.base import Backend

# ── utilities ────────────────────────────────────────────────────────────────

# 1-qubit Clifford gates (24). We use canonical generators.
# Clifford group on 1 qubit has 24 elements.
_CLIFFORD_1Q: tuple[tuple[str, ...], ...] = (
    ("id",),
    ("h",),
    ("x",),
    ("y",),
    ("z",),
    ("sx",),
    ("h", "x"),
    ("h", "y"),
    ("h", "z"),
    ("x", "y"),
    ("x", "z"),
    ("y", "z"),
    ("sx", "h"),
    ("h", "sx"),
    ("x", "h"),
    ("h", "x", "h"),
    ("y", "h"),
    ("z", "h"),
    ("sx", "x"),
    ("x", "sx"),
    ("sx", "z"),
    ("z", "sx"),
    ("sx", "h", "x"),
    ("h", "x", "h", "z"),
)

def _random_clifford_1q(rng: random.Random) -> tuple[str, ...]:
    """Return a random 1-qubit Clifford sequence (from canonical generators)."""
    return _CLIFFORD_1Q[rng.randint(0, len(_CLIFFORD_1Q) - 1)]


# ── 1. Randomized Benchmarking ─────────────────────────────────────────────

@dataclass
class RBResult:
    """Result of a Randomized Benchmarking run."""

    depths: list[int]
    survival_probabilities: list[float]
    estimated_error_rate: float | None
    fitted_curve: tuple[float, float, float] | None  # A, B, p


def randomized_benchmarking(
    backend: Backend,
    num_qubits: int = 1,
    depths: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    circuits_per_depth: int = 10,
    shots: int = 1024,
    seed: int | None = None,
) -> RBResult:
    """Run 1-qubit Randomized Benchmarking.

    Returns survival probabilities at each Clifford depth, plus the
    estimated error rate per Clifford from an exponential fit.
    """
    rng = random.Random(seed)
    survival_probs: list[float] = []

    for depth in depths:
        probs: list[float] = []
        for _ in range(circuits_per_depth):
            # Forward: a random sequence of `depth` Clifford generators on qubit 0.
            forward: list[Gate] = []
            for _ in range(depth):
                for name in _random_clifford_1q(rng):
                    forward.append(Gate(name, (0,)))
            # Recovery: the exact inverse of the whole forward sequence
            # (reverse order, invert each gate). Noiseless ⇒ returns to |0⟩.
            recovery: list[Gate] = []
            for g in reversed(forward):
                recovery.extend(inverse_gates(g))

            c = replace(Circuit(num_qubits), gates=tuple(forward + recovery)).measure(0)
            result = backend.run(c, shots=shots)
            # Survival = probability of |0⟩ (only qubit 0 measured ⇒ key is "0"/"1")
            p0 = result.counts.get("0", 0) / shots
            probs.append(p0)

        survival_probs.append(sum(probs) / len(probs) if probs else 0.0)

    # Fit exponential P = A·p^depth + B. The 3-parameter scipy fit needs at least
    # 4 depths to be well-determined; with fewer points (or no scipy, or a fit that
    # fails to converge) fall back to a simple per-depth-decay estimate.
    p_est = None
    coeffs = None

    def _simple_estimate() -> float | None:
        if len(survival_probs) >= 2:
            ratios = [survival_probs[i + 1] / survival_probs[i]
                      for i in range(len(survival_probs) - 1)
                      if survival_probs[i] > 0]
            if ratios:
                return 1.0 - (sum(ratios) / len(ratios))
        return None

    if depths and survival_probs and len(depths) >= 4:
        try:
            import numpy as np
            from scipy.optimize import curve_fit

            def _exp_decay(x, a, b, p):
                return a * (p ** x) + b

            xdata = np.array(depths, dtype=float)
            ydata = np.array(survival_probs, dtype=float)
            init_guess = [0.5, 0.5, 0.99]
            fit, _ = curve_fit(_exp_decay, xdata, ydata, p0=init_guess, maxfev=5000)
            a_fit, b_fit, p_fit = fit
            p_est = (1 - p_fit) * (2**num_qubits - 1) / 2**num_qubits
            coeffs = (a_fit, b_fit, p_fit)
        except Exception:
            # No scipy, or the fit failed to converge — use the simple estimate.
            p_est = _simple_estimate()
    elif depths and survival_probs:
        p_est = _simple_estimate()

    return RBResult(
        depths=list(depths),
        survival_probabilities=survival_probs,
        estimated_error_rate=p_est,
        fitted_curve=coeffs,
    )


# ── 2. Quantum Volume ──────────────────────────────────────────────────────

@dataclass
class QVResult:
    """Result of a Quantum Volume measurement."""

    width: int
    depth: int
    heavy_output_probability: float | None
    success: bool | None  # True if HOP > 2/3 with statistical confidence


def _heavy_outputs(probabilities: dict[int, float] | None) -> set[int]:
    """Indices whose ideal probability exceeds the median — the heavy outputs.

    This is the definition the Quantum Volume protocol actually uses, and it is
    a property of *each circuit's own* output distribution.

    The previous implementation returned every bitstring whose integer value was
    at least 2^(n-1) — the top half of the numeric range — on the stated grounds
    that "the ideal output distribution has half its probability" there. That is
    true only of a *uniform* distribution, and QV exists precisely because a
    random circuit's distribution is not uniform: it is Porter-Thomas, which is
    what puts ~0.85 of the weight in the heavy half rather than 0.5. Measuring
    the numeric top half instead simply samples an arbitrary 50% of outcomes, so
    an ideal device scored ~0.35–0.5 and could never pass.
    """
    if not probabilities:
        return set()
    ordered = sorted(probabilities.values())
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    return {index for index, p in probabilities.items() if p > median}


def _haar_su2(rng: random.Random, qubit: int) -> list[Gate]:
    """A Haar-random single-qubit rotation, as rz·ry·rz.

    ``beta`` is drawn from ``2·arccos(sqrt(u))`` rather than uniformly: uniform
    Euler angles are *not* uniform on SU(2), and a QV circuit built from them
    does not produce the Porter-Thomas distribution the benchmark assumes.
    """
    alpha = rng.uniform(0, 2 * math.pi)
    beta = 2.0 * math.acos(math.sqrt(rng.random()))
    gamma = rng.uniform(0, 2 * math.pi)
    return [
        Gate("rz", (qubit,), (alpha,)),
        Gate("ry", (qubit,), (beta,)),
        Gate("rz", (qubit,), (gamma,)),
    ]


def _random_su4(rng: random.Random, q0: int, q1: int) -> list[Gate]:
    """A random two-qubit gate with the structure of a general SU(4).

    Any SU(4) factors as (A₁⊗A₂)·N(a,b,c)·(B₁⊗B₂) — local rotations either side
    of a canonical entangler — and the entangler needs three CNOTs. Building
    that shape with random local rotations and random interaction angles gives
    the entangling power a QV layer requires.

    The previous circuit was a fixed h-cx-rz-rz-cx-h with two random angles.
    That is one point in SU(4) with two knobs, not a random element of it, so
    the output distribution never became Porter-Thomas.
    """
    gates: list[Gate] = []
    gates += _haar_su2(rng, q0) + _haar_su2(rng, q1)
    gates.append(Gate("cx", (q1, q0)))
    gates.append(Gate("rz", (q0,), (rng.uniform(0, 2 * math.pi),)))
    gates.append(Gate("ry", (q1,), (rng.uniform(0, 2 * math.pi),)))
    gates.append(Gate("cx", (q0, q1)))
    gates.append(Gate("ry", (q1,), (rng.uniform(0, 2 * math.pi),)))
    gates.append(Gate("cx", (q1, q0)))
    gates += _haar_su2(rng, q0) + _haar_su2(rng, q1)
    return gates


def _su4_su2_su4_circuit(num_qubits: int, rng: random.Random) -> Circuit:
    """A width-n, depth-n Quantum Volume model circuit.

    Each layer shuffles the qubits into pairs and applies a random SU(4) to
    each, which is the model circuit the QV protocol specifies.
    """
    c = Circuit(num_qubits)
    for _layer in range(num_qubits):
        qubits = list(range(num_qubits))
        rng.shuffle(qubits)
        for i in range(0, num_qubits - 1, 2):
            for gate in _random_su4(rng, qubits[i], qubits[i + 1]):
                c = c._add(gate.name, *gate.qubits, params=gate.params)
    return c


def quantum_volume(
    backend: Backend,
    width: int,
    shots: int = 8192,
    trials: int = 20,
    seed: int | None = None,
) -> QVResult:
    """Measure Quantum Volume at a given width.

    Quantum Volume QV = 2^width if heavy output probability > 2/3
    with >97.5% confidence (binomial test).
    """
    rng = random.Random(seed)
    hop_list: list[float] = []

    for _ in range(trials):
        model = _su4_su2_su4_circuit(width, rng)
        circuit = model.measure(*range(width))
        result = backend.run(circuit, shots=shots)

        # Heavy outputs are defined by each circuit's own *ideal* distribution,
        # so they have to be computed from a noiseless simulation of that
        # circuit — not from the measured counts, which is what the device is
        # being judged against.
        heavy = _heavy_outputs(_compute_ideal_probs(model, width))
        total_heavy = sum(
            count for bitstring, count in result.counts.items()
            if int(bitstring, 2) in heavy
        )
        hop = total_heavy / shots if shots > 0 else 0.0
        hop_list.append(hop)

    mean_hop = sum(hop_list) / len(hop_list) if hop_list else None

    # Statistical test: probability that mean HOP > 2/3
    success = None
    if mean_hop is not None:
        n_trials = len(hop_list)
        se = math.sqrt(2/3 * 1/3 / n_trials) if n_trials > 0 else 0
        z = (mean_hop - 2/3) / se if se > 0 else 0
        success = z > 1.96  # 97.5% one-sided confidence

    return QVResult(
        width=width,
        depth=width,
        heavy_output_probability=mean_hop,
        success=success,
    )


# ── 2b. Quantum Volume Sweep ────────────────────────────────────────────

@dataclass
class QVSweepResult:
    """Result of a Quantum Volume sweep across multiple widths."""

    results: list[QVResult]
    max_passing_width: int
    quantum_volume: int  # = 2^max_passing_width


def qv_sweep(
    backend: Backend,
    start_width: int = 2,
    end_width: int | None = None,
    trials: int = 20,
    shots: int = 8192,
    stop_on_fail: bool = True,
    seed: int | None = None,
) -> QVSweepResult:
    """Sweep Quantum Volume across a range of widths.

    Runs ``quantum_volume`` at each width from ``start_width`` to ``end_width``
    (or until a width fails, if ``stop_on_fail`` is True).

    Returns the maximum QV achieved, defined as 2^{max_passing_width}.
    """
    from qorch.backends.base import BackendProperties

    props: BackendProperties = backend.properties()
    max_qubits = end_width or props.num_qubits
    max_qubits = min(max_qubits, props.num_qubits)

    sweep_results: list[QVResult] = []
    last_success = 0

    for w in range(start_width, max_qubits + 1):
        r = quantum_volume(backend, width=w, shots=shots, trials=trials, seed=seed)
        sweep_results.append(r)
        if r.success:
            last_success = w
        elif stop_on_fail:
            break

    return QVSweepResult(
        results=sweep_results,
        max_passing_width=last_success,
        quantum_volume=1 << last_success,
    )


# ── 3. Cross-Entropy Benchmarking ──────────────────────────────────────────

@dataclass
class XEBResult:
    """Result of a cross-entropy benchmarking run."""

    depth: int
    num_circuits: int
    linear_xeb: float | None
    estimated_fidelity: float | None


def _random_su2_circuit(num_qubits: int, depth: int, rng: random.Random) -> Circuit:
    """Generate a random circuit of single-qubit SU(2) rotations + CX."""
    c = Circuit(num_qubits)
    for d in range(depth):
        # Single-qubit random rotations
        for q in range(num_qubits):
            theta = rng.uniform(0, 2 * math.pi)
            phi = rng.uniform(0, 2 * math.pi)
            if d % 2 == 0:
                c = c.rz(q, theta).rx(q, phi)
            else:
                c = c.rx(q, theta).rz(q, phi)
        # CX layer on alternating pairs
        for q in range(0, num_qubits - 1, 2):
            if (d + q) % 2 == 0:
                c = c.cx(q, q + 1)
    return c


def cross_entropy_benchmarking(
    backend: Backend,
    num_qubits: int,
    depth: int,
    num_circuits: int = 10,
    shots: int = 4096,
    seed: int | None = None,
) -> XEBResult:
    """Run cross-entropy benchmarking.

    Computes linear XEB fidelity for random circuits at given depth.
    """
    rng = random.Random(seed)
    xeb_values: list[float] = []

    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return XEBResult(
            depth=depth,
            num_circuits=num_circuits,
            linear_xeb=None,
            estimated_fidelity=None,
        )

    for _ in range(num_circuits):
        circuit = _random_su2_circuit(num_qubits, depth, rng)
        circuit = circuit.measure(*range(num_qubits))
        result = backend.run(circuit, shots=shots)

        # Compute ideal probabilities using statevector simulation
        # (fallback: uniform for noisy data)
        ideal_probs = _compute_ideal_probs(circuit, num_qubits)
        if ideal_probs is None:
            continue

        # Standard linear cross-entropy benchmarking fidelity (Google 2019):
        #   F_XEB = 2ⁿ · Σ_x p_ideal(x) p_exp(x) − 1
        # which is 0 for a fully-depolarized (uniform) output and → 1 for the
        # noiseless distribution in the Porter–Thomas limit.
        N = 1 << num_qubits
        total = 0.0
        for bitstring, count in result.counts.items():
            idx = int(bitstring, 2)
            p_ideal = ideal_probs.get(idx, 0.0)
            total += p_ideal * (count / shots)
        f_xeb = N * total - 1.0
        xeb_values.append(f_xeb)

    mean_xeb = float(np.mean(xeb_values)) if xeb_values else None
    fidelity = max(0.0, min(1.0, mean_xeb)) if mean_xeb is not None else None

    return XEBResult(
        depth=depth,
        num_circuits=num_circuits,
        linear_xeb=mean_xeb,
        estimated_fidelity=fidelity,
    )


def _compute_ideal_probs(
    circuit: Circuit, num_qubits: int
) -> dict[int, float] | None:
    """Compute ideal output probabilities via exact statevector simulation.

    Uses qorch's dependency-free ``LocalSimulator`` evolution (no sampling) so
    XEB has a noiseless reference distribution to score against.
    """
    from qorch.backends.simulator import LocalSimulator

    amps = LocalSimulator()._evolve(circuit)
    return {i: abs(a) ** 2 for i, a in enumerate(amps)}
