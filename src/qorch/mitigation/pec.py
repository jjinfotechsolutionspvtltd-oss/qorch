"""Probabilistic Error Cancellation (PEC).

PEC inverts a known noise channel by Monte Carlo sampling. For each noisy gate,
we randomly insert Pauli gates with signed weights, then re-weight the measurement
results to recover the noiseless expectation value.

This implementation assumes local depolarizing noise:
  Λ(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)

The inverse channel in the Pauli transfer matrix is diag(1, μ, μ, μ)
  where μ = 1/(1 - 4p/3).
Quasi-probability decomposition Λ^{-1} = Σᵢ cᵢ Pᵢ:
  c_I = (1 + 3μ) / 4
  c_X = c_Y = c_Z = (1 - μ) / 4
Sampling overhead γ = |c_I| + 3|c_X|.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from qorch.backends.base import Backend
from qorch.ir import Circuit, Gate, static_gates


@dataclass(frozen=True)
class PECResult:
    """Result of PEC mitigation."""

    raw_value: float
    mitigated_value: float
    n_samples: int
    gamma: float
    noise_prob: float

    @property
    def improvement(self) -> float:
        err_raw = abs(self.raw_value)
        err_mit = abs(self.mitigated_value)
        if err_raw < 1e-12:
            return 0.0
        return (err_raw - err_mit) / err_raw * 100


def _pec_coeffs(p: float) -> tuple[float, float, float, float]:
    """Compute quasi-probability coefficients for inverse depolarizing channel.

    Returns (c_I, c_X, c_Y, c_Z).
    """
    if p == 0:
        return (1.0, 0.0, 0.0, 0.0)
    mu = 1.0 / (1.0 - 4.0 * p / 3.0)
    c_i = (1.0 + 3.0 * mu) / 4.0
    c_p = (1.0 - mu) / 4.0
    return (c_i, c_p, c_p, c_p)


def _sample_pauli(
    rng: random.Random,
    coeffs: tuple[float, float, float, float],
    gamma: float,
) -> tuple[str | None, float]:
    """Sample a Pauli from the quasi-probability distribution.

    Returns (gate_name_or_None, weight_factor) where weight_factor is
    sign(c_P) * gamma.
    """
    r = rng.random()
    cumulative = 0.0
    for i, g in enumerate([None, "x", "y", "z"]):
        cumulative += abs(coeffs[i]) / gamma
        if r < cumulative:
            return (g, gamma * (1.0 if coeffs[i] >= 0 else -1.0))
    return (None, gamma)


def _pec_sample(
    rng: random.Random,
    circuit: Circuit,
    backend: Backend,
    coeffs: tuple[float, float, float, float],
    gamma: float,
    observable: Callable[[dict[str, int]], float],
    shots: int,
) -> tuple[float, float]:
    """Run one PEC sample.

    For each gate in the circuit, sample a Pauli insertion and its weight.
    Returns (weighted_observable, weight).
    """
    pec_gates: list[Gate] = []
    total_weight = 1.0

    for g in static_gates(circuit.gates):
        pec_gates.append(g)
        p_name, w = _sample_pauli(rng, coeffs, gamma)
        total_weight *= w
        if p_name is not None:
            for q in g.qubits:
                pec_gates.append(Gate(p_name, (q,)))

    pec_circuit = Circuit(
        num_qubits=circuit.num_qubits,
        gates=tuple(pec_gates),
        measured=circuit.measured,
    )
    result = backend.run(pec_circuit, shots=shots)
    obs = observable(result.counts)

    return total_weight * obs, total_weight


def pec_expectation(
    backend: Backend,
    circuit: Circuit,
    observable: Callable[[dict[str, int]], float],
    noise_prob: float = 0.05,
    n_samples: int = 50,
    shots_per_sample: int = 200,
    seed: int | None = None,
) -> PECResult:
    """Estimate the noise-free expectation value using PEC.

    Args:
        backend: Noisy backend to run on.
        circuit: Circuit to evaluate.
        observable: Maps counts to a float.
        noise_prob: Assumed depolarizing probability per gate.
        n_samples: Number of PEC samples.
        shots_per_sample: Shots per individual circuit run.
        seed: RNG seed for reproducibility.

    Returns:
        ``PECResult`` with raw and PEC-mitigated values.
    """
    rng = random.Random(seed)
    coeffs = _pec_coeffs(noise_prob)
    gamma = sum(abs(c) for c in coeffs)

    # Raw baseline
    raw_result = backend.run(circuit, shots=n_samples * shots_per_sample)
    raw_value = observable(raw_result.counts)

    # PEC samples
    weighted_sum = 0.0
    total_weight = 0.0
    for _ in range(n_samples):
        val, w = _pec_sample(rng, circuit, backend, coeffs, gamma, observable, shots_per_sample)
        weighted_sum += val
        total_weight += w

    mitigated = weighted_sum / total_weight if total_weight != 0 else raw_value

    return PECResult(
        raw_value=raw_value,
        mitigated_value=mitigated,
        n_samples=n_samples,
        gamma=gamma,
        noise_prob=noise_prob,
    )
