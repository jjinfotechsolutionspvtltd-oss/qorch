# qorch — Indian Quantum Orchestration Layer

[![CI](https://github.com/jjinfotechsolutionspvtltd-oss/qorch/actions/workflows/ci.yml/badge.svg)](https://github.com/jjinfotechsolutionspvtltd-oss/qorch/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: none](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](pyproject.toml)

A sovereign, minimal, correct quantum software stack designed for India's emerging quantum hardware ecosystem. **Hardware-agnostic from day one** — any Indian QPU (from DRDO, ISRO, IITs, C-DAC) plugs in as one `Backend` adapter with zero core changes.

```bash
pip install -e ".[dev]" && python -m pytest      # 840 tests, ~40s, no services required
```

```python
from qorch import Circuit, LocalSimulator

bell = Circuit(2).h(0).cx(0, 1).measure(0, 1)
LocalSimulator(seed=1).run(bell, shots=1000).counts    # {'00': ~500, '11': ~500}
```

## Why qorch?

As India invests in indigenous quantum processors (superconducting at TIFR/DRDO, ion traps at IIT Jodhpur, photonic at IISc), a vendor-neutral software stack is essential. qorch provides:

- **Zero required dependencies** — the core is stdlib-only and never imports Qiskit or Cirq; `import qorch` pulls in nothing third-party. It is fully usable air-gapped.
- **Interoperability without lock-in** — Qiskit is an *optional, opt-in* adapter (`pip install qorch[qiskit]`) so the *same* `Circuit` can also run on Qiskit Aer or IBM hardware. You choose it; nothing in qorch requires it, and no qorch capability depends on a foreign vendor. (numpy/scipy are likewise optional, used only for a couple of benchmark fits.)
- **Sovereign architecture** — clean hardware-abstraction layer designed for Indian hardware adapters; reproducible and auditable.
- **Correct by construction** — immutable IR, 840 tests, mypy-clean, with property and cross-simulator validation.
- **Active research** — error mitigation, tomography, benchmarking, Clifford+T decomposition, dynamic circuits, and a full quantum-error-correction stack.

## Install

```bash
pip install -e .            # dependency-free core
pip install -e .[qiskit]    # optional: Qiskit Aer / IBM hardware adapter
pip install -e .[dev]       # optional: pytest, ruff, mypy, numpy
```

---

## Features

Each feature below states **what** it is, **why** it exists, and **how** to use it.

### 1. Circuit IR + serialization

**What.** An immutable circuit value type with builder methods, plus OpenQASM-3 (subset), JSON, and a compact binary format.
**Why.** A single, correct, hashable representation every layer programs against. Immutability makes transpiler passes safe to compose. Leftmost bit = qubit 0 throughout (one documented convention avoids endianness bugs).

```python
from qorch import Circuit, from_qasm3, to_qasm3

c = Circuit(2).h(0).cx(0, 1).measure(0, 1)
c = from_qasm3('OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0], q[1];')
qasm = to_qasm3(c)
json_str = c.to_json(); c2 = Circuit.from_json(json_str)
```

### 2. Symbolic parameters

**What.** `Parameter` placeholders for gate angles, bound to numbers later with `Circuit.bind`.
**Why.** Variational algorithms (VQE/QAOA) evaluate one ansatz at many angles. Build the circuit once and rebind per iteration instead of rebuilding it — fewer allocations and the standard ergonomic pattern.

```python
from qorch import Circuit, Parameter, LocalSimulator

theta = Parameter("theta")
ansatz = Circuit(1).rx(0, theta).measure(0)     # built once
for angle in (0.0, 1.57, 3.14):
    bound = ansatz.bind({theta: angle})          # or {"theta": angle}
    LocalSimulator(seed=1).run(bound, shots=1000)
print(ansatz.parameters)                          # (Parameter(name='theta'),)
```

### 3. Backends (the hardware-abstraction layer)

**What.** Every execution target — simulator or real QPU — implements the 3-method `Backend` interface (`properties`, `run`, `validate`).
**Why.** The scheduler, mitigation, and benchmarking layers program against this interface only, so a new QPU is one adapter with zero core changes.

| Backend | Qubits | Topology | Native gates | Notes |
|---|---|---|---|---|
| **LocalSimulator** | ~18 | all-to-all | all | dependency-free statevector (+ gate/readout noise, dynamic circuits) |
| **DensitySimulator** | ~8 | all-to-all | all | Kraus density-matrix sim (T1/T2, depolarizing) |
| **StabilizerSimulator** | 100s | all-to-all | Clifford | polynomial-time CHP tableau (for QEC) |
| `iit-jodhpur-ion-trap` | 6 | all-to-all | rx, ry, ms | IIT Jodhpur trapped-ion |
| `tifr-superconducting` | 5 | linear | cx, sx, rz, x | TIFR Mumbai superconducting |
| `drdo-mirai` | 6 | grid 2×3 | cx, rx, rz, x | DRDO MIRAI Lab |
| **QiskitBackend** | any | varies | all 13 gates | IBM / Qiskit Aer adapter (optional) |

```python
from qorch import LocalSimulator, IndianQPU

sim = LocalSimulator(seed=42)
qpu = IndianQPU.from_preset("tifr-superconducting", seed=42)
result = qpu.run(Circuit(2).h(0).cx(0, 1), shots=2000)
```

### 4. Device calibration (Backend API v2)

**What.** Optional `calibration()` / `coupling_map()` hooks expose structured device data (T1/T2, per-edge error, gate durations, topology). `IndianQPU` can also run an exact T1/T2 noise path.
**Why.** Calibration-aware compilation and noise-adaptive scheduling need a structured source of truth, not a single fidelity hint. Simple/simulator backends stay a 3-method adapter (defaults return `None`).

```python
from qorch import IndianQPU, DeviceCalibration
from qorch.backends.density_simulator import DensitySimulator

qpu = IndianQPU.from_preset("tifr-superconducting")
cal: DeviceCalibration = qpu.calibration()
print(cal.qubits[0].t1_us, cal.coupling_map, cal.gate_durations_us)

# exact density-matrix noise driven by T1/T2 from calibration
noisy = IndianQPU.from_preset("tifr-superconducting", seed=1, exact_noise=True)
sim = DensitySimulator.from_calibration(cal, seed=0)
```

### 5. Transpiler — decompose → route → lower → optimize → DD

**What.** Lower any circuit to a target native gate set, route it for limited connectivity (greedy or SabreSWAP lookahead), lower it again, optimize, and insert dynamical decoupling.
**Why.** Real devices have a fixed gate set and topology. Routing is **semantically transparent** — measurements and single-qubit gates are remapped through the final layout, so results are correct.

```
decompose → route → [decompose → fix directions → decompose] → optimize → [insert DD → decompose]
                     └────────────── lower to target ──────────────┘
```

**The pipeline is not a single pass in each direction, and the order is load-bearing:**

- **Decomposition runs twice.** Routing inserts `swap` gates *after* the first pass, and `swap` is native to almost no real target. Without the second pass a `swap` survives into the output of a cx/sx/rz/x target — the circuit looks compiled but cannot run.
- **Direction fixing sits between the two.** Lowering `swap → cx cx cx` emits a *reversed* CX, illegal on a one-way coupling edge. `fix_gate_directions` rewrites it — symmetric gates (`swap`, `ms`) by exchanging operands, `cx` by Hadamard conjugation. Those Hadamards are themselves non-native, hence the final decomposition; it touches only single-qubit gates, so it cannot reintroduce a violation.
- **DD runs last, after the optimizer.** DD sequences are logically the identity (`xy4 = XYXY = I`, `hahn = XX = I`), so an optimizer that saw them would cancel away exactly the pulses you asked for. Inserting afterwards also measures idle windows against the final circuit.

The guarantee: every gate in the output is in `target.basis_gates`, and every two-qubit gate sits on a `coupling_map` edge **in the direction the hardware implements it**.

Every stage is **dynamic-circuit aware**: mid-circuit measurement and reset follow their qubit through the layout permutation, a conditioned gate decomposes into a sequence carrying that same condition, the lookahead router tracks classical-bit hazards so feed-forward never overtakes the measurement that decides it, and the optimizer refuses to cancel across a measurement or between differently-conditioned gates.

```python
from qorch.transpiler import transpile, TIFR_SUPERCONDUCTING, CouplingMap

c = Circuit(3).h(0).cx(0, 2).rx(1, 0.5)          # cx(0,2) is not adjacent → routing inserts SWAPs
cmap = CouplingMap(edges=((0, 1), (1, 0), (1, 2), (2, 1)))
result = transpile(c, target=TIFR_SUPERCONDUCTING, coupling_map=cmap, use_lookahead=True)

{g.name for g in result.gates} <= set(TIFR_SUPERCONDUCTING.basis_gates)   # True

# DD survives the optimizer that would otherwise cancel it
protected = transpile(c, TIFR_SUPERCONDUCTING, coupling_map=cmap, dd_sequence="hahn")
```

Targets: `IIT_JODHPUR_ION_TRAP`, `TIFR_SUPERCONDUCTING`, `DRDO_MIRAI`, `CLIFFORD_T`. **Every supported gate lowers to every target** — including across architectures, so an `ms`-based ion-trap circuit compiles to a CX machine and vice versa. This is enforced exhaustively: each of the 13 gates × 4 gate sets is checked to emit only native gates *and* to preserve the unitary up to global phase.

### 6. SabreSWAP lookahead routing

**What.** A router that scores candidate SWAPs with a DAG front-layer + extended-layer heuristic and optional noise-awareness.
**Why.** Fewer SWAPs than greedy routing, and it can prefer paths through higher-fidelity qubits.

```python
from qorch.transpiler.routing import route_lookahead, CouplingMap, QubitQuality

cmap = CouplingMap(edges=((0, 1), (1, 2), (2, 3)))
quality = {0: QubitQuality(0.99), 1: QubitQuality(0.95), 2: QubitQuality(0.99), 3: QubitQuality(0.90)}
routed = route_lookahead(c, cmap, qubit_quality=quality, lookahead=20, decay=0.5)
```

### 7. Clifford+T decomposition + rotation synthesis

**What.** Decompose arbitrary circuits into the fault-tolerant `{h, cx, t}` gate set with T-count and T-depth reporting, including **meet-in-the-middle synthesis** of arbitrary-angle rotations.
**Why.** T gates dominate fault-tolerant cost (magic-state distillation); T-count is the headline resource metric.

Clifford+T is a *discrete* gate set: multiples of π/4 are exact (powers of T), and every other angle must be approximated. qorch reaches **error ≤ 1e-3 in ~18–22 T gates** — at or under the ~3·log₂(1/ε) an optimal synthesizer needs — and **always reports the error it achieved**.

The search returns the *cheapest* word meeting the requested precision, not the most accurate one it can find. Past the target, extra accuracy is worth nothing and costs T gates.

All three rotation axes cost the same. `rx` is `H·rz·H` and `H` is native, so its conjugation is free; `ry` is searched *directly* rather than via `rz(-π/2)·rx(θ)·rz(π/2)`, which would be exact but spell two Clifford quarter-turns as `T⁶` and `T²` — eight T gates a resource estimate then counts as magic states.

> Solovay–Kitaev is the textbook answer here and the wrong one for this library: it reaches any precision but needs thousands of T gates per rotation, which would inflate exactly the metric the resource estimator consumes. Meet-in-the-middle finds near-minimal words instead.

```python
import math
from qorch.transpiler import decompose_to_clifford_t
from qorch.transpiler.decompose import clifford_t_synthesis_error
from qorch.transpiler.synthesis import synthesize_ry, synthesize_rz

result, t_count, t_depth = decompose_to_clifford_t(Circuit(2).h(0).cx(0, 1).rz(0, 0.3))

synthesize_rz(math.pi / 4)                   # exact=True,  error=0.0,     1 T gate
synthesize_rz(0.3)                           # exact=False, error≈8.9e-4, ~18 T gates
synthesize_rz(0.3, precision=3e-4).t_count   # ask for more accuracy, pay in T gates
synthesize_ry(0.3).t_count                   # 17 — no Clifford-conjugation surcharge

clifford_t_synthesis_error(Circuit(1).rz(0, 0.3))   # worst per-rotation error
```

### 8. Fault-tolerant resource estimation

**What.** Convert a circuit's Clifford+T cost into a first-order surface-code estimate: code distance, physical qubits, and runtime.
**Why.** "How big a machine would this need?" is a compelling, fundable output — and it builds on the T-count the decomposer already computes.

```python
from qorch.resource_estimation import estimate_resources, format_estimate

est = estimate_resources(Circuit(4).h(0).cx(0, 1).t(1).rz(2, 0.3), physical_error_rate=1e-3)
print(format_estimate(est))   # code distance, physical qubits, est. runtime
```

### 9. Algorithm templates (ADP)

**What.** Ready-made algorithm builders/runners: QFT, Grover's search, QAOA (MaxCut), VQE (H₂), and Quantum Phase Estimation.
**Why.** High-level entry points that compose on the IR and run on any `Backend`, returning typed result objects.

```python
from qorch.adp import run_qft, run_grover, run_qaoa, run_vqe, run_qpe

g = run_grover(LocalSimulator(seed=3), num_qubits=3, marked="101", shots=4000)
print(g.top_outcomes[0])      # ('101', ~3800) — marked state amplified
```

### 10. State tomography

**What.** Reconstruct 1Q and 2Q density matrices from Pauli-basis measurements.
**Why.** Characterize a QPU and verify state preparation; reports purity and trace.

```python
from qorch.tomography import state_tomography_1q, purity

rho = state_tomography_1q(LocalSimulator(seed=42), Circuit(1).h(0), shots=4096)
print(f"Purity: {purity(rho.rho):.4f}")   # ~1.0 for a pure state
```

### 11. Error mitigation

**What.** Readout-error correction, zero-noise extrapolation, probabilistic error cancellation, dynamical decoupling, and Pauli twirling.
**Why.** Squeeze useful signal out of noisy NISQ hardware — qorch's research differentiator.

```python
from qorch.mitigation import ReadoutMitigator, zne_expectation
from qorch.mitigation.dd import insert_dd

# Readout-error mitigation from a calibration matrix A[i][j] = P(measure i | prepared j)
mitigator = ReadoutMitigator.from_calibration_matrix(
    labels=["0", "1"], matrix=[[0.95, 0.10], [0.05, 0.90]])
corrected = mitigator.apply(result.counts)

# Zero-noise extrapolation (valid for parametrized circuits — true gate inverses)
zne = zne_expectation(sim, circuit, observable, shots=8192, scales=(1, 3, 5))

# Dynamical decoupling
c_dd = insert_dd(c, sequence="xy4", qubits=(0, 1))
```

### 12. Noise-model builders

**What.** Construct gate/readout/Kraus noise models from device specs (fidelity, T1/T2).
**Why.** Reproduce realistic hardware behavior in simulation to validate mitigation.

```python
from qorch.backends.simulator import GateNoise, ReadoutNoise
from qorch.backends.density_simulator import NoiseChannel

chan = NoiseChannel.from_gate_fidelity(gate_fidelity=0.99, t1_us=50, t2_us=80, gate_time_us=0.1)
readout = ReadoutNoise.from_readout_fidelity(0.95, asymmetric=True)
gates = GateNoise.from_gate_fidelity(0.995)
```

### 13. Benchmarking & certification

**What.** Randomized Benchmarking, Quantum Volume (+ sweep), and cross-entropy benchmarking, plus a `certify` CLI suite (Bell, CHSH, RB, QV).
**Why.** Vendor-neutral "is this QPU any good?" metrics for indigenous hardware — a standards play.

```python
from qorch.benchmarking import qv_sweep, randomized_benchmarking

qv = qv_sweep(LocalSimulator(seed=42), start_width=2, end_width=5, trials=20, shots=4096)
print(f"QV = 2^{qv.max_passing_width} = {qv.quantum_volume}")
```

### 14. QMI binary format

**What.** A compact binary encoding of circuits for QPU microcode / firmware transfer (with input-validated decoding).
**Why.** Low-latency job submission; 4–10× smaller than JSON/QASM.

```python
from qorch.qmi import to_qmi, from_qmi, QMIEncoder

data = to_qmi(circuit); c2 = from_qmi(data)        # roundtrip
print(QMIEncoder.hexdump(data))
```

### 15. Dynamic circuits (mid-circuit measurement + feed-forward)

**What.** Classical registers, mid-circuit measurement, reset, and classically-conditioned gates (multi-bit conditions).
**Why.** The basis for teleportation, repeat-until-success, and quantum error correction — capabilities a static circuit cannot express.

```python
from qorch.dynamic import run_teleportation, run_repetition_code

# Teleport |1> with feed-forward X/Z corrections
run_teleportation(LocalSimulator(seed=1), state_prep=Circuit(1).x(0))   # {'1': ~1.0}

# 3-qubit bit-flip code corrects a single error via 2-bit syndrome decoding
run_repetition_code(LocalSimulator(seed=1), state_prep=Circuit(1).x(0), error_qubit=1)

# Build dynamic circuits directly:
c = (Circuit(2, num_clbits=2)
     .h(0).measure_into(0, 0)     # mid-circuit measurement → classical bit 0
     .x_if(1, 0, 1)               # feed-forward: X on qubit 1 if bit 0 == 1
     .measure_into(1, 1))

# Dynamic circuits compile like any other — same classical-register distribution
from qorch.transpiler import transpile, TIFR_SUPERCONDUCTING, CouplingMap
from qorch.dynamic import repetition_code_circuit

code = repetition_code_circuit(error_qubit=1)
native = transpile(code, TIFR_SUPERCONDUCTING,
                   coupling_map=CouplingMap(TIFR_SUPERCONDUCTING.coupling_map))
LocalSimulator(seed=7).run(native, shots=200).counts    # {'110': 200} — syndrome 11, logical 0
```

### 16. Quantum error correction

**What.** A polynomial-time stabilizer (CHP tableau) simulator, a distance-d repetition code, the Steane [[7,1,3]] code, and a toric surface code with an exact MWPM decoder and a verified ~10% threshold.
**Why.** The full near-term QEC research loop — stabilizer simulation, syndrome extraction, decoding, and threshold estimation — on a dependency-free core.

```python
from qorch import StabilizerSimulator
from qorch.qec import run_repetition, run_steane, repetition_logical_error_rate
from qorch.surface_code import toric_logical_error_rate

# Stabilizer simulator scales where statevector cannot (40+ qubit Clifford circuits)
sim = StabilizerSimulator(seed=1)

run_repetition(5, logical=1, errors=(2,)).corrected     # True (distance-5 corrects 2 errors)
run_steane(logical=1, error=("x", 4)).corrected         # True (Steane corrects any 1-qubit X/Y)

# Threshold: below ~10%, larger code distance suppresses the logical error rate
repetition_logical_error_rate(7, 0.08, trials=4000)     # << rate at d=3
toric_logical_error_rate(5, 0.05, trials=6000)          # < rate at distance 3
```

### 17. Batch scheduler

**What.** Route multiple circuits to best-fit backends with minimum qubit waste.
**Why.** A pluggable selection policy that grows into cost/latency-aware routing.

```python
from qorch.scheduler import Scheduler, BatchJob

sched = Scheduler(backends=[sim1, qpu2])
results = sched.run_batch([BatchJob(circuit=c1, shots=1024, label="bell")])
```

### 18. Pulse-level control

**What.** Complex baseband waveforms (constant / Gaussian / DRAG), an exact rotating-frame single-qubit simulator, and calibration helpers that reproduce gates from physical pulses.
**Why.** The gate IR hides *how* a gate is physically realized; calibration and pulse-aware compilation bottom out here. Closed-form per-slice evolution, so it stays dependency-free.

```python
from qorch import Waveform, x_pulse, sx_pulse, calibrate_gaussian, pulse_unitary

from qorch.pulse import equals_gate

equals_gate(pulse_unitary(x_pulse()), (0, 1, 1, 0))   # True — the π pulse *is* X
sx = pulse_unitary(sx_pulse())                        # two of these compose to X

# A θ-area pulse drives |0> to P(1) = sin²(θ/2) — exact Rabi physics
p1 = abs(pulse_unitary(calibrate_gaussian(1.3))[2]) ** 2   # index 2 = U[1,0]

Waveform.drag(duration=20, amp=1.0, sigma=5.0, beta=0.3)   # leakage-suppressing DRAG
```

### 19. CLI

```text
python -m qorch <command> [options]

  run        — Execute a circuit on a backend
  batch      — Compare backends on the same circuit
  sched      — Batch-schedule multiple circuits
  mitigate   — Run with error mitigation (readout / zne / pec)
  transpile  — Decompose for an Indian QPU
  report     — Circuit depth, gate count, fidelity analysis
  certify    — Run QPU certification suite (Bell, CHSH, RB, QV)
  qv-sweep   — Quantum Volume sweep across widths
  list       — List backends and techniques
```

```bash
python -m qorch run --gates "h0,cx01" --backend tifr-superconducting
python -m qorch certify --backend local-simulator --shots 4096
python -m qorch qv-sweep --backend local-simulator --start 2 --end 5
python -m qorch transpile --gates "h0,cx01" --target tifr-superconducting
```

---

## Architecture

```
src/qorch/
  ir.py                  # immutable IR (gates, params, dynamic ops) + QASM-3 + JSON
  adp.py                 # algorithm templates: QFT, Grover, QAOA, VQE, QPE
  dynamic.py             # teleportation + repetition code (dynamic circuits)
  pulse.py               # waveforms + rotating-frame pulse simulator + calibration
  qec.py                 # repetition-d + Steane [[7,1,3]] codes + threshold
  surface_code.py        # toric code geometry + MWPM decoder + 2D threshold
  resource_estimation.py # fault-tolerant resource estimates from T-count
  tomography.py          # 1Q + 2Q state tomography
  entanglement.py        # Bell fidelity, CHSH, entanglement witness
  benchmarking.py        # RB, QV (+ sweep), XEB
  qmi.py                 # QMI binary format (validated decode)
  scheduler.py           # FIFO queue + best-fit batch scheduler
  analysis.py            # circuit depth / gate count / fidelity
  visual.py              # ASCII circuit drawing
  cli.py                 # command-line interface (9 commands)
  backends/
    base.py              # Backend HAL + BackendProperties + DeviceCalibration + JobResult
    simulator.py         # dependency-free statevector (+ noise + dynamic execution)
    density_simulator.py # Kraus-operator density-matrix simulation (T1/T2)
    stabilizer.py        # CHP tableau simulator (polynomial-time Clifford)
    indian_backend.py    # Indian QPU adapter (IIT Jodhpur, TIFR, DRDO MIRAI)
    qiskit_backend.py    # IBM / Qiskit Aer adapter (optional dependency)
  transpiler/
    gateset.py           # Indian-native + Clifford+T gate-set definitions
    decompose.py         # recursive decomposition (incl. Clifford+T)
    synthesis.py         # meet-in-the-middle Rz → Clifford+T, with reported error
    routing.py           # greedy + SabreSWAP routing (layout-correct) + edge-direction fixing
    optimizer.py         # gate cancellation + rotation merging
  mitigation/
    readout.py  zne.py  pec.py  dd.py  twirling.py  pipeline.py
tests/                   # 840 unit tests (~95% coverage)
```

## Tests

```bash
python -m pytest                 # 840 tests
python -m pytest --cov=qorch     # with coverage (~95%)
ruff check src/ tests/           # lint
mypy src/                        # type check
```

## Indian QPU backends

| Backend | Qubits | Topology | Native gates | Modeled on |
|---|---|---|---|---|
| `iit-jodhpur-ion-trap` | 6 | all-to-all | rx, ry, ms | IIT Jodhpur trapped-ion |
| `tifr-superconducting` | 5 | linear | cx, sx, rz, x | TIFR Mumbai superconducting |
| `drdo-mirai` | 6 | grid 2×3 | cx, rx, rz, x | DRDO MIRAI Lab |

## Mitigation benchmark

| Technique | Error reduction |
|---|---|
| Readout calibration | ~95% |
| ZNE extrapolation | ~40% |
| Dynamical decoupling | inserts XY-4 / XY-8 / CPMG / Hahn sequences |

## Strategic context

qorch is a research project focused on **Indian quantum readiness**:
- Clean HAL (with calibration + async-ready hooks) that any future Indian QPU implements against
- Transpiler targeting Indian-native + Clifford+T gate sets, with layout-correct routing
- Tomography, QV sweep, and certification for vendor-neutral hardware validation
- Dynamic circuits + a full QEC stack (stabilizer sim, codes, surface-code threshold)
- QMI binary format for low-latency QPU communication
- No foreign vendor lock-in — sovereign, dependency-free core

## Contributing

Contributions are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers setup, the five
rules that govern every change (the dependency-free core is the load-bearing one), and
what a convincing test looks like in a codebase where bugs return plausible-looking
wrong answers rather than crashing.

Good places to start: compiler passes (layout, commutation-based cancellation,
scheduling), QEC decoders, and new backend adapters behind the existing HAL. Issues
labeled `good first issue` are scoped to be self-contained.

Participation is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). To report a
vulnerability, see [SECURITY.md](SECURITY.md) — please don't open a public issue.

## License

[Apache License 2.0](LICENSE) — Copyright 2026 JJISPL Quantum Technologies.

Permissive, with an explicit patent grant. Use it commercially, modify it, redistribute
it; keep the notice and state your changes.
```
