# qorch — Indian Quantum Orchestration Layer

[![CI](https://github.com/jjinfotechsolutionspvtltd-oss/qorch/actions/workflows/ci.yml/badge.svg)](https://github.com/jjinfotechsolutionspvtltd-oss/qorch/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Dependencies: none](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](pyproject.toml)

A sovereign, minimal, correct quantum software stack designed for India's emerging quantum hardware ecosystem. **Hardware-agnostic from day one** — any Indian QPU (from DRDO, ISRO, IITs, C-DAC) plugs in as one `Backend` adapter with zero core changes.

```bash
pip install -e ".[dev]" && python -m pytest      # 1032 tests, ~40s, no services required
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
- **Correct by construction** — immutable IR, 1032 tests, mypy-clean, with property and cross-simulator validation.
- **Active research** — error mitigation, tomography, benchmarking, Clifford+T decomposition, dynamic circuits, and a full quantum-error-correction stack.

## Getting started — Linux, macOS, Windows

The core has **no third-party dependencies and no compiled extensions**, so the same source runs identically on all three platforms. Nothing below needs a compiler, a service, or a GPU. Python 3.11+ is the only requirement.

### Linux / macOS

```bash
git clone https://github.com/jjinfotechsolutionspvtltd-oss/qorch.git
cd qorch
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

### Windows — PowerShell

```powershell
git clone https://github.com/jjinfotechsolutionspvtltd-oss/qorch.git
cd qorch
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m pytest
```

> If activation is blocked, PowerShell's execution policy is the cause:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### Windows — cmd.exe

```bat
py -m venv .venv
.venv\Scripts\activate.bat
pip install -e ".[dev]"
python -m pytest
```

### Optional extras

| extra | command | platforms | notes |
|---|---|---|---|
| *(none)* | `pip install -e .` | Linux, macOS, Windows | the whole core; stdlib only, works air-gapped |
| `dev` | `pip install -e ".[dev]"` | all | pytest, ruff, mypy, numpy |
| `qiskit` | `pip install -e ".[qiskit]"` | all | Qiskit Aer / IBM adapter |
| `gpu` | `pip install -e ".[gpu]"` | Linux, Windows **only** | CuPy needs CUDA; **macOS has none**. ⚠️ [unverified](#2b-gpu-kernel--️-shipped-unverified) |

### Platform notes worth knowing

- **Quoting.** Use `pip install -e ".[dev]"` everywhere. Unquoted `.[dev]` works in `cmd.exe` and PowerShell but is glob-expanded by `zsh`, the default shell on macOS.
- **`python` vs `python3`.** macOS and most Linux distributions reserve `python3`; Windows ships the `py` launcher. Inside an activated venv, plain `python` is correct on all three.
- **Line endings.** Everything is text and no test depends on newline style, so `core.autocrlf` needs no special setting.
- **GPU on macOS.** Apple hardware has no CUDA, so the `gpu` extra cannot install. The pure-Python and numpy kernels are unaffected and are what the library uses by default.

---

## Features

Each feature below states **what** it is, **why** it exists, and **how** to use it.

| | | |
|---|---|---|
| [1. Circuit IR](#1-circuit-ir--serialization) | [2. Symbolic parameters](#2-symbolic-parameters) | [2b. GPU kernel ⚠️](#2b-gpu-kernel--️-shipped-unverified) |
| [3. Backends (HAL)](#3-backends-the-hardware-abstraction-layer) | [4. Device calibration](#4-device-calibration-backend-api-v2) | [5. Transpiler pipeline](#5-transpiler--decompose--route--lower--optimize--dd) |
| [6. SabreSWAP routing](#6-sabreswap-lookahead-routing) | [7. Clifford+T synthesis](#7-cliffordt-decomposition--rotation-synthesis) | [8. FT resource estimation](#8-fault-tolerant-resource-estimation) |
| [9. Algorithm templates](#9-algorithm-templates-adp) | [10. State tomography](#10-state-tomography) | [11. Error mitigation](#11-error-mitigation) |
| [12. Noise builders](#12-noise-model-builders) | [13. Benchmarking](#13-benchmarking--certification) | [14. QMI format](#14-qmi-binary-format) |
| [15. Dynamic circuits](#15-dynamic-circuits-mid-circuit-measurement--feed-forward) | [16. Error correction](#16-quantum-error-correction) | [17. Batch scheduler](#17-batch-scheduler) |
| [18. Pulse control](#18-pulse-level-control) | [19. Pass manager](#19-pass-manager--transpile-metrics) | [20. Gate registry](#20-gate-registry) |
| [21. Layout pass](#21-layout-pass) | [22. Fusion + commutation](#22-euler-fusion--commutation-cancellation) | [23. Cost model](#23-calibration-cost-model) |
| [24. Scheduling & timing](#24-scheduling--timing) | [25. Async execution](#25-async-execution-submitpollcancel) | [26. JobResult v2](#26-jobresult-v2) |
| [27. Fan-out scheduler](#27-fan-out-scheduler--job-store) | [28. Certification suite](#28-certification-suite) | [29. Tool layer (MCP)](#29-tool-layer-mcp-ready) |
| [30. Offline copilot](#30-grounded-offline-copilot) | [31. Photonic IR](#31-photonic-ir-family) | [32. Neutral atoms](#32-neutral-atom-arrays) |
| [33. CLI](#33-cli) | | |

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

### 2b. GPU kernel — ⚠️ shipped unverified

**Status: this has never been run on a CUDA device by its authors.** It is included so that someone with hardware can try it, not because it is known to work.

**What is tested.** The whole kernel algorithm. `evolve_with` takes its array module as a parameter, so CI runs the identical code path with numpy injected and compares against the pure-Python kernel across 40 random circuits. The maths is not guesswork.

**What is not tested.** Importing CuPy, detecting a device, and moving the statevector to and from device memory. If it breaks, those four lines are where to look.

```python
LocalSimulator(use_gpu=True)      # opt-in only; warns that it is unverified
```

It is **never selected automatically**. The numpy kernel switches on above a *measured* 8-qubit crossover; no such measurement exists for a GPU, and inventing a threshold would present a guess as a tuning decision. If you have hardware and it works, measuring that crossover is the natural next contribution.

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

### 19. Pass manager + transpile metrics

**What.** The compile pipeline as an ordered list of named passes, with per-pass cost recorded.
**Why.** A transpiled circuit is far larger than its input, and without attribution you cannot tell which stage caused the growth. The ordering constraints are also real and subtle, and expressing them as a value rather than as statement order makes them inspectable and extensible.

```python
from qorch.transpiler import build_pass_manager, transpile_with_layout, TIFR_SUPERCONDUCTING, CouplingMap

line = CouplingMap(TIFR_SUPERCONDUCTING.coupling_map)
[name for name, _ in build_pass_manager(TIFR_SUPERCONDUCTING, line).passes]
# ['decompose', 'route', 'fuse', 'lower', 'optimize']

result = transpile_with_layout(Circuit(5).h(0).cx(0, 4), TIFR_SUPERCONDUCTING, coupling_map=line)
print(result.metrics.format())    # gates in→out, SWAPs, depth, and a per-pass breakdown
```

`swaps_inserted` is counted the moment routing finishes, **before** lowering turns each SWAP into a CX triple — counting afterwards reports zero on every target without a native SWAP, which is most of them.

### 20. Gate registry

**What.** One `GateDef` per gate — arity, params, matrix, `is_clifford`, `self_inverse`, `angle_inverse`, duration — with every other table derived from it.
**Why.** Gate facts previously lived in five places with nothing keeping them consistent, and they drifted: the optimizer believed `sx` was self-inverse. Since `SX·SX = X`, it cancelled `sx sx` pairs and turned a circuit outputting 1 into one outputting 0.

```python
from qorch.gates import GATES, gate_matrix, gate_duration_ns, CLIFFORD_GATES

gate_matrix("rz", (0.7,))       # the single definition every layer uses
gate_duration_ns("cx")          # 300.0 — advisory, overridden by DeviceCalibration
"t" in CLIFFORD_GATES           # False — T is what Clifford+T accounting counts
```

### 21. Layout pass

**What.** Choose where logical qubits *start*, before routing moves them.
**Why.** Routing began from the identity placement, an arbitrary choice it then paid to correct. Qubits that interact constantly but start at opposite ends of a line cost SWAPs a better placement avoids outright.

```python
transpile_with_layout(circuit, TIFR_SUPERCONDUCTING, coupling_map=line,
                      layout_method="dense")          # or "noise-adaptive", "cost-aware"
```

Measured on the TIFR line: a repeated distant pair goes 4 SWAPs → 0, a star 4 → 2, two distant pairs 6 → 0. Default stays `"trivial"`, so nothing changes unless asked.

### 22. Euler fusion + commutation cancellation

**What.** Collapse runs of single-qubit gates into `Rz·Ry·Rz`, and cancel inverse pairs separated only by gates they commute past.
**Why.** The base optimizer only combines gates already adjacent. A run of nine single-qubit gates is one rotation however long it is, and `rz · cx · rz⁻¹` never becomes adjacent even though `rz` commutes through a CX control.

```python
from qorch.transpiler import fuse_single_qubit_runs, cancel_commuting

cancel_commuting(fuse_single_qubit_runs(circuit))     # 35% fewer gates on random circuits
```

Fusion keeps its rewrite **only if it survives lowering**: collapsing to `Rz·Ry·Rz` is fewer gates as written, but a target with no native `ry` expands it into more than fusion saved.

### 23. Calibration cost model

**What.** Predict the probability a circuit produces the right answer on a specific device — gate error, read-out error, and decoherence from the schedule.
**Why.** Every other compiler decision optimizes a proxy. A SWAP across two excellent qubits can beat a direct gate on a bad pair, and no amount of SWAP counting discovers that.

```python
from qorch.transpiler import estimate_cost, compare_costs

estimate = estimate_cost(circuit, qpu.calibration())
print(estimate.format())        # success probability, split by loss source
compare_costs({"a": circuit_a, "b": circuit_b}, calibration)    # ranked best first
```

Used by `layout_method="cost-aware"`, which routes each candidate placement and scores the *result* rather than counting SWAPs.

### 24. Scheduling & timing

**What.** ASAP/ALAP start times in nanoseconds, circuit duration, and real idle windows.
**Why.** Everything else counts gates; hardware cares how long qubits are alive. A CX is ~300 ns where an `rz` is a frame change costing nothing, so a circuit with fewer gates can take longer than one with more.

```python
from qorch.transpiler.scheduling import circuit_duration_ns, idle_report, schedule_asap
from qorch.mitigation.dd import insert_dd_timed

circuit_duration_ns(circuit, calibration)     # wall clock, not gate count
idle_report(circuit)                          # {qubit: nanoseconds spent idle}
insert_dd_timed(circuit, "xy4")               # DD only where the pulses actually fit
```

Slot-based DD packs pulses into a gap of three `rz` — which takes zero time. Timed DD measures the window and declines.

### 25. Async execution (submit/poll/cancel)

**What.** A job lifecycle around a serializable `JobHandle`, plus an authentication hook.
**Why.** `run` blocks, which is right for a simulator and wrong for queued hardware where a job waits behind other people's for hours.

```python
import json
from qorch.backends.async_backend import LocalAsyncSimulator, JobHandle

backend = LocalAsyncSimulator(seed=1)
handle = backend.submit(circuit, shots=1000)      # returns immediately
backend.status(handle)                             # queued / running / done / cancelled / error
result = backend.wait(handle, timeout=30)          # or backend.cancel(handle)

JobHandle.from_dict(json.loads(json.dumps(handle.to_dict())))   # survives a restart
```

Credentials are never stored on the instance or taken as a constructor argument — they are read from the environment at point of use, so a token cannot reach a repr, a pickle, or a traceback.

### 26. JobResult v2

**What.** Per-shot memory, quasi-probabilities, expectation values, and the final layout alongside counts.
**Why.** A histogram discards shot order, cannot represent a mitigated distribution (whose values may be negative), and cannot say which physical qubit produced each bit.

```python
result = LocalSimulator(seed=1, memory=True).run(circuit, shots=1000)
result.memory[:3]                  # ('00', '11', '00') — order preserved
result.probabilities               # follows mitigation when it has been applied
result.expectation_z(0)            # ⟨Z⟩ on one qubit
result.parity_expectation((0, 1))  # ⟨ZZ⟩ — 1.0 for a Bell pair
```

### 27. Fan-out scheduler + job store

**What.** Submit a batch across async backends, collect as results land, and persist handles to SQLite.
**Why.** With queued hardware the wait dominates, so everything should be queued at once. And a handle held only in memory dies with the process, taking hours of queue position with it.

```python
from qorch.async_scheduler import AsyncScheduler, JobStore, cost_based_policy

scheduler = AsyncScheduler(backends=[qpu_a, qpu_b], policy=cost_based_policy,
                           store=JobStore("jobs.db"))
results = scheduler.run_batch([(circuit, 1000, "experiment-1")])
JobStore("jobs.db").unfinished()      # what a restarted process should reclaim
```

`cost_based_policy` ranks devices by predicted fidelity from their own calibration, not by whether the circuit merely fits. `sqlite3` is standard library, so the dependency-free core is intact.

### 28. Certification suite

**What.** A reproducible, vendor-neutral device evaluation with explicit thresholds and provenance.
**Why.** The benchmarks existed; what was missing is a report you can act on, compare, and re-run — and one that distinguishes *passed* from *could not be measured*.

```python
from qorch.certification import certify_backend, compare_reports, Thresholds

report = certify_backend(qpu, shots=2048, seed=7)
print(report.format())        # per-check pass/fail, uncertainties, verdict, provenance
report.to_json()              # machine-readable, with seed and thresholds recorded

print(compare_reports(report_a, report_b).format())
```

A device publishing no calibration reports `NOT_APPLICABLE`, never `PASS` — it has not taken the check, and blurring those would let an unmeasured device look good.

### 29. Tool layer (MCP-ready)

**What.** `simulate` / `analyze` / `transpile` / `draw` / `certify` as JSON-in, JSON-out functions.
**Why.** The protocol binding is the easy half. What has to be right is a tool layer that validates at the boundary and never raises a traceback at a caller who cannot see the stack.

```python
from qorch.tools import call_tool, describe_tools

call_tool("simulate", {"circuit": {"num_qubits": 2,
                                   "gates": [{"name": "h", "qubits": [0]},
                                             {"name": "cx", "qubits": [0, 1]}]},
                       "shots": 1000, "seed": 1})
# {'ok': True, 'counts': {...}, 'probabilities': {...}}
```

Errors return `{"ok": False, "error": ...}` rather than raising, and shot counts are bounded — an unbounded shot count on a tool endpoint is a way to ask a shared process for an hour of work.

### 30. Grounded offline copilot

**What.** Template-first circuit assistance, plus verification for circuits produced elsewhere.
**Why.** Grounding means the guarantee comes from *executing the output*, not from trusting whatever produced it. No model, no network, no API key — it runs air-gapped like the rest of the library.

```python
from qorch.copilot import assist, accept_free_form

assist("build me a bell pair")     # → verified circuit + explanation
assist("write me a poem")          # → no circuit; says what it *can* do

accept_free_form(spec_from_an_llm)  # verified by execution, or not returned at all
```

It cannot invent a circuit that does not exist, and it is allowed to say it does not understand — a far better failure than confidently producing a plausible circuit that computes the wrong thing.

### 31. Photonic IR family

**What.** A **separate** mode-based IR: beam splitters, phase shifters, transfer matrices, and photon statistics from permanents.
**Why.** Linear optics does not fit the qubit IR. A qubit circuit is a unitary on 2^n amplitudes; linear optics is an n×n unitary on *modes*, with the physics in how indistinguishable photons populate them. Two photons in two modes is not a two-qubit state.

```python
from qorch.photonic import PhotonicCircuit, output_distribution, hong_ou_mandel_coincidence

hong_ou_mandel_coincidence()        # 0.0 exactly — identical photons always bunch
circuit = PhotonicCircuit(2).beam_splitter(0, 1)
output_distribution(circuit, (0, 1))    # {(0, 0): 0.5, (1, 1): 0.5}
```

Photon statistics come from **permanents**, which is what makes boson sampling hard — exponential by physics, not by neglect, so these helpers are for small systems and say so.

### 32. Neutral-atom arrays

**What.** Connectivity derived from atom positions and the Rydberg blockade radius, and rearrangeable.
**Why.** A superconducting coupling map is fixed at fabrication. Neutral atoms sit where the operator puts them, so the coupling map is an *output* of the arrangement rather than an input to compilation.

```python
from qorch.neutral_atom import line_array, grid_array, ring_array, gate_set_for

array = grid_array(2, 3, spacing_um=5.0, blockade_radius_um=8.0)
array.coupling_map()        # derived from geometry — the router needs no special case
array.is_connected()
transpile_with_layout(circuit, gate_set_for(array), coupling_map=array.coupling_map())
```

`rearranged()` moves atoms to suit a circuit, measurably reducing SWAPs — the thing a fixed lattice cannot do.

### 33. CLI

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

Two companion documents go deeper than this tree:

- **[docs/architecture.md](docs/architecture.md)** — how the layers fit together, the recorded architecture decisions (ADR-2/5/6) that the source cites by number, the pipeline's load-bearing ordering constraints, the bit-ordering convention, and what the tests are actually checking.
- **[docs/CODEMAP.md](docs/CODEMAP.md)** — every module, what it owns, what it deliberately does *not* do, and where to start for a given kind of change.

```
src/qorch/
  ir.py                  # immutable IR (gates, params, dynamic ops) + QASM-3 + JSON
  gates.py               # GateDef registry — one definition per gate, all else derived
  adp.py                 # algorithm templates: QFT, Grover, QAOA, VQE, QPE
  dynamic.py             # teleportation + repetition code (dynamic circuits)
  pulse.py               # waveforms + rotating-frame pulse simulator + calibration
  qec.py                 # repetition-d + Steane [[7,1,3]] codes + threshold
  surface_code.py        # toric code geometry + MWPM decoder + 2D threshold
  resource_estimation.py # fault-tolerant resource estimates from T-count
  tomography.py          # 1Q + 2Q state tomography
  entanglement.py        # Bell fidelity, CHSH, entanglement witness
  benchmarking.py        # RB, QV (+ sweep), XEB
  certification.py       # vendor-neutral device evaluation + report comparison
  photonic.py            # SEPARATE IR: modes, beam splitters, permanents
  neutral_atom.py        # Rydberg-blockade geometry → coupling map, rearrangeable
  copilot.py             # template-first assistance + verification (offline, no model)
  tools.py               # simulate/analyze/transpile/draw/certify as JSON tools
  qmi.py                 # QMI binary format (validated decode)
  scheduler.py           # FIFO queue + best-fit batch scheduler
  async_scheduler.py     # fan-out over async backends + SQLite job store
  analysis.py            # circuit depth / gate count / fidelity
  visual.py              # ASCII circuit drawing
  cli.py                 # command-line interface (9 commands)
  backends/
    base.py              # Backend HAL + BackendProperties + DeviceCalibration + JobResult
    simulator.py         # dependency-free statevector (+ noise + dynamic execution)
    numpy_kernel.py      # optional numpy kernel, auto above a measured crossover
    gpu_kernel.py        # optional CuPy kernel — ⚠️ UNVERIFIED on hardware
    async_backend.py     # submit/poll/cancel lifecycle + authentication hook
    density_simulator.py # Kraus-operator density-matrix simulation (T1/T2)
    stabilizer.py        # CHP tableau simulator (polynomial-time Clifford)
    indian_backend.py    # Indian QPU adapter (IIT Jodhpur, TIFR, DRDO MIRAI)
    qiskit_backend.py    # IBM / Qiskit Aer adapter (optional dependency)
  transpiler/
    gateset.py           # Indian-native + Clifford+T gate-set definitions
    passes.py            # pass manager + per-pass transpile metrics
    layout.py            # initial placement: dense / noise-adaptive / cost-aware
    decompose.py         # recursive decomposition (incl. Clifford+T)
    synthesis.py         # meet-in-the-middle Rz → Clifford+T, with reported error
    routing.py           # greedy + SabreSWAP routing (layout-correct) + edge-direction fixing
    fusion.py            # Euler fusion + commutation-aware cancellation
    optimizer.py         # gate cancellation + rotation merging
    scheduling.py        # ASAP/ALAP timing, idle windows, circuit duration
    cost.py              # calibration cost model for ranking compilations
  mitigation/
    readout.py  zne.py  pec.py  dd.py  twirling.py  pipeline.py
tests/                   # 1032 unit tests (~95% coverage)
```

## Tests

```bash
python -m pytest                 # 1032 tests
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
