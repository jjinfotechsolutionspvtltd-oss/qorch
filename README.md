# qorch — Indian Quantum Orchestration Layer

A sovereign, minimal, correct quantum software stack designed for India's emerging quantum hardware ecosystem. **Hardware-agnostic from day one** — any Indian QPU (from DRDO, ISRO, IITs, C-DAC) plugs in as one `Backend` adapter with zero core changes.

## Why qorch?

As India invests in indigenous quantum processors (superconducting at TIFR/DRDO, ion traps at IIT Jodhpur, photonic at IISc), a vendor-neutral software stack is essential. qorch provides:

- **No vendor lock-in** — stdlib-only core, zero dependency on IBM Qiskit or Google Cirq
- **Sovereign architecture** — clean HAL designed for Indian hardware adapters
- **Active research** — error mitigation, tomography, benchmarking, Clifford+T decomposition

## Install

```bash
pip install -e .
```

## Features

### 1. Circuit IR + QASM3 ingestion
```python
from qorch import Circuit, from_qasm3, to_qasm3

c = Circuit(2).h(0).cx(0, 1).measure(0, 1)
c = from_qasm3('OPENQASM 3.0; qubit[2] q; h q[0]; cx q[0], q[1];')
qasm = to_qasm3(c)
```

### 2. Backends (HAL)

| Backend | Qubits | Topology | Native Gates | Modeled On |
|---|---|---|---|---|
| **LocalSimulator** | any | All-to-all | all | Dependency-free statevector |
| **DensitySimulator** | any | All-to-all | all + noise | Kraus-based noise simulation |
| `iit-jodhpur-ion-trap` | 6 | All-to-all | rx, ry, ms | IIT Jodhpur trapped-ion |
| `tifr-superconducting` | 5 | Linear | cx, sx, rz, x | TIFR Mumbai superconducting |
| `drdo-mirai` | 6 | Grid 2×3 | cx, rx, rz, x | DRDO MIRAI Lab |
| QiskitBackend | any | varies | varies | IBM/Qiskit Aer adapter |

```python
from qorch import LocalSimulator, IndianQPU

sim = LocalSimulator(seed=42)
qpu = IndianQPU.from_preset("tifr-superconducting", seed=42)
result = qpu.run(Circuit(2).h(0).cx(0, 1), shots=2000)
```

### 3. Transpiler — Decompose + Route

Decompose any circuit to an Indian-native gate set, then route for limited-connectivity topologies.

```python
from qorch.transpiler import transpile, CLIFFORD_T
from qorch import Circuit

c = Circuit(3).h(0).cx(0, 2).rx(1, 0.5)

# Decompose to TIFR native gates + route for linear topology
result = transpile(c, target=TIFR_SUPERCONDUCTING, coupling_map=..., use_lookahead=True)
```

Available targets: `IIT_JODHPUR_ION_TRAP`, `TIFR_SUPERCONDUCTING`, `DRDO_MIRAI`, `CLIFFORD_T`.

### 4. SabreSWAP Lookahead Routing

Advanced routing that scores candidate SWAPs using a DAG front-layer + extended-layer heuristic, producing fewer SWAPs than greedy routing.

```python
from qorch.transpiler.routing import route_lookahead, CouplingMap, QubitQuality

cmap = CouplingMap(edges=((0, 1), (1, 2), (2, 3)))
# Noise-aware: prefers paths through high-fidelity qubits
quality = {0: QubitQuality(0.99), 1: QubitQuality(0.95), 2: QubitQuality(0.99), 3: QubitQuality(0.90)}
routed = route_lookahead(c, cmap, qubit_quality=quality, lookahead=20, decay=0.5)
```

### 5. Clifford+T Decomposition

Decompose arbitrary circuits into the fault-tolerant Clifford+T gate set ({h, cx, t}) with T-count and T-depth reporting.

```python
from qorch.transpiler import decompose_to_clifford_t

c = Circuit(2).h(0).cx(0, 1).rz(0, 0.3)
result, t_count, t_depth = decompose_to_clifford_t(c)
print(f"T-count: {t_count}, T-depth: {t_depth}")
```

T-counts for common gates: T=1, S=2, Z=4, X=5, SX=4, Y=11.

### 6. State Tomography

Reconstruct 1Q and 2Q density matrices from Pauli-basis measurements.

```python
from qorch.tomography import state_tomography_1q, state_tomography_2q, purity, trace_rho

sim = LocalSimulator(seed=42)
c = Circuit(1).h(0)
rho = state_tomography_1q(sim, c, shots=4096)
print(f"Purity: {purity(rho):.4f}")  # 1.0 for pure state
```

### 7. Error Mitigation

```python
from qorch.mitigation import ReadoutMitigator, zne_expectation
from qorch.backends.simulator import GateNoise, ReadoutNoise, LocalSimulator

# Readout-error mitigation
mitigator = ReadoutMitigator(sim, cal_shots=8192)
corrected = mitigator.correct(result)

# Zero-noise extrapolation
zne_result = zne_expectation(sim, circuit, observable, shots=8192, scales=[1, 3, 5])

# Dynamical decoupling
from qorch.mitigation.dd import insert_dd
c_dd = insert_dd(c, sequence="xy4", qubits=(0, 1))
```

### 8. Noise-model Builders

Construct noise models from device-level specifications.

```python
from qorch.backends.simulator import GateNoise, ReadoutNoise
from qorch.backends.density_simulator import NoiseChannel

# From gate fidelity
chan = NoiseChannel.from_gate_fidelity(gate_fidelity=0.99, t1=50e-6, t2=80e-6, t_gate=100e-9)

# Nonsymmetric readout noise
readout = ReadoutNoise.from_readout_fidelity(fidelity=0.95, symmetric=False)

# Per-gate noise
gates = GateNoise.from_gate_fidelity(
    gate_fidelity=0.995, t1=50e-6, t2=80e-6, t_gate=100e-9,
    gate_names=("h", "cx", "rx"),
)
```

### 9. Quantum Volume Sweep

Parameterized QV benchmarking across qubit widths to find the maximum QV a device can achieve.

```python
from qorch.benchmarking import qv_sweep

sim = LocalSimulator(seed=42)
result = qv_sweep(sim, start_width=2, end_width=5, trials=20, shots=4096)
print(f"QV = 2^{result.max_passing_width} = {result.quantum_volume}")
```

### 10. QMI Binary Format

Compact binary encoding of quantum circuits for QPU microcode and firmware transfer.

```python
from qorch.qmi import to_qmi, from_qmi, QMIEncoder

data = to_qmi(circuit)        # 4-10× smaller than JSON/QASM
c2 = from_qmi(data)           # roundtrip
print(QMIEncoder.hexdump(data))  # human-readable hex dump
```

### 11. Batch Scheduler

Route multiple circuits to best-fit backends with minimum qubit waste.

```python
from qorch.scheduler import Scheduler, BatchJob

sched = Scheduler(backends=[sim1, qpu2])
jobs = [BatchJob(circuit=c1, shots=1024, label="bell"),
        BatchJob(circuit=c2, shots=2048, label="ghz")]
results = sched.run_batch(jobs)
```

### 12. CLI

```text
Usage: python -m qorch <command> [options]

Commands:
  run        — Execute a circuit on a backend
  batch      — Compare backends on the same circuit
  sched      — Batch-schedule multiple circuits
  mitigate   — Run with error mitigation
  transpile  — Decompose for an Indian QPU
  report     — Circuit depth, gate count, fidelity analysis
  certify    — Run QPU certification suite (Bell, CHSH, RB, QV)
  qv-sweep   — Quantum Volume sweep across widths
  list       — List backends and techniques
```

Examples:
```bash
# Run a circuit
python -m qorch run --gates "h0,cx01" --backend tifr-superconducting

# Run certification suite
python -m qorch certify --backend local-simulator --shots 4096

# QV sweep
python -m qorch qv-sweep --backend local-simulator --start 2 --end 5

# Batch schedule multiple circuits
python -m qorch sched --spec "bell:h0,cx01,measure01" --spec "ghz:h0,cx01,cx01,measure01"

# Transpile for an Indian QPU
python -m qorch transpile --gates "h0,cx01" --target tifr-superconducting
```

## Architecture

```
src/qorch/
  ir.py                 # immutable circuit IR + OpenQASM-3 + JSON serialization
  cli.py                # command-line interface (12 commands)
  tomography.py         # 1Q + 2Q state tomography
  qmi.py                # QMI binary format encoder/decoder
  scheduler.py          # FIFO queue + batch scheduler with best-fit routing
  backends/
    base.py             # Backend interface + BackendProperties + JobResult
    simulator.py        # dependency-free statevector (+ gate/readout noise)
    density_simulator.py# Kraus-operator density-matrix simulation
    indian_backend.py   # Indian QPU adapter (IIT Jodhpur, TIFR, DRDO MIRAI)
  transpiler/
    gateset.py          # Indian-native + Clifford+T gate set definitions
    decompose.py        # gate decomposition (recursive, supports Clifford+T)
    routing.py          # qubit routing (greedy + SabreSWAP lookahead)
    optimizer.py        # circuit optimization passes
  mitigation/
    readout.py          # readout-error calibration & correction
    zne.py              # zero-noise extrapolation via unitary folding
    pec.py              # probabilistic error cancellation
    dd.py               # dynamical decoupling (XY-4, XY-8, CPMG, Hahn)
    twirling.py         # Pauli twirling for noise tailoring
  benchmarking.py       # RB, QV, XEB benchmarks + QV sweep
  analysis.py           # circuit analysis (depth, gate count, fidelity)
tests/                  # 289 unit tests (92.96% coverage)
```

## Tests

```bash
python -m pytest                 # 289 tests
python -m pytest --cov=qorch     # with coverage
python -m pytest --cov=qorch --cov-report=term-missing  # uncovered lines
```

## Indian QPU backends

| Backend | Qubits | Topology | Native Gates | Modeled On |
|---|---|---|---|---|
| `iit-jodhpur-ion-trap` | 6 | All-to-all | rx, ry, ms | IIT Jodhpur trapped-ion |
| `tifr-superconducting` | 5 | Linear | cx, sx, rz, x | TIFR Mumbai superconducting |
| `drdo-mirai` | 6 | Grid 2×3 | cx, rx, rz, x | DRDO MIRAI Lab |

## Mitigation benchmark

| Technique | Error reduction |
|---|---|
| Readout calibration | 95% |
| ZNE extrapolation | 40% |
| Dynamical decoupling | Insert XY-4/XY-8/CPMG/Hahn sequences |

## Strategic context

qorch is a research project focused on **Indian quantum readiness**:
- Clean HAL that any future Indian QPU can implement against
- Transpiler targeting Indian-native + Clifford+T gate sets
- State tomography, QV sweep, and certification for hardware validation
- QMI binary format for low-latency QPU communication
- Error mitigation as a research differentiator
- No foreign vendor lock-in
