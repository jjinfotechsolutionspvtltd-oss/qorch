# qorch — Indian Quantum Orchestration Layer

A sovereign, minimal, correct quantum software stack designed for India's emerging quantum hardware ecosystem. **Hardware-agnostic from day one** — any Indian QPU (from DRDO, ISRO, IITs, C-DAC) plugs in as one `Backend` adapter with zero core changes.

## Why qorch?

As India invests in indigenous quantum processors (superconducting at TIFR/DRDO, ion traps at IIT Jodhpur, photonic at IISc), a vendor-neutral software stack is essential. qorch provides:

- **No vendor lock-in** — stdlib-only core, zero dependency on IBM Qiskit or Google Cirq
- **Sovereign architecture** — clean HAL designed for Indian hardware adapters
- **Active research** — error mitigation (readout calibration, ZNE, dynamical decoupling) as a differentiator

## What's new (June 2026)

- **Transpiler** — decomposes circuits to Indian-native gate sets (IIT Jodhpur ion-trap: rx/ry/ms; TIFR superconducting: cx/sx/rz/x; DRDO MIRAI: cx/rx/rz/x) + qubit routing for limited-connectivity topologies
- **Indian QPU backend** — simulates published Indian quantum hardware characteristics with realistic noise (gate infidelity, readout errors, limited connectivity)
- **CLI** — run circuits from QASM files, list backends, apply mitigation, transpile for Indian QPUs: `python -m qorch run circuit.qasm --backend tifr-superconducting`
- **Dynamical Decoupling** — XY-4, XY-8, CPMG, and Hahn echo sequences to suppress decoherence
- **CI pipeline** — GitHub Actions with lint + test + coverage gates

## Layout

```
src/qorch/
  ir.py                 # immutable circuit IR + OpenQASM-3 (extended subset) ingestion
  cli.py                # command-line interface
  backends/
    base.py             # the HAL: Backend interface + BackendProperties + JobResult
    simulator.py        # dependency-free statevector backend (+ readout/gate noise)
    qiskit_backend.py   # Qiskit Aer + real IBM hardware adapter
    indian_backend.py   # Indian QPU adapter (IIT Jodhpur, TIFR, DRDO MIRAI)
  transpiler/
    gateset.py          # Indian-native gate set definitions
    decompose.py        # gate decomposition to native sets
    routing.py          # qubit routing for limited connectivity
  mitigation/
    readout.py          # readout-error calibration & correction
    zne.py              # zero-noise extrapolation via unitary folding
    dd.py               # dynamical decoupling (XY-4, XY-8, CPMG, Hahn)
  scheduler.py          # FIFO queue + pluggable backend-selection policy
tests/                  # unit + end-to-end slice tests
.github/workflows/      # CI pipeline
```

## Quick start

```bash
pip install -e .
python -m qorch list
python -m qorch run --gates "h0,cx01" --backend tifr-superconducting
python -m qorch transpile --gates "h0,cx01" --target tifr-superconducting
```

### Python API

```python
from qorch import Circuit, IndianQPU

# Run on simulated Indian superconducting processor
qpu = IndianQPU.from_preset("tifr-superconducting", seed=42)
bell = Circuit(num_qubits=2).h(0).cx(0, 1)
result = qpu.run(bell, shots=2000)
print(result.counts)  # {'00': ..., '11': ...} with noise
```

## Indian QPU backends

| Backend | Qubits | Topology | Native Gates | Modeled On |
|---|---|---|---|---|
| `iit-jodhpur-ion-trap` | 6 | All-to-all | rx, ry, ms | IIT Jodhpur trapped-ion publications |
| `tifr-superconducting` | 5 | Linear | cx, sx, rz, x | TIFR Mumbai superconducting roadmap |
| `drdo-mirai` | 6 | Grid (2×3) | cx, rx, rz, x | DRDO MIRAI Lab |

## Mitigation benchmark

| Technique | Error reduction |
|---|---|
| Readout calibration | **95%** |
| Zero-noise extrapolation | **40%** |
| Dynamical decoupling | Insert XY-4/XY-8/CPMG/Hahn sequences |

## Tests

```bash
python -m pytest           # 47 tests, stdlib-only
python -m pytest --cov=qorch  # with coverage
```

## Strategic context

qorch is a research project focused on **Indian quantum readiness**:
- Clean HAL that any future Indian QPU can implement against
- Transpiler targeting Indian-native gate sets
- Error mitigation as a research differentiator
- No foreign vendor lock-in
