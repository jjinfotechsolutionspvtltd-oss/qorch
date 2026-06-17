# qorch — QS-004 Quantum Orchestration Layer

The classical **control plane** that orchestrates quantum hardware: IR ingestion → backend
HAL → error mitigation → scheduling. Hardware-agnostic from day one — a real QPU plugs in as
one new `Backend` adapter with zero core changes.

> This is the **M1 vertical slice**: it ships dependency-free (stdlib only) so it runs and
> tests anywhere, no QPU account required. See [`../charter.md`](../charter.md) and
> [`../architecture.md`](../architecture.md).

## Layout

```
src/qorch/
  ir.py                 # immutable circuit IR + OpenQASM-3 (subset) ingestion
  backends/base.py      # the HAL: Backend interface + BackendProperties + JobResult
  backends/simulator.py # dependency-free statevector backend (+ optional readout noise)
  mitigation/readout.py # readout-error calibration & correction (the M1 differentiator)
  mitigation/zne.py     # M3: zero-noise extrapolation via unitary circuit folding
  backends/qiskit_backend.py  # M2 adapter: Qiskit Aer + real IBM hardware, same interface
  scheduler.py          # FIFO queue + pluggable backend-selection policy
tests/                  # unit + end-to-end slice tests
benchmarks/             # raw-vs-mitigated report generator → BENCHMARK.md
```

## The payoff ([BENCHMARK.md](BENCHMARK.md))

The same circuit, raw vs. routed through QS-004's mitigation, on the noisy simulator:

| Technique | Error (raw → mitigated) | Reduction |
|---|---|---|
| Readout calibration | 0.0523 → 0.0029 | **95%** |
| Zero-noise extrapolation | 0.2448 → 0.1472 | **40%** |

Regenerate: `PYTHONPATH=src python benchmarks/benchmark_mitigation.py`

## Backends

| Backend | Status | Needs |
|---------|--------|-------|
| `LocalSimulator` | ✅ ships now | nothing (stdlib only) |
| `QiskitBackend.aer(...)` | ✅ code ready | `pip install qorch[qiskit]` |
| `QiskitBackend.ibm(name)` | ✅ code ready | `qorch[qiskit]` + `QISKIT_IBM_TOKEN` env var (secret) |

The same `Circuit` runs on all three unchanged — that's the hardware-abstraction proof. The
endianness translation (Qiskit is little-endian; qorch puts qubit 0 leftmost) is handled and
unit-tested in `reorder_counts_qiskit_to_qorch`. Live Aer tests skip automatically until the
SDK is installed; the IBM token is read from the environment and never committed.

## Run the tests

```bash
cd orchestrator
python -m pytest                 # stdlib-only; no install needed (pythonpath=src)
python -m pytest --cov=qorch     # with coverage (pip install pytest-cov)
```

## 30-second example

```python
from qorch import Circuit, LocalSimulator, ReadoutNoise
from qorch.mitigation.readout import ReadoutMitigator

# Build a Bell state and run it
bell = Circuit(num_qubits=2).h(0).cx(0, 1)
print(LocalSimulator(seed=1).run(bell, shots=1000).counts)   # ~ {'00': 500, '11': 500}

# Show the mitigation payoff on a noisy readout (asymmetric, as on real hardware)
noise = ReadoutNoise(p1_given0=0.05, p0_given1=0.15)
raw = LocalSimulator(seed=1, readout_noise=noise).run(Circuit(1).h(0), 20000)
mit = ReadoutMitigator.from_calibration_matrix(
    ["0", "1"], [[0.95, 0.15], [0.05, 0.85]]
).apply(raw.counts)
print("raw:", raw.counts, "→ mitigated:", {k: round(v) for k, v in mit.items()})
```

## What's next (M4)

M1–M3 are in: the slice, the real-hardware adapter, and both mitigation techniques with a
benchmark. Next:
- Run the live Aer + IBM parity tests on a machine with `qorch[qiskit]` installed.
- Wrap the Qiskit/tket transpiler explicitly for native-gate compilation + topology routing.
- Policy scheduler (cost/latency/availability routing) + the public SDK/API surface.
- Backend-aware mitigation tuning (calibration pulled from live backend properties).
