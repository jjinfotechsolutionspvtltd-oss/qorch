# qorch — Codemap

Every module, what it owns, and — where it matters — what it deliberately does
*not* do. Generated against the tree and kept in the same order as the package.

**52 modules · ~11,000 lines · 1,032 tests · ~95% coverage**

See [architecture.md](architecture.md) for how these fit together and why.

---

## Core IR

| module | lines | owns |
|---|---:|---|
| `ir.py` | 469 | `Circuit`, `Gate`, `Measure`, `Reset`, `Parameter`; QASM-3 subset; JSON |
| `gates.py` | 151 | `GateDef` registry — the single definition of every gate |

**`ir.py`** is the type everything else programs against. Immutable (ADR-5):
every builder returns a new circuit. `_appending` is the fast path that keeps
construction linear rather than quadratic — validating only the operation being
added, since the receiver was already valid.

**`gates.py`** is the single source of truth for gate metadata: matrix, arity,
params, `is_clifford`, `self_inverse`, `angle_inverse`, duration. Every other
table in the library is a *derived view* of it. Do not add a gate fact anywhere
else — that duplication has already caused one silent wrong-answer bug.

---

## Backends (the HAL)

| module | lines | owns |
|---|---:|---|
| `backends/base.py` | 164 | `Backend`, `BackendProperties`, `DeviceCalibration`, `JobResult` |
| `backends/simulator.py` | 456 | dependency-free statevector + noise + dynamic execution |
| `backends/numpy_kernel.py` | 126 | optional numpy kernel, auto above a **measured** 8-qubit crossover |
| `backends/gpu_kernel.py` | 168 | optional CuPy kernel — ⚠️ **unverified on hardware**, opt-in only |
| `backends/density_simulator.py` | 403 | Kraus density-matrix simulation (T1/T2, depolarizing) |
| `backends/stabilizer.py` | 218 | CHP tableau — polynomial-time Clifford, scales past statevector |
| `backends/async_backend.py` | 244 | `AsyncBackend` submit/status/result/cancel + `AuthenticatedBackend` |
| `backends/indian_backend.py` | 318 | IIT Jodhpur / TIFR / DRDO MIRAI presets |
| `backends/qiskit_backend.py` | 143 | IBM / Qiskit Aer adapter (optional dependency) |

`Backend` is three methods — `properties`, `run`, `validate` — plus optional
`calibration()` / `coupling_map()` that default to `None`. Everything above this
layer programs against the interface only.

`qiskit_backend.py` is the **only** place a foreign bit-ordering convention
enters, and it converts at that single boundary.

---

## Transpiler

| module | lines | owns |
|---|---:|---|
| `transpiler/__init__.py` | 308 | `transpile`, `transpile_with_layout`, `build_pass_manager` |
| `transpiler/passes.py` | 170 | pass manager, `PassState`, per-pass metrics |
| `transpiler/gateset.py` | 62 | native gate-set definitions for each target |
| `transpiler/layout.py` | 227 | initial placement: trivial / dense / noise-adaptive / cost-aware |
| `transpiler/routing.py` | 585 | greedy + SabreSWAP routing, edge-direction fixing, final layout |
| `transpiler/decompose.py` | 485 | recursive lowering to any native gate set |
| `transpiler/synthesis.py` | 257 | meet-in-the-middle Rz → Clifford+T, with reported error |
| `transpiler/fusion.py` | 243 | Euler fusion + commutation-aware cancellation |
| `transpiler/optimizer.py` | 147 | adjacency cancellation + rotation merging |
| `transpiler/scheduling.py` | 172 | ASAP/ALAP timing, idle windows, circuit duration |
| `transpiler/cost.py` | 146 | calibration cost model for ranking compilations |

**Pipeline order is load-bearing** and enforced by tests:

```
decompose → [layout] → route → fuse → lower → optimize → [dd]
```

`routing.py` is the largest module and the one most worth reading before
changing: it holds both routers, the classical-hazard DAG that keeps feed-forward
after its measurement, and direction fixing.

---

## Mitigation

| module | lines | owns |
|---|---:|---|
| `mitigation/zne.py` | 100 | zero-noise extrapolation via unitary folding |
| `mitigation/pec.py` | 162 | probabilistic error cancellation (quasi-probabilities) |
| `mitigation/dd.py` | 170 | dynamical decoupling — slot-based and **duration-aware** |
| `mitigation/twirling.py` | 132 | Pauli twirling: coherent noise → stochastic |
| `mitigation/readout.py` | 62 | calibration-matrix inversion |
| `mitigation/pipeline.py` | 121 | compose mitigation steps in the correct order |

`insert_dd_timed` is the one to prefer: slot-based DD measures idleness in gate
*slots* and will pack pulses into a gap of three `rz`, which takes zero time.

---

## Algorithms, QEC, and analysis

| module | lines | owns |
|---|---:|---|
| `adp.py` | 517 | QFT, Grover, QAOA, VQE, QPE |
| `dynamic.py` | 129 | teleportation + repetition code (dynamic circuits) |
| `qec.py` | 257 | repetition-d, Steane [[7,1,3]], thresholds |
| `surface_code.py` | 298 | toric code geometry, MWPM decoder, 2D threshold |
| `resource_estimation.py` | 153 | T-count → magic states → physical qubits and runtime |
| `tomography.py` | 199 | 1Q + 2Q state tomography |
| `pulse.py` | 177 | waveforms, rotating-frame simulator, gate calibration |
| `analysis.py` | 75 | depth, gate counts, estimated fidelity |
| `visual.py` | 153 | ASCII circuit drawing |

---

## Measurement and evaluation

| module | lines | owns |
|---|---:|---|
| `benchmarking.py` | 447 | randomized benchmarking, **quantum volume**, QV sweep, XEB |
| `entanglement.py` | 174 | Bell fidelity, CHSH S-value, entanglement witness |
| `certification.py` | 376 | vendor-neutral device evaluation, reports, comparison |

`certification.py` distinguishes **passed** from **could not be measured**: a
device publishing no calibration reports `NOT_APPLICABLE`, never `PASS`.

The quantum-volume implementation in `benchmarking.py` computes heavy outputs
from each circuit's *ideal probability distribution* (above the median), not
from bitstring values. Getting that wrong is silent — it produced a plausible
number that no ideal device could pass.

---

## Separate IR families

| module | lines | owns |
|---|---:|---|
| `photonic.py` | 229 | modes, beam splitters, phase shifters, transfer matrices, permanents |
| `neutral_atom.py` | 179 | atom arrays, Rydberg blockade → coupling map, rearrangement |

`photonic.py` does **not** use `Circuit` and should not be made to: linear optics
transforms n modes, not 2^n amplitudes.

`neutral_atom.py` deliberately does the opposite — it produces an ordinary
`CouplingMap` so every existing pass works against it unchanged.

---

## Orchestration and entry points

| module | lines | owns |
|---|---:|---|
| `scheduler.py` | 99 | FIFO queue + best-fit batch routing |
| `async_scheduler.py` | 227 | fan-out over async backends, SQLite job store, cost-based policy |
| `tools.py` | 231 | simulate/analyze/transpile/draw/certify as JSON tools |
| `copilot.py` | 401 | template-first assistance + verification (offline, no model) |
| `cli.py` | 428 | nine commands |
| `qmi.py` | 157 | QMI binary format with validated decode |

`tools.py` and `copilot.py` are the boundaries where untrusted input arrives.
Both validate rather than trust: tools return `{"ok": false, "error": ...}`
instead of raising, and the copilot verifies a circuit **by executing it** before
returning it.

`qmi.py` parses external bytes and is the primary trust boundary — see
[SECURITY.md](../SECURITY.md).

---

## Where to start

| if you want to… | read |
|---|---|
| add a gate | `gates.py`, then `transpiler/decompose.py` |
| add a device | `backends/base.py`, then `backends/indian_backend.py` as a worked example |
| add a compiler pass | `transpiler/passes.py`, then insert into `build_pass_manager` |
| understand a wrong result | `docs/architecture.md` §5 (bit ordering) and §8 (what the tests check) |
| add an architecture | `neutral_atom.py` if it is qubit-like, `photonic.py` if it is not |
