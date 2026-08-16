# qorch — Architecture

This document exists because the source referenced it. `qorch/__init__.py` and
`qorch/ir.py` both cited `architecture.md` — including a specific section — and
the file was never written. What follows is the architecture the code actually
implements, not a plan for one.

---

## 1. The shape of the system

qorch is a **control plane**, not a simulator that grew features. Everything is
organized around one interface — `Backend` — with layers that program against it
and never against a particular device.

```
                    ┌──────────────────────────────────────────┐
   user / LLM  ───▶ │  tools.py · copilot.py · cli.py          │  entry points
                    └───────────────────┬──────────────────────┘
                                        │
      ┌─────────────────────────────────┼─────────────────────────────────┐
      │                                 │                                 │
┌─────▼─────┐                  ┌────────▼────────┐               ┌────────▼────────┐
│    ir.py  │                  │   transpiler/   │               │  mitigation/    │
│  Circuit  │ ───────────────▶ │  pass pipeline  │ ───────────▶  │  ZNE/PEC/DD/... │
│  (immutable)                 └────────┬────────┘               └────────┬────────┘
└─────┬─────┘                           │                                 │
      │                                 │                                 │
      │                        ┌────────▼─────────────────────────────────▼──────┐
      └───────────────────────▶│              backends/base.Backend              │
                               │   properties() · run() · validate()             │
                               │   optional: calibration() · coupling_map()      │
                               └────────┬────────────────────────────────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬────────────────┐
   LocalSimulator  DensitySim     StabilizerSim    IndianQPU      QiskitBackend
   (+ numpy/GPU)   (T1/T2 Kraus)  (CHP tableau)   (presets)      (optional dep)

   separate IR families:  photonic.py (modes, not qubits) · neutral_atom.py (geometry)
```

The arrows only ever point *toward* `Backend`. Nothing below it knows what is
above; nothing above it knows which device is below.

---

## 2. Architecture decisions

These are cited by number throughout the source. They are recorded here because
code that says "see ADR-6" and has nowhere to point is worse than code that says
nothing.

### ADR-2 — One backend interface

Every execution target implements `Backend`: `properties()`, `run()`,
`validate()`. Optional hooks (`calibration()`, `coupling_map()`) default to
`None` so a three-method adapter remains a complete implementation.

*Consequence:* the scheduler, mitigation, benchmarking, and certification layers
were written once and work against a simulator, an Indian QPU preset, and a
Qiskit device with no branching. Vendor-neutral comparison in
`certification.py` is only meaningful because of this.

### ADR-5 — The IR is immutable

Every `Circuit` builder returns a **new** circuit. Nothing mutates in place.

*Consequence:* transpiler passes compose safely and can be reordered, retried,
or run speculatively — the layout pass evaluates candidate placements by
compiling each one, which is only cheap and safe because compiling cannot
disturb the input.

*Cost, and how it was paid:* naive immutability made construction O(n²), because
every append revalidated the whole gate list. `Circuit._appending` validates only
the new operation, since the receiver was already valid and operations cannot
change. Construction is linear; a 2000-gate circuit went 355 ms → 8.5 ms.

### ADR-6 — The core has no third-party dependencies

`dependencies = []` in `pyproject.toml` is an enforced invariant, not an
aspiration. `import qorch` pulls in nothing outside the standard library.

*Consequence:* the library runs air-gapped, which is the point of a sovereign
stack. Everything optional sits behind an extra with a working fallback: numpy
accelerates the statevector but the pure-Python kernel remains and is what runs
without it; CuPy likewise; Qiskit is an adapter, never a requirement.

*Enforcement:* the fallback paths are tested standing alone rather than assumed
to work, because a fallback that is never exercised is a fallback that has
already rotted.

---

## 3. The compile pipeline

`transpile()` is an ordered list of named passes (`transpiler/passes.py`), not a
function with stages inlined. The ordering constraints are real, and **each one
is a bug that was fixed by putting a pass where it is**:

```
decompose → [layout] → route → fuse → lower → optimize → [dd]
                                       │
                        lower = decompose → fix directions → decompose
```

| constraint | why it is not negotiable |
|---|---|
| `lower` after `route` | routing emits SWAPs the first decomposition never saw, and SWAP is native to almost no target |
| direction fixing *inside* lower | it is the `swap → cx·cx·cx` expansion that produces the reversed CX a one-way edge forbids |
| `fuse` before `lower` | fusion emits `rz`/`ry`, which are not native everywhere |
| `optimize` after `lower` | the optimizer only merges or drops gates, so it cannot undo lowering |
| `dd` after `optimize` | DD sequences are logically the identity, so an optimizer that saw them would cancel exactly the pulses requested |

`layout` composes with routing rather than replacing it: layout decides where
qubits *begin*, routing where they must *move*, and the final layout is the
composition `routed[layout[q]]`.

**What the pipeline guarantees.** Every gate in the output is in
`target.basis_gates`, and every two-qubit gate sits on a coupling-map edge *in
the direction the hardware implements it*.

---

## 4. Single sources of truth

The recurring failure in this codebase has been the same one three times: a fact
stated in two places, drifting apart, and producing a plausible wrong answer
rather than a crash. Each is now stated once.

| fact | lives in | previously also in |
|---|---|---|
| what a gate *is* — matrix, arity, Clifford, self-inverse, duration | `gates.py` | simulator table, IR sets, optimizer sets, Indian backend |
| where a logical qubit ended up | `TranspileResult.final_layout` | nowhere — it was computed and discarded |
| what a device costs | `DeviceCalibration` | scattered scalar hints |

The `gates.py` consolidation fixed a live bug on contact: the optimizer listed
`sx` as self-inverse while the IR did not, and since `SX·SX = X` it cancelled
`sx sx` pairs — turning a circuit that outputs 1 into one that outputs 0.

---

## 5. Bit ordering

**Qubit 0 is the leftmost character of a result bitstring, and the most
significant bit of a state index.**

This is the convention `ir.py` refers to. It is stated once and obeyed
everywhere: the statevector kernels index qubit `q` with stride
`1 << (n - 1 - q)`, the numpy and GPU kernels reshape to `(2,)*n` so that qubit
`q` is axis `q`, and measurement assembles bitstrings in `readout_qubits` order.

Endianness disagreements are silent — they produce a valid-looking distribution
that is permuted — so the one place a foreign convention enters, the Qiskit
adapter, converts at that single boundary via
`reorder_counts_qiskit_to_qorch`.

---

## 6. Separate IR families

Not every architecture is a qubit machine, and forcing one into `Circuit` would
be a category error rather than an inconvenience.

**Photonics** (`photonic.py`) has its own IR. A qubit circuit is a unitary on
2^n amplitudes; linear optics is an **n×n unitary on modes**, with the physics in
how indistinguishable photons populate them. Two photons in two modes is not a
two-qubit state. Photon statistics come from permanents, which is what makes
boson sampling hard — exponential by physics, not by implementation.

**Neutral atoms** (`neutral_atom.py`) need no new IR — atoms are two-level
systems and lasers drive ordinary rotations. What differs is that connectivity
is *derived from geometry* and can be changed by moving atoms, so the coupling
map is an output of the arrangement rather than an input to compilation. That
makes it a `CouplingMap` producer, and every existing pass works unchanged.

---

## 7. Execution model

Synchronous `run()` is the common denominator and always works. `AsyncBackend`
adds the lifecycle real hardware needs — `submit` / `status` / `result` /
`cancel` — around a `JobHandle` that is **plain serializable data with no live
connection inside it**, so a job can be reclaimed by a different process after a
restart. That is most of the point of asynchronous submission and would be
impossible if the handle held a socket.

`AsyncBackend` subclasses `Backend` and implements `run()` as submit-then-wait,
so everything that already accepts a `Backend` accepts an async one unchanged.

Credentials are never stored on an instance or accepted as a constructor
argument: they are read from the environment at point of use, so a token cannot
reach a repr, a pickle, a log line, or a traceback.

---

## 8. What the tests are for

This is a library whose bugs do not crash. A wrong transpiler pass returns a
plausible probability distribution; a wrong benchmark returns a confident
number. Six such bugs were found and fixed, none of which raised an exception.

The suite is therefore built on checks with an *independent* reference:

- **physics** — a θ-area pulse must drive |0⟩ to P(1) = sin²(θ/2); UU† = I; two
  identical photons at a 50:50 splitter never exit separately
- **semantic equivalence across a transformation** — compile the circuit, run
  both, compare distributions
- **known-answer circuits** — the repetition code with an injected error has
  exactly one correct syndrome
- **cross-implementation agreement** — the numpy and GPU kernels are compared
  against the pure-Python one on random circuits

Tests that assert a function "runs without error" are close to worthless here
and are avoided.

---

## 9. Related documents

- [CODEMAP.md](CODEMAP.md) — every module and what it is responsible for
- [../README.md](../README.md) — features, usage, and platform setup
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — the five rules every change holds to
