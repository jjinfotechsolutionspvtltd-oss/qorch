# Contributing to qorch

Thanks for your interest. qorch is a hardware-agnostic control plane for quantum
execution, and it has one unusual property that shapes everything below: **the core
has zero runtime dependencies**. That constraint is deliberate, and it is the first
thing to understand before you write code.

## The five rules

These hold for every change. They are not stylistic preferences — a PR that breaks
one of them will be asked to change, however good the idea is.

1. **Preserve functionality.** Refactors are behavior-preserving; tests prove it.
2. **Remove nothing.** Pure-Python kernels stay as fallbacks. Old APIs deprecate,
   they do not disappear.
3. **No vendor lock-in.** Qiskit, numpy, and CUDA-Q are optional extras behind the
   backend HAL — never imported by the core.
4. **Sovereign core.** `dependencies = []` in `pyproject.toml` is an enforced
   invariant. If your feature needs a third-party package, it belongs behind an
   optional extra with a pure-Python fallback.
5. **Cost your proposal.** Problem / why it matters / difficulty / benefit. An issue
   that states the tradeoff gets a decision faster than one that states a wish.

## Getting set up

```bash
git clone https://github.com/jjinfotechsolutionspvtltd-oss/qorch.git
cd qorch
python -m pip install -e ".[dev]"
python -m pytest
```

Python 3.11 or newer. That is the whole setup — no services, no containers, no
hardware. The simulators are pure Python, so the full suite runs in well under a
minute on a laptop.

## Before you open a PR

The CI runs exactly four things. Run them locally and there will be no surprises:

```bash
ruff check src/ tests/
python -m mypy src/
python -m pytest --cov=qorch --cov-fail-under=75
python benchmarks/benchmark_mitigation.py
```

Coverage sits around 95%; the gate is 75%, so there is room, but new code is
expected to arrive with tests.

## What good tests look like here

This is a project where a bug does not crash — it returns a plausible-looking
probability distribution that happens to be wrong. Tests that assert a function
"runs without error" are close to worthless. The suite is built on tests that pin
behavior against something independent:

- **Physics, not implementation.** A θ-area pulse must drive |0⟩ to P(1) = sin²(θ/2).
  A simulated evolution must satisfy UU† = I. These fail loudly when the model drifts.
- **Semantic equivalence across a transformation.** A transpiled circuit must produce
  the same output distribution as the original. Routing, decomposition, and
  optimization are all validated this way — compile it, run both, compare.
- **Known-answer circuits.** The repetition code with an injected error has exactly
  one correct syndrome and one correct decoded bit. Assert both.

If you are adding a compiler pass, the burden of proof is a test showing the pass
does not change what the circuit computes.

## Areas that welcome help

- **Compiler:** layout passes, commutation-based cancellation, calibration-aware
  routing, scheduling and timing.
- **Error correction:** decoders, threshold estimation, logical-qubit abstractions.
- **Backends:** new hardware adapters behind the existing HAL. The `Backend`
  protocol in `src/qorch/backends/base.py` is the whole contract.
- **Performance:** vectorized kernels — as an *optional* path, with the pure-Python
  implementation retained (see rule 2).

Issues labeled `good first issue` are scoped to be self-contained.

## Commits and pull requests

Conventional-commit prefixes: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`,
`perf`, `ci`.

Write the commit message for someone reading it in a year with no memory of the
conversation. Say what changed and *why the alternative was rejected* — that second
part is what makes history worth having.

For the PR description: what problem, what approach, and how you convinced yourself
it is correct. If you found a limitation you did not fix, say so explicitly rather
than leaving it for a reviewer to discover.

## Reporting bugs

A quantum bug report needs the circuit. Include a minimal `Circuit` construction, the
backend and shot count, what distribution you expected, and what you got. A seeded
`LocalSimulator(seed=...)` makes it reproducible.

## Security

Do not open a public issue for a vulnerability. See [SECURITY.md](SECURITY.md).

## License

Contributions are licensed under [Apache License 2.0](LICENSE), the same terms as the
project. By submitting a PR you confirm you have the right to license your
contribution under those terms.
