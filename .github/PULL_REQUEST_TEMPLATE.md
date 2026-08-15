<!--
Thanks for contributing. Keep this short — the goal is to give a reviewer what they
need to judge correctness, not to fill in boxes.
-->

## What and why

<!-- The problem, and the approach you took. If you rejected an alternative, say which and why. -->

## How you know it's correct

<!--
This project's failure mode is a plausible-looking wrong answer, so this section
carries the weight. What pins the behavior — a physics identity, a semantic
equivalence across the transformation, a known-answer circuit?
-->

## Checks

- [ ] `ruff check src/ tests/`
- [ ] `python -m mypy src/`
- [ ] `python -m pytest`
- [ ] New behavior has tests; changed behavior has updated tests

## The five rules

- [ ] Preserves existing functionality (no silent behavior changes)
- [ ] Removes nothing (fallbacks kept, APIs deprecated rather than deleted)
- [ ] No new runtime dependency in the core — `dependencies = []` still holds
- [ ] Any optional dependency sits behind an extra, with a pure-Python fallback

## Known limitations

<!--
Anything you did not fix, edge cases you're unsure about, follow-up work. Stating it
here is far better than a reviewer finding it — and it will not be held against the PR.
-->
