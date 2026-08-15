"""Performance properties that are easy to lose and expensive to notice.

Both invariants here were regressions waiting to happen rather than hypotheticals:
circuit construction really was quadratic, and it dominated QEC threshold
estimation above the stabilizer simulator itself.

These assert *structure*, never wall-clock time. A timing assertion on CI is a
flaky test with extra steps; counting the work actually performed is
deterministic and says the same thing.
"""

from __future__ import annotations

import itertools

import pytest

from qorch import Circuit
from qorch.ir import Gate
from qorch.surface_code import ToricCode


# ── circuit construction is linear, not quadratic ────────────────────────


def _count_validations(build, monkeypatch) -> int:
    calls = 0
    original = Circuit._validate_op

    def counting(self, op):
        nonlocal calls
        calls += 1
        return original(self, op)

    monkeypatch.setattr(Circuit, "_validate_op", counting)
    build()
    return calls


def test_building_a_circuit_validates_each_op_once(monkeypatch) -> None:
    """n builder calls must cost n validations, not n²/2.

    Re-validating the whole gate list on every append is the natural thing to
    write and quietly makes every circuit O(n²) to build.
    """
    n = 200

    def build():
        c = Circuit(8)
        for i in range(n):
            c = c.h(i % 8)
        return c

    assert _count_validations(build, monkeypatch) == n


def test_dynamic_builders_are_also_linear(monkeypatch) -> None:
    """The dynamic-circuit builders share the fast path."""
    n = 60

    def build():
        c = Circuit(4, num_clbits=2)
        for i in range(n):
            c = c.measure_into(i % 4, i % 2).reset(i % 4).x_if(i % 4, i % 2, 1)
        return c

    assert _count_validations(build, monkeypatch) == 3 * n


def test_direct_construction_still_validates_everything() -> None:
    """The fast path is for *growing* a valid circuit; the front door is unchanged."""
    with pytest.raises(ValueError, match="out of range"):
        Circuit(2, gates=(Gate("h", (7,)),))


@pytest.mark.parametrize("build,message", [
    (lambda: Circuit(2).h(5), "out of range"),
    (lambda: Circuit(2).cx(0, 9), "out of range"),
    (lambda: Circuit(2, num_clbits=1).measure_into(0, 7), "clbit"),
    (lambda: Circuit(2, num_clbits=1).x_if(0, 5, 1), "clbit"),
    (lambda: Circuit(2).reset(3), "out of range"),
    (lambda: Circuit(2).gate_if("nope", (0,), ()), "unsupported gate"),
])
def test_fast_path_still_rejects_invalid_operations(build, message: str) -> None:
    """Skipping redundant work must not skip the check that matters."""
    with pytest.raises(ValueError, match=message):
        build()


def test_fast_path_preserves_value_semantics() -> None:
    """Bypassing __init__ must not break equality, hashing, or fields."""
    a = Circuit(2).h(0).cx(0, 1)
    b = Circuit(2).h(0).cx(0, 1)
    assert a == b
    assert hash(a) == hash(b)
    assert (a.num_qubits, a.measured, a.num_clbits) == (2, (), 0)
    assert Circuit.from_json(a.to_json()) == a


# ── the MWPM decoder still returns a *minimum* weight matching ───────────


def _brute_force_min_cost(code: ToricCode, defects: list[tuple[int, int]]) -> int:
    """Minimum matching cost by exhaustive enumeration — the ground truth."""
    best = None
    idx = list(range(len(defects)))

    def recurse(remaining: list[int], cost: int) -> None:
        nonlocal best
        if not remaining:
            best = cost if best is None else min(best, cost)
            return
        i = remaining[0]
        for j in remaining[1:]:
            rest = [x for x in remaining[1:] if x != j]
            recurse(rest, cost + code.plaq_distance(defects[i], defects[j]))

    recurse(idx, 0)
    return best or 0


def test_dp_matching_is_genuinely_minimum_weight() -> None:
    """Checked against exhaustive enumeration, not against itself.

    The DP was optimized for speed; the property that must survive is that it
    still finds a *minimum* weight matching. An approximate decoder would still
    produce valid-looking corrections and silently shift the threshold.
    """
    import random

    code = ToricCode(5)
    rng = random.Random(4)
    checked = 0
    for _ in range(300):
        err = {e for e in range(code.num_edges) if rng.random() < 0.12}
        defects = code.syndrome(err)
        if not defects or len(defects) > 8:      # keep brute force tractable
            continue
        pairs = code._dp_match(defects)
        cost = sum(code.plaq_distance(defects[i], defects[j]) for i, j in pairs)
        assert cost == _brute_force_min_cost(code, defects)
        checked += 1
    assert checked >= 20, "sample produced too few cases to be meaningful"


def test_dp_matching_pairs_every_defect_exactly_once() -> None:
    """A perfect matching: no defect left over, none matched twice."""
    import random

    code = ToricCode(5)
    rng = random.Random(9)
    for _ in range(200):
        err = {e for e in range(code.num_edges) if rng.random() < 0.15}
        defects = code.syndrome(err)
        if not defects or len(defects) > 16:
            continue
        pairs = code._dp_match(defects)
        touched = list(itertools.chain.from_iterable(pairs))
        assert sorted(touched) == list(range(len(defects)))
