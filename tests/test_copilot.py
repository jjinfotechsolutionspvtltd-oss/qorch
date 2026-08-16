"""A circuit assistant that is grounded by construction and runs offline.

Two halves. Template-first matching, which cannot invent a circuit that does not
exist and is allowed to say it does not understand. And verification, which is
what makes accepting a circuit from an untrusted producer — a language model, a
script — safe: the guarantee comes from executing the result, not from believing
the source.

There is no model and no network anywhere in this, which is the point: the
assistant works air-gapped, like the rest of the library.
"""

from __future__ import annotations

import pytest

from qorch import Circuit, LocalSimulator
from qorch.copilot import (
    TEMPLATES,
    accept_free_form,
    assist,
    build,
    suggest,
    verify,
)


# ── template matching ────────────────────────────────────────────────────


@pytest.mark.parametrize("request_text,expected", [
    ("make me a bell pair", "bell"),
    ("I want a GHZ cat state", "ghz"),
    ("grover search for a marked item", "grover"),
    ("teleport a qubit", "teleportation"),
    ("correct a bit flip error", "repetition-code"),
    ("quantum fourier transform", "qft"),
    ("equal superposition over everything", "uniform"),
])
def test_requests_match_the_right_template(request_text: str, expected: str) -> None:
    assert suggest(request_text)[0].template == expected


def test_an_unrelated_request_matches_nothing() -> None:
    """Returning nothing is the honest answer, and better than the nearest guess."""
    assert suggest("bake me a cake") == []


def test_matches_explain_themselves() -> None:
    """A suggestion the caller cannot interrogate is not much of a suggestion."""
    match = suggest("bell pair")[0]
    assert "bell" in match.matched_terms
    assert match.description
    assert match.score > 0


def test_suggestions_are_ranked_and_limited() -> None:
    matches = suggest("entangled bell pair state", limit=2)
    assert len(matches) <= 2
    assert matches == sorted(matches, key=lambda m: (-m.score, m.template))


def test_matching_is_deterministic() -> None:
    assert suggest("ghz state") == suggest("ghz state")


# ── every template in the library actually works ─────────────────────────


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_template_builds_and_verifies(name: str) -> None:
    """The library's whole claim is that these are known-good."""
    template = TEMPLATES[name]
    circuit = build(name)
    expected = template.expected(circuit) if template.expected else None
    assert verify(circuit, expected=expected, shots=256).ok


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_template_is_described_and_has_vocabulary(name: str) -> None:
    template = TEMPLATES[name]
    assert template.description.strip()
    assert template.keywords


def test_bell_template_produces_only_correlated_outcomes() -> None:
    counts = LocalSimulator(seed=1).run(build("bell"), shots=500).counts
    assert set(counts) == {"00", "11"}


def test_ghz_respects_its_qubit_parameter() -> None:
    assert build("ghz", num_qubits=4).num_qubits == 4


def test_an_unknown_template_lists_the_real_ones() -> None:
    with pytest.raises(ValueError, match="unknown template"):
        build("does-not-exist")


# ── verification is the grounding ────────────────────────────────────────


def test_a_working_circuit_verifies() -> None:
    result = verify(Circuit(2).h(0).cx(0, 1).measure(0, 1), shots=200)
    assert result.ok
    assert result.counts
    assert not result.failures


def test_an_empty_circuit_fails_verification() -> None:
    assert not verify(Circuit(2)).ok


def test_a_circuit_that_declares_clbits_but_never_writes_them_fails() -> None:
    """The read-out check worth making: this returns all-zeros whatever it computes.

    A static circuit always reads out something, so checking for that would be
    dead code. A dynamic circuit reports its classical register — and one that
    never measures into it looks like it ran and tells you nothing.
    """
    dead = Circuit(2, num_clbits=2).h(0).cx(0, 1)
    assert LocalSimulator(seed=1).run(dead, shots=50).counts == {"00": 50}

    result = verify(dead)
    assert not result.ok
    assert "never measures into them" in result.failures[0]


def test_a_proper_dynamic_circuit_verifies() -> None:
    good = Circuit(2, num_clbits=2).h(0).measure_into(0, 0).measure_into(1, 1)
    assert verify(good).ok


def test_verification_rejects_a_circuit_too_large_to_simulate() -> None:
    """Verification must not become the thing that hangs."""
    result = verify(Circuit(30))
    assert not result.ok
    assert "too many" in result.failures[0]


def test_an_unexpected_outcome_fails_verification() -> None:
    """If the caller knows the answer, producing a different one is a failure."""
    result = verify(Circuit(1).h(0).measure(0), expected={"0"}, shots=200)
    assert not result.ok
    assert "outside the expected set" in result.failures[0]


def test_verification_reports_what_it_checked() -> None:
    text = verify(Circuit(2).h(0).cx(0, 1).measure(0, 1), shots=100).format()
    assert "PASS" in text
    assert "simulable" in text


# ── the assembled assistant ──────────────────────────────────────────────


def test_assist_returns_a_verified_circuit_with_an_explanation() -> None:
    result = assist("build me a bell pair", shots=256)

    assert result.template == "bell"
    assert result.circuit is not None
    assert result.verification.ok
    assert "bell" in result.explanation.lower()


def test_assist_refuses_rather_than_guessing() -> None:
    """The failure mode that matters: no circuit, and it says why."""
    result = assist("write me a poem about ducks")

    assert result.template is None
    assert result.circuit is None
    assert "No verified template matches" in result.explanation
    assert "bell" in result.explanation      # names what it *can* do


def test_assist_offers_alternatives() -> None:
    result = assist("entangled bell ghz state")
    assert result.alternatives


def test_assist_never_returns_an_unverified_circuit() -> None:
    for request in ("bell pair", "ghz state", "teleport a qubit", "grover search"):
        result = assist(request, shots=128)
        if result.circuit is not None:
            assert result.verification.ok


# ── free-form: verified, not trusted ─────────────────────────────────────


def test_a_valid_free_form_spec_is_accepted() -> None:
    result = accept_free_form({
        "num_qubits": 2,
        "gates": [{"name": "h", "qubits": [0]}, {"name": "cx", "qubits": [0, 1]}],
        "measured": [0, 1],
    })
    assert result.verification.ok
    assert result.circuit is not None


def test_an_invalid_spec_is_rejected_without_raising() -> None:
    result = accept_free_form({"num_qubits": 2, "gates": [{"name": "nope", "qubits": [0]}]})
    assert not result.verification.ok
    assert result.circuit is None
    assert "spec rejected" in result.verification.failures[0]


def test_a_failing_free_form_circuit_is_not_returned() -> None:
    """The guarantee: what comes back has been executed, not merely parsed."""
    result = accept_free_form({"num_qubits": 2, "num_clbits": 1, "gates": []})
    assert not result.verification.ok
    assert result.circuit is None


def test_free_form_explains_its_decision() -> None:
    """Two distinct rejections, distinguished: unparseable vs parsed-but-broken.

    They fail at different stages and a caller can act on the difference — a bad
    spec needs rewriting, a spec that parses but does not work needs rethinking.
    """
    accepted = accept_free_form({
        "num_qubits": 1, "gates": [{"name": "h", "qubits": [0]}],
    })
    assert "verified by execution" in accepted.explanation

    unparseable = accept_free_form({
        "num_qubits": 1, "gates": [{"name": "bad", "qubits": [0]}],
    })
    assert "not valid and was not accepted" in unparseable.explanation

    parses_but_useless = accept_free_form({
        "num_qubits": 2, "num_clbits": 1, "gates": [],
    })
    assert "failed verification and was not returned" in parses_but_useless.explanation


# ── it really is offline ─────────────────────────────────────────────────


def test_the_copilot_imports_nothing_networked() -> None:
    """Air-gapped operation is a property of this library, not an aspiration."""
    import inspect

    import qorch.copilot as copilot

    source = inspect.getsource(copilot)
    for forbidden in ("import requests", "import urllib", "import socket",
                      "import http", "openai", "anthropic"):
        assert forbidden not in source
