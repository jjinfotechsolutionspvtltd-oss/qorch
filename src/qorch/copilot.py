"""A circuit assistant that is grounded by construction and runs offline.

Roadmap 6.2 asks for "template-first + verified free-form, grounded and
offline-capable". The ordering in that phrase is the whole design.

**Template-first.** A request is matched against a library of circuits that are
known-good — they are the ones this library already implements and tests. The
matcher is keyword scoring, not a language model: it needs no network, no
weights, no API key, and it cannot invent a circuit that does not exist. It can
fail to understand a request, which is a far better failure than confidently
producing a plausible circuit that computes the wrong thing.

**Verified free-form.** For anything the templates do not cover, a caller — a
language model, a script, a person — supplies a circuit spec and it is *verified
before being returned*. That is what "grounded" means operationally: the
assistant's guarantee comes from checking the output, not from trusting whatever
produced it.

Nothing here calls out to a model. An LLM is a fine way to *produce* a candidate
spec, and this module is what makes accepting one safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from qorch.ir import Circuit

# ── the template library ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Template:
    """A known-good circuit, its vocabulary, and how to check it did its job."""

    name: str
    description: str
    keywords: frozenset[str]
    build: Callable[..., Circuit]
    defaults: dict[str, Any] = field(default_factory=dict)
    #: Expected measurement outcomes, when the template has a definite answer.
    expected: Callable[[Circuit], set[str]] | None = None


def _bell(num_qubits: int = 2) -> Circuit:
    return Circuit(2).h(0).cx(0, 1).measure(0, 1)


def _ghz(num_qubits: int = 3) -> Circuit:
    c = Circuit(num_qubits).h(0)
    for i in range(num_qubits - 1):
        c = c.cx(i, i + 1)
    return c.measure(*range(num_qubits))


def _uniform(num_qubits: int = 3) -> Circuit:
    c = Circuit(num_qubits)
    for q in range(num_qubits):
        c = c.h(q)
    return c.measure(*range(num_qubits))


def _qft(num_qubits: int = 3) -> Circuit:
    from qorch.adp import qft_circuit

    return qft_circuit(num_qubits).measure(*range(num_qubits))


def _grover(num_qubits: int = 3, target: str = "101") -> Circuit:
    from qorch.adp import grover_diffusion, oracle_by_bitstring, _combine_circuits

    target = target[:num_qubits].ljust(num_qubits, "0")
    c = Circuit(num_qubits)
    for q in range(num_qubits):
        c = c.h(q)
    for _ in range(max(1, int(math.pi / 4 * math.sqrt(2 ** num_qubits)))):
        c = _combine_circuits(c, oracle_by_bitstring(num_qubits, target))
        c = _combine_circuits(c, grover_diffusion(num_qubits))
    return c.measure(*range(num_qubits))


def _teleportation() -> Circuit:
    from qorch.dynamic import teleportation_circuit

    return teleportation_circuit(state_prep=Circuit(1).x(0))


def _repetition_code(error_qubit: int | None = 1) -> Circuit:
    from qorch.dynamic import repetition_code_circuit

    return repetition_code_circuit(error_qubit=error_qubit)


TEMPLATES: dict[str, Template] = {
    "bell": Template(
        name="bell",
        description="Two-qubit Bell pair — the canonical entangled state.",
        keywords=frozenset({"bell", "epr", "entangle", "entangled", "pair", "two"}),
        build=_bell,
        expected=lambda _c: {"00", "11"},
    ),
    "ghz": Template(
        name="ghz",
        description="N-qubit GHZ state — all-zeros plus all-ones.",
        keywords=frozenset({"ghz", "cat", "greenberger", "multipartite", "all"}),
        build=_ghz,
        defaults={"num_qubits": 3},
        expected=lambda c: {"0" * c.num_qubits, "1" * c.num_qubits},
    ),
    "uniform": Template(
        name="uniform",
        description="Equal superposition over every basis state.",
        keywords=frozenset({"uniform", "superposition", "hadamard", "equal", "random"}),
        build=_uniform,
        defaults={"num_qubits": 3},
    ),
    "qft": Template(
        name="qft",
        description="Quantum Fourier transform.",
        keywords=frozenset({"qft", "fourier", "transform", "phase", "frequency"}),
        build=_qft,
        defaults={"num_qubits": 3},
    ),
    "grover": Template(
        name="grover",
        description="Grover search amplifying one marked bitstring.",
        keywords=frozenset({"grover", "search", "amplitude", "amplification", "find",
                            "oracle", "unstructured"}),
        build=_grover,
        defaults={"num_qubits": 3, "target": "101"},
    ),
    "teleportation": Template(
        name="teleportation",
        description="Quantum teleportation with feed-forward corrections.",
        keywords=frozenset({"teleport", "teleportation", "feedforward", "feed",
                            "dynamic", "transfer"}),
        build=_teleportation,
    ),
    "repetition-code": Template(
        name="repetition-code",
        description="Three-qubit bit-flip code: encode, inject an error, correct.",
        keywords=frozenset({"repetition", "code", "error", "correction", "qec",
                            "syndrome", "bitflip", "correct"}),
        build=_repetition_code,
        defaults={"error_qubit": 1},
    ),
}


# ── intent matching (no model, no network) ───────────────────────────────


@dataclass(frozen=True)
class Match:
    """A candidate template and why it was suggested."""

    template: str
    score: float
    matched_terms: tuple[str, ...]
    description: str


def _tokenize(text: str) -> set[str]:
    return {
        word.strip(".,!?;:'\"()").lower()
        for word in text.split()
        if word.strip(".,!?;:'\"()")
    }


def suggest(request: str, limit: int = 3) -> list[Match]:
    """Rank templates against a request by keyword overlap.

    Deliberately simple and deliberately not a model. Scoring is the fraction of
    a template's vocabulary the request hits, so a short precise request ("bell
    state") beats a long vague one, and a request matching nothing returns
    nothing rather than the closest thing in the library.

    Returning an empty list is a supported, meaningful answer here: "I do not
    have a verified circuit for that" is the honest response, and far better
    than the nearest template presented as if it were what was asked for.
    """
    terms = _tokenize(request)
    matches: list[Match] = []
    for template in TEMPLATES.values():
        hits = terms & template.keywords
        if not hits:
            continue
        # Normalize by the template's own vocabulary so a template with many
        # keywords is not favoured merely for being verbose.
        score = len(hits) / math.sqrt(len(template.keywords))
        matches.append(Match(
            template=template.name,
            score=round(score, 4),
            matched_terms=tuple(sorted(hits)),
            description=template.description,
        ))
    matches.sort(key=lambda m: (-m.score, m.template))
    return matches[:limit]


def build(name: str, **params: Any) -> Circuit:
    """Build a template by name, with its defaults filled in."""
    template = TEMPLATES.get(name)
    if template is None:
        raise ValueError(
            f"unknown template {name!r}; available: {sorted(TEMPLATES)}"
        )
    merged = {**template.defaults, **params}
    return template.build(**merged)


# ── verification: the grounding ──────────────────────────────────────────


@dataclass(frozen=True)
class Verification:
    """What was checked about a circuit, and what failed."""

    ok: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]
    counts: dict[str, int] | None = None

    def format(self) -> str:
        lines = [f"Verification: {'PASS' if self.ok else 'FAIL'}"]
        lines += [f"  ok    {check}" for check in self.checks]
        lines += [f"  FAIL  {failure}" for failure in self.failures]
        return "\n".join(lines)


def verify(
    circuit: Circuit,
    expected: set[str] | None = None,
    shots: int = 512,
    seed: int | None = 7,
) -> Verification:
    """Check a circuit is well-formed, runnable, and — if known — correct.

    This is the function that makes accepting a circuit from an untrusted
    producer safe, so it checks the things that actually go wrong rather than
    the things that are easy to check:

    - it builds and its operations are in range (the IR enforces this, but a
      spec assembled elsewhere may never have been through the IR)
    - it is small enough to simulate, so verification itself cannot hang
    - it actually runs
    - a circuit that declares classical bits actually writes to them. This is
      the read-out check worth making: a static circuit always reads out
      something (the IR defaults to measuring every qubit), but a *dynamic*
      circuit reports its classical register, and one that declares clbits and
      never measures into them returns all-zeros no matter what it computed.
      That is a plausible thing for a generator to emit and an easy thing to
      mistake for a working circuit.
    - if the caller knows what the answer should be, it produces that answer
    """
    checks: list[str] = []
    failures: list[str] = []
    counts: dict[str, int] | None = None

    if circuit.num_qubits > 20:
        failures.append(f"{circuit.num_qubits} qubits is too many to verify by simulation")
        return Verification(False, tuple(checks), tuple(failures))
    checks.append(f"{circuit.num_qubits} qubits is simulable")

    if not circuit.gates:
        failures.append("circuit contains no operations")
    else:
        checks.append(f"{len(circuit.gates)} operations")

    from qorch.ir import Measure

    if circuit.num_clbits > 0 and not any(
        isinstance(op, Measure) for op in circuit.gates
    ):
        failures.append(
            f"declares {circuit.num_clbits} classical bits but never measures into "
            f"them, so every shot reports all-zeros regardless of what it computes"
        )
    else:
        checks.append(f"reads out {len(circuit.readout_qubits)} qubits")

    try:
        from qorch import LocalSimulator

        counts = LocalSimulator(seed=seed).run(circuit, shots=shots).counts
        checks.append(f"executed for {shots} shots")
    except Exception as exc:                                # noqa: BLE001
        failures.append(f"execution failed: {type(exc).__name__}: {exc}")
        return Verification(False, tuple(checks), tuple(failures), counts)

    if expected is not None:
        observed = set(counts)
        unexpected = observed - expected
        if unexpected:
            failures.append(
                f"produced outcomes outside the expected set: {sorted(unexpected)[:4]}"
            )
        else:
            checks.append(f"every outcome is in the expected set {sorted(expected)}")

    return Verification(not failures, tuple(checks), tuple(failures), counts)


# ── the assembled assistant ──────────────────────────────────────────────


@dataclass(frozen=True)
class Assistance:
    """A suggested circuit, why it was chosen, and proof that it works."""

    request: str
    template: str | None
    circuit: Circuit | None
    verification: Verification | None
    alternatives: tuple[Match, ...]
    explanation: str


def assist(request: str, shots: int = 512, seed: int | None = 7) -> Assistance:
    """Answer a request with a verified circuit, or with an honest refusal.

    A circuit is only ever returned *after* it has been built from a known
    template and verified by execution. If nothing matches, the alternatives are
    empty and the explanation says so — the assistant does not guess.
    """
    matches = suggest(request)
    if not matches:
        return Assistance(
            request=request,
            template=None,
            circuit=None,
            verification=None,
            alternatives=(),
            explanation=(
                "No verified template matches that request. Available templates: "
                + ", ".join(sorted(TEMPLATES))
            ),
        )

    best = matches[0]
    template = TEMPLATES[best.template]
    circuit = build(best.template)
    expected = template.expected(circuit) if template.expected else None
    verification = verify(circuit, expected=expected, shots=shots, seed=seed)

    return Assistance(
        request=request,
        template=best.template,
        circuit=circuit,
        verification=verification,
        alternatives=tuple(matches[1:]),
        explanation=(
            f"Matched template {best.template!r} on {', '.join(best.matched_terms)}. "
            f"{template.description} "
            f"Verification {'passed' if verification.ok else 'FAILED'}."
        ),
    )


def accept_free_form(
    spec: dict[str, Any], shots: int = 512, seed: int | None = 7
) -> Assistance:
    """Verify a circuit spec produced elsewhere — by a model, a script, a person.

    The counterpart to the template path, and the reason this module can be
    pointed at a language model safely: whatever produced the spec is untrusted,
    and the guarantee comes from executing the result rather than from believing
    the source.
    """
    from qorch.tools import _parse_circuit

    try:
        circuit = _parse_circuit(spec)
    except (ValueError, KeyError, TypeError) as exc:
        return Assistance(
            request="<free-form>",
            template=None,
            circuit=None,
            verification=Verification(
                False, (), (f"spec rejected: {type(exc).__name__}: {exc}",)
            ),
            alternatives=(),
            explanation="The supplied circuit spec is not valid and was not accepted.",
        )

    verification = verify(circuit, shots=shots, seed=seed)
    return Assistance(
        request="<free-form>",
        template=None,
        circuit=circuit if verification.ok else None,
        verification=verification,
        alternatives=(),
        explanation=(
            "Free-form circuit verified by execution."
            if verification.ok else
            "Free-form circuit failed verification and was not returned."
        ),
    )
