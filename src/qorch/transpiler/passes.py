"""Pass manager: the transpile pipeline as an ordered, inspectable list of passes.

``transpile`` used to be a single function with the stage order baked into its
body. That worked, but it made two things hard that matter as the compiler grows:

  - **Seeing what happened.** A transpiled circuit is much larger than its input
    and it is not obvious which stage produced the growth. Attributing gates to
    the pass that emitted them is the difference between "routing inserted 9
    SWAPs" and "the output has 22 gates, somehow".
  - **Changing the order.** The pipeline's ordering constraints are real and
    subtle — lowering has to run after routing, DD after the optimizer — and
    they were expressed as the sequence of statements in one function. A list of
    named passes makes them a value that can be inspected, reordered, or
    extended (a layout pass slots in before routing) without rewriting control
    flow.

Passes thread a small :class:`PassState` alongside the circuit, because routing
produces information — the final layout — that later passes must not lose.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

from qorch.analysis import circuit_depth
from qorch.ir import Circuit, Gate


@dataclass(frozen=True)
class PassState:
    """Information passes accumulate that is not part of the circuit itself."""

    final_layout: tuple[int, ...]


# A pass maps (circuit, state) to a new (circuit, state).
Pass = Callable[[Circuit, PassState], tuple[Circuit, PassState]]


def _two_qubit_count(circuit: Circuit) -> int:
    return sum(1 for op in circuit.gates if len(op.qubits) == 2)


def _swap_count(circuit: Circuit) -> int:
    return sum(1 for op in circuit.gates
               if isinstance(op, Gate) and op.name == "swap")


@dataclass(frozen=True)
class PassMetrics:
    """What a single pass did to the circuit."""

    name: str
    gate_count: int
    two_qubit_count: int
    depth: int

    @property
    def summary(self) -> str:
        return (f"{self.name}: {self.gate_count} gates "
                f"({self.two_qubit_count} 2q), depth {self.depth}")


@dataclass(frozen=True)
class TranspileMetrics:
    """Aggregate cost of a transpile, plus the per-pass breakdown.

    ``swaps_inserted`` is counted at the moment routing finishes, before lowering
    turns each SWAP into a CX triple. Counting afterwards would report zero on
    every target without a native SWAP, which is most of them — and the number a
    caller wants is how much the topology cost them, not how it was spelled.
    """

    input_gate_count: int
    output_gate_count: int
    two_qubit_count: int
    swaps_inserted: int
    depth: int
    passes: tuple[PassMetrics, ...]

    def format(self) -> str:
        lines = [
            "Transpile metrics",
            "=================",
            f"  Gates:    {self.input_gate_count} → {self.output_gate_count}",
            f"  2q gates: {self.two_qubit_count}",
            f"  SWAPs:    {self.swaps_inserted}",
            f"  Depth:    {self.depth}",
            "  Passes:",
        ]
        lines += [f"    {p.summary}" for p in self.passes]
        return "\n".join(lines)


@dataclass(frozen=True)
class PassManager:
    """An ordered pipeline of named passes.

    Running it records what each pass did, so the cost of a transpile can be
    attributed rather than guessed at.
    """

    passes: tuple[tuple[str, Pass], ...]

    def run(self, circuit: Circuit) -> tuple[Circuit, PassState, TranspileMetrics]:
        state = PassState(final_layout=tuple(range(circuit.num_qubits)))
        current = circuit
        records: list[PassMetrics] = []
        swaps_inserted = 0

        for name, pass_fn in self.passes:
            current, state = pass_fn(current, state)
            records.append(PassMetrics(
                name=name,
                gate_count=len(current.gates),
                two_qubit_count=_two_qubit_count(current),
                depth=circuit_depth(current),
            ))
            # Count SWAPs while they still exist as SWAPs (see TranspileMetrics).
            if name.startswith("route"):
                swaps_inserted = _swap_count(current)

        metrics = TranspileMetrics(
            input_gate_count=len(circuit.gates),
            output_gate_count=len(current.gates),
            two_qubit_count=_two_qubit_count(current),
            swaps_inserted=swaps_inserted,
            depth=circuit_depth(current),
            passes=tuple(records),
        )
        return current, state, metrics


def circuit_pass(fn: Callable[[Circuit], Circuit]) -> Pass:
    """Lift a plain ``Circuit -> Circuit`` function into a pass.

    Most passes neither read nor write :class:`PassState`; only routing does.
    """
    def run(circuit: Circuit, state: PassState) -> tuple[Circuit, PassState]:
        return fn(circuit), state
    return run


def with_layout(
    fn: Callable[[Circuit], tuple[Circuit, tuple[int, ...]]],
) -> Pass:
    """Lift a routing function that also reports a layout, **composing** layouts.

    Routing reports where the qubit that *entered* on each wire ended up. If a
    layout pass already moved logical qubit ``q`` onto wire ``state[q]``, then
    ``q`` finishes on ``routed[state[q]]`` — composition, not replacement.

    Overwriting would be silently correct only while the incoming layout is the
    identity, i.e. exactly until a layout pass exists. It now does.
    """
    def run(circuit: Circuit, state: PassState) -> tuple[Circuit, PassState]:
        routed, layout = fn(circuit)
        composed = tuple(layout[wire] for wire in state.final_layout)
        return routed, replace(state, final_layout=composed)
    return run


def layout_pass(fn: Callable[[Circuit], tuple[Circuit, tuple[int, ...]]]) -> Pass:
    """Lift a layout function, which *establishes* the initial placement."""
    def run(circuit: Circuit, state: PassState) -> tuple[Circuit, PassState]:
        placed, layout = fn(circuit)
        composed = tuple(layout[wire] for wire in state.final_layout)
        return placed, replace(state, final_layout=composed)
    return run
