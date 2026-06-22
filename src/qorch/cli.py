"""Command-line interface for qorch — quantum circuit submission without Python."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qorch.backends.base import Backend
from qorch.backends.simulator import LocalSimulator
from qorch.backends.indian_backend import IndianQPU, INDIAN_QPU_CONFIGS
from qorch.ir import Circuit, from_qasm3
from qorch.mitigation.readout import ReadoutMitigator
from qorch.mitigation.zne import zne_expectation, expectation_z


def _build_backend(name: str, seed: int | None = None) -> Backend:
    if name == "local-simulator":
        return LocalSimulator(seed=seed)
    if name in INDIAN_QPU_CONFIGS:
        return IndianQPU.from_preset(name, seed=seed)
    raise ValueError(f"unknown backend: {name!r}. Available: local-simulator, {list(INDIAN_QPU_CONFIGS)}")


def cmd_run(args: argparse.Namespace) -> None:
    circuit = _load_circuit(args.circuit, args.gates)
    backend = _build_backend(args.backend, seed=args.seed)
    result = backend.run(circuit, shots=args.shots)
    print(f"Backend: {result.backend_name}")
    print(f"Shots:   {result.shots}")
    print("Counts:")
    for bitstring, count in sorted(result.counts.items()):
        bar = "#" * max(1, count * 40 // result.shots)
        print(f"  {bitstring}: {count:6d} {bar}")


def cmd_mitigate(args: argparse.Namespace) -> None:
    circuit = _load_circuit(args.circuit, args.gates)
    if args.technique == "readout":
        _mitigate_readout(circuit, args)
    elif args.technique == "zne":
        _mitigate_zne(circuit, args)
    else:
        print(f"unknown technique: {args.technique}", file=sys.stderr)
        sys.exit(1)


def _mitigate_readout(circuit: Circuit, args: argparse.Namespace) -> None:
    p1 = args.p1_given0 or 0.05
    p0 = args.p0_given1 or 0.15
    from qorch import ReadoutNoise
    noisy_backend = LocalSimulator(seed=args.seed, readout_noise=ReadoutNoise(p1_given0=p1, p0_given1=p0))
    raw = noisy_backend.run(circuit, shots=args.shots)
    mitigator = ReadoutMitigator.from_calibration_matrix(
        labels=["0", "1"],
        matrix=[[1 - p1, p0], [p1, 1 - p0]],
    )
    mitigated = mitigator.apply(raw.counts)
    for bitstring, count in sorted(raw.counts.items()):
        bar = "#" * max(1, count * 40 // args.shots)
        print(f"  raw {bitstring}: {count:6d} {bar}")
    for bitstring, val in sorted(mitigated.items()):
        bar = "#" * max(1, int(val) * 40 // args.shots)
        print(f"  mit {bitstring}: {val:6.1f} {bar}")


def _mitigate_zne(circuit: Circuit, args: argparse.Namespace) -> None:
    from qorch import GateNoise
    backend = LocalSimulator(seed=args.seed, gate_noise=GateNoise(depolarizing_prob=args.depolarizing or 0.05))
    scales = tuple(int(s) for s in (args.scales or "1,3,5").split(","))
    result = zne_expectation(backend, circuit, expectation_z, scales=scales, shots=args.shots)
    print(f"Scales: {result.scales}")
    print(f"Values: {[round(v, 4) for v in result.values]}")
    print(f"Raw <Z>:       {result.raw:.4f}")
    print(f"Mitigated <Z>: {result.mitigated:.4f}")
    improvement = (abs(result.mitigated - 1.0) - abs(result.raw - 1.0)) / (abs(result.raw - 1.0) + 1e-12)
    print(f"Improvement:   {improvement * 100:+.1f}%")


def cmd_list(args: argparse.Namespace) -> None:
    print("Available backends:")
    print("  local-simulator  (dependency-free statevector, stdlib only)")
    for name, cfg in INDIAN_QPU_CONFIGS.items():
        print(f"  {name:20s} ({cfg.num_qubits} qubits, {cfg.gate_set.basis_gates})")
    print()
    print("Available mitigation techniques:\n  readout  - Readout-error calibration\n  zne      - Zero-noise extrapolation")


def cmd_transpile(args: argparse.Namespace) -> None:
    circuit = _load_circuit(args.circuit, args.gates)
    backend = IndianQPU.from_preset(args.target, seed=args.seed)
    from qorch.transpiler import decompose, route
    decomposed = decompose(circuit, backend._config.gate_set)
    routed = route(decomposed, backend._config.coupling_map)
    print(f"Original gates:  {len(circuit.gates)}")
    print(f"Decomposed:      {len(decomposed.gates)}")
    print(f"Routed:          {len(routed.gates)}")
    print(f"Native gates:    {backend._config.gate_set.basis_gates}")
    print("Gate sequence:")
    for i, g in enumerate(routed.gates):
        print(f"  [{i}] {g.name} q{g.qubits}" + (f" θ={g.params[0]:.3f}" if g.params else ""))


def _load_circuit(circuit_path: str | None, gates_str: str | None) -> Circuit:
    if circuit_path:
        text = Path(circuit_path).read_text()
        return from_qasm3(text)
    if gates_str:
        return _parse_gates(gates_str)
    print("error: provide either a .qasm file or --gates", file=sys.stderr)
    sys.exit(1)


def _parse_gates(gates_str: str) -> Circuit:
    circuit = Circuit(num_qubits=2)
    gates = [g.strip() for g in gates_str.split(",")]
    for g in gates:
        if g == "h0":
            circuit = circuit.h(0)
        elif g == "h1":
            circuit = circuit.h(1)
        elif g == "x0":
            circuit = circuit.x(0)
        elif g == "cx01":
            circuit = circuit.cx(0, 1)
        elif g.startswith("measure"):
            circuit = circuit.measure(0, 1)
        else:
            print(f"warning: unknown gate {g!r}", file=sys.stderr)
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser(description="qorch — Quantum Orchestration Layer")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")

    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run a circuit on a backend")
    run_p.add_argument("circuit", nargs="?", default=None, help="Path to .qasm file")
    run_p.add_argument("--gates", default=None, help="Comma-separated gates (e.g. 'h0,cx01')")
    run_p.add_argument("--backend", default="local-simulator", help="Backend name")
    run_p.add_argument("--shots", type=int, default=1024, help="Number of shots")

    mit_p = sub.add_parser("mitigate", help="Run circuit with error mitigation")
    mit_p.add_argument("circuit", nargs="?", default=None, help="Path to .qasm file")
    mit_p.add_argument("--gates", default=None, help="Comma-separated gates")
    mit_p.add_argument("--technique", default="readout", choices=["readout", "zne"], help="Mitigation technique")
    mit_p.add_argument("--shots", type=int, default=8192, help="Number of shots")
    mit_p.add_argument("--p1-given0", type=float, default=None, help="Readout noise P(1|0)")
    mit_p.add_argument("--p0-given1", type=float, default=None, help="Readout noise P(0|1)")
    mit_p.add_argument("--depolarizing", type=float, default=None, help="Depolarizing prob for ZNE")
    mit_p.add_argument("--scales", default=None, help="ZNE scales (comma-separated, e.g. '1,3,5')")

    transpile_p = sub.add_parser("transpile", help="Decompose circuit for an Indian QPU")
    transpile_p.add_argument("circuit", nargs="?", default=None, help="Path to .qasm file")
    transpile_p.add_argument("--gates", default=None, help="Comma-separated gates")
    transpile_p.add_argument("--target", required=True, choices=list(INDIAN_QPU_CONFIGS), help="Target Indian QPU")

    sub.add_parser("list", help="List available backends and techniques")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "mitigate":
        cmd_mitigate(args)
    elif args.command == "transpile":
        cmd_transpile(args)
    elif args.command == "list":
        cmd_list(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
