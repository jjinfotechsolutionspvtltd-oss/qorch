"""Command-line interface for qorch — quantum circuit submission without Python."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qorch.backends.base import Backend
from qorch.backends.indian_backend import IndianQPU, INDIAN_QPU_CONFIGS
from qorch.backends.simulator import LocalSimulator
from qorch.ir import Circuit, from_qasm3
from qorch.mitigation.readout import ReadoutMitigator
from qorch.mitigation.zne import expectation_z, zne_expectation
from qorch.scheduler import BatchJob, Scheduler


def _build_backend(name: str, seed: int | None = None) -> Backend:
    if name == "local-simulator":
        return LocalSimulator(seed=seed)
    if name in INDIAN_QPU_CONFIGS:
        return IndianQPU.from_preset(name, seed=seed)
    raise ValueError(
        f"unknown backend: {name!r}. Available: local-simulator, {list(INDIAN_QPU_CONFIGS)}"
    )


def _print_histogram(counts: dict[str, float | int], shots: int, label: str = "") -> None:
    max_key_len = max(len(k) for k in counts) if counts else 1
    max_bar = 50
    top = max(counts.values()) if counts else 1
    header = f"{'Outcome':>{max_key_len}}  {'Count':>6}  {'%':>5}  Distribution"
    if label:
        print(f"\n{label}")
    print(header)
    print("-" * len(header))
    for bitstring in sorted(counts):
        count = counts[bitstring]
        pct = count / shots * 100 if shots else 0
        bar_len = max(1, int(count / top * max_bar)) if top else 1
        bar = "█" * bar_len
        print(f"  {bitstring:>{max_key_len}}  {int(count):6d}  {pct:5.1f}%  {bar}")


def cmd_run(args: argparse.Namespace) -> None:
    circuit = _load_circuit(args.circuit, args.gates)
    backend = _build_backend(args.backend, seed=args.seed)
    result = backend.run(circuit, shots=args.shots)
    print(f"Backend: {result.backend_name}")
    print(f"Shots:   {result.shots}")
    _print_histogram(result.counts, result.shots, "Counts")


def cmd_mitigate(args: argparse.Namespace) -> None:
    circuit = _load_circuit(args.circuit, args.gates)
    if args.technique == "readout":
        _mitigate_readout(circuit, args)
    elif args.technique == "zne":
        _mitigate_zne(circuit, args)
    elif args.technique == "pec":
        _mitigate_pec(circuit, args)
    else:
        print(f"unknown technique: {args.technique}", file=sys.stderr)
        sys.exit(1)


def _mitigate_readout(circuit: Circuit, args: argparse.Namespace) -> None:
    p1 = args.p1_given0 or 0.05
    p0 = args.p0_given1 or 0.15
    from qorch import ReadoutNoise

    noisy_backend = LocalSimulator(
        seed=args.seed, readout_noise=ReadoutNoise(p1_given0=p1, p0_given1=p0)
    )
    raw = noisy_backend.run(circuit, shots=args.shots)
    mitigator = ReadoutMitigator.from_calibration_matrix(
        labels=["0", "1"],
        matrix=[[1 - p1, p0], [p1, 1 - p0]],
    )
    mitigated = mitigator.apply(raw.counts)
    _print_histogram(raw.counts, args.shots, "Raw (noisy)")
    _print_histogram(mitigated, args.shots, "Mitigated")


def _mitigate_zne(circuit: Circuit, args: argparse.Namespace) -> None:
    from qorch import GateNoise

    backend = LocalSimulator(
        seed=args.seed,
        gate_noise=GateNoise(depolarizing_prob=args.depolarizing or 0.05),
    )
    scales = tuple(int(s) for s in (args.scales or "1,3,5").split(","))
    result = zne_expectation(
        backend, circuit, expectation_z, scales=scales, shots=args.shots
    )
    print(f"Scales:       {result.scales}")
    print(f"Values:       {[round(v, 4) for v in result.values]}")
    print(f"Raw <Z>:      {result.raw:.4f}")
    print(f"Mitigated <Z>: {result.mitigated:.4f}")
    imp = (abs(result.mitigated - 1.0) - abs(result.raw - 1.0)) / (
        abs(result.raw - 1.0) + 1e-12
    )
    print(f"Improvement:  {imp * 100:+.1f}%")


def _mitigate_pec(circuit: Circuit, args: argparse.Namespace) -> None:
    from qorch import GateNoise
    from qorch.mitigation.pec import pec_expectation

    backend = LocalSimulator(
        seed=args.seed,
        gate_noise=GateNoise(depolarizing_prob=args.depolarizing or 0.05),
    )
    result = pec_expectation(
        backend,
        circuit,
        expectation_z,
        noise_prob=args.depolarizing or 0.05,
        n_samples=args.pec_samples or 50,
        shots_per_sample=args.shots // (args.pec_samples or 50),
        seed=args.seed,
    )
    print("PEC Result")
    print(f"  Raw <Z>:      {result.raw_value:.4f}")
    print(f"  Mitigated <Z>: {result.mitigated_value:.4f}")
    print(f"  Improvement:  {result.improvement:+.1f}%")
    print(f"  Samples:       {result.n_samples}")
    print(f"  Gamma:         {result.gamma:.3f}")


def cmd_batch(args: argparse.Namespace) -> None:
    circuit = _load_circuit(args.circuit, args.gates)
    backends = args.backends.split(",")
    print(f"Circuit: {circuit.num_qubits} qubits, {len(circuit.gates)} gates")
    print(f"{'Backend':>25}  {'Shots':>6}  {'Top outcome':>12}  {'Frac':>6}")
    print("-" * 60)
    for name in backends:
        backend = _build_backend(name.strip(), seed=args.seed)
        result = backend.run(circuit, shots=args.shots)
        top = max(result.counts, key=result.counts.get)
        frac = result.counts[top] / result.shots
        print(f"{result.backend_name:>25}  {result.shots:>6}  {top:>12}  {frac:>6.3f}")


def cmd_sched(args: argparse.Namespace) -> None:
    """Run multiple circuits through the batch scheduler."""
    backends = [_build_backend(name.strip(), seed=args.seed) for name in args.backends.split(",")]
    sched = Scheduler(backends=backends)
    jobs: list[BatchJob] = []
    for spec in args.spec:
        if ":" in spec:
            label, gates_str = spec.split(":", 1)
            c = _parse_gates(gates_str)
        else:
            label = Path(spec).stem
            c = from_qasm3(Path(spec).read_text())
        jobs.append(BatchJob(circuit=c, shots=args.shots, label=label))
    results = sched.run_batch(jobs)
    print(f"{'Label':>20}  {'Backend':>25}  {'Shots':>6}  {'Top outcome':>12}  {'Frac':>6}")
    print("-" * 80)
    for r in results:
        if r.error:
            print(f"{r.label:>20}  {'ERROR':>25}  {'—':>6}  {r.error:>30}")
        else:
            top = max(r.result.counts, key=r.result.counts.get)
            frac = r.result.counts[top] / r.result.shots
            print(f"{r.label:>20}  {r.backend_name:>25}  {r.result.shots:>6}  {top:>12}  {frac:>6.3f}")


def cmd_report(args: argparse.Namespace) -> None:
    circuit = _load_circuit(args.circuit, args.gates)
    from qorch.analysis import circuit_report, format_report

    report = circuit_report(circuit, gate_error_rate=args.error_rate or 0.01)
    print(format_report(report))


def cmd_list(args: argparse.Namespace) -> None:
    print("Backends:")
    print("  local-simulator            (dependency-free statevector)")
    for name, cfg in sorted(INDIAN_QPU_CONFIGS.items()):
        print(f"  {name:26s} ({cfg.num_qubits} qubits, {cfg.gate_set.basis_gates})")
    print()
    print("Mitigation techniques:")
    print("  readout  — Readout-error calibration (M1)")
    print("  zne      — Zero-noise extrapolation (M3)")
    print("  pec      — Probabilistic Error Cancellation (research)")
    print()
    print("Commands:")
    print("  run        — Execute a circuit on a backend")
    print("  mitigate   — Run with error mitigation")
    print("  transpile  — Decompose for an Indian QPU")
    print("  batch      — Compare backends on the same circuit")
    print("  sched      — Batch-schedule multiple circuits")
    print("  report     — Circuit depth, gate count, fidelity analysis")
    print("  certify    — Run QPU certification suite")
    print("  qv-sweep   — Quantum Volume sweep across widths")
    print("  list       — This help")


def cmd_certify(args: argparse.Namespace) -> None:
    """Run the full QPU certification suite."""
    backend = _build_backend(args.backend, seed=args.seed)
    props = backend.properties()
    nq = props.num_qubits

    print("=" * 52)
    print("  QPU Certification Report")
    print(f"  Backend: {backend.name}")
    print(f"  Qubits:  {nq}")
    print(f"  Shots:   {args.shots}")
    print("=" * 52)

    # 1 — Bell state fidelity
    print("\n  -- 1. Bell State Fidelity --")
    from qorch.entanglement import bell_state_fidelity
    bell = bell_state_fidelity(backend, shots=args.shots)
    if bell.fidelity is not None:
        status = "PASS" if bell.fidelity > 0.5 else "FAIL"
        print(f"    F = {bell.fidelity:.4f}  [{status}] (threshold: 0.5)")
    else:
        print("    F = N/A")

    # 2 — CHSH S-value
    print("\n  -- 2. CHSH S-Value --")
    from qorch.entanglement import chsh_s_value
    chsh = chsh_s_value(backend, shots=args.shots)
    if chsh.s_value is not None:
        status = "VIOLATION" if chsh.violation else "no violation"
        print(f"    S = {chsh.s_value:.4f}  [{status}] (threshold: 2.0)")
    else:
        print("    S = N/A")

    # 3 — Entanglement witness
    print("\n  -- 3. Entanglement Witness --")
    from qorch.entanglement import entanglement_witness
    ew = entanglement_witness(backend, shots=args.shots)
    if ew.witness_value is not None:
        status = "ENTANGLED" if ew.entangled else "separable"
        print(f"    W = {ew.witness_value:.4f}  [{status}] (threshold: 0)")
    else:
        print("    W = N/A")

    # 4 — Randomized benchmarking (1-qubit)
    print("\n  -- 4. Randomized Benchmarking (1Q) --")
    from qorch.benchmarking import randomized_benchmarking
    rb = randomized_benchmarking(backend, depths=(1, 2, 4, 8), circuits_per_depth=3, shots=args.shots // 4, seed=args.seed)
    if rb.estimated_error_rate is not None:
        print(f"    Error rate r = {rb.estimated_error_rate:.6f}")
        print(f"    Gate fidelity ~ {1 - rb.estimated_error_rate:.6f}")
    else:
        print("    Error rate = N/A (need >=2 depths)")

    # 5 — Quantum Volume
    print("\n  -- 5. Quantum Volume --")
    if nq >= 2:
        from qorch.benchmarking import quantum_volume
        w = min(4, nq)
        qv = quantum_volume(backend, width=w, shots=args.shots // 2, trials=5, seed=args.seed)
        if qv.heavy_output_probability is not None:
            hop_str = f"HOP = {qv.heavy_output_probability:.4f}"
            if qv.success:
                hop_str += f"  [QV >= {2**w}]"
            else:
                hop_str += f"  [QV < {2**w}]"
            print(f"    {hop_str} (threshold: 2/3)")
        else:
            print("    HOP = N/A")
    else:
        print(f"    (need >=2 qubits, backend has {nq})")

    print()


def cmd_qv_sweep(args: argparse.Namespace) -> None:
    """Run a Quantum Volume sweep across qubit widths."""
    backend = _build_backend(args.backend, seed=args.seed)
    from qorch.benchmarking import qv_sweep

    result = qv_sweep(
        backend,
        start_width=args.start,
        end_width=args.end,
        trials=args.trials,
        shots=args.shots,
        seed=args.seed,
    )

    print(f"QV Sweep on {result.quantum_volume} (max passing width = {result.max_passing_width})")
    print()
    print(f"{'Width':>6}  {'Depth':>6}  {'HOP':>8}  {'Status':>10}")
    print("-" * 40)
    for r in result.results:
        status = "PASS" if r.success else ("FAIL" if r.success is False else "N/A")
        hop = f"{r.heavy_output_probability:.4f}" if r.heavy_output_probability is not None else "N/A"
        print(f"{r.width:>6}  {r.depth:>6}  {hop:>8}  {status:>10}")


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
        extra = f" θ={g.params[0]:.3f}" if g.params else ""
        print(f"  [{i}] {g.name} q{g.qubits}{extra}")


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
    parser = argparse.ArgumentParser(description="qorch — Indian Quantum Orchestration Layer")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")

    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run a circuit on a backend")
    run_p.add_argument("circuit", nargs="?", default=None, help=".qasm file")
    run_p.add_argument("--gates", default=None, help="Gates, e.g. 'h0,cx01'")
    run_p.add_argument("--backend", default="local-simulator", help="Backend name")
    run_p.add_argument("--shots", type=int, default=1024, help="Shots")

    mit_p = sub.add_parser("mitigate", help="Run with error mitigation")
    mit_p.add_argument("circuit", nargs="?", default=None, help=".qasm file")
    mit_p.add_argument("--gates", default=None, help="Gates, e.g. 'h0'")
    mit_p.add_argument("--technique", default="readout", choices=["readout", "zne", "pec"])
    mit_p.add_argument("--shots", type=int, default=8192, help="Shots")
    mit_p.add_argument("--p1-given0", type=float, default=None, help="Readout noise P(1|0)")
    mit_p.add_argument("--p0-given1", type=float, default=None, help="Readout noise P(0|1)")
    mit_p.add_argument("--depolarizing", type=float, default=None, help="Depolarizing prob")
    mit_p.add_argument("--scales", default=None, help="ZNE scales, e.g. '1,3,5'")
    mit_p.add_argument("--pec-samples", type=int, default=50, help="PEC samples")

    batch_p = sub.add_parser("batch", help="Compare backends on the same circuit")
    batch_p.add_argument("circuit", nargs="?", default=None, help=".qasm file")
    batch_p.add_argument("--gates", default=None, help="Gates, e.g. 'h0,cx01'")
    batch_p.add_argument("--backends", default="local-simulator,tifr-superconducting", help="Comma-separated backends")
    batch_p.add_argument("--shots", type=int, default=2048, help="Shots per backend")

    report_p = sub.add_parser("report", help="Circuit analysis report")
    report_p.add_argument("circuit", nargs="?", default=None, help=".qasm file")
    report_p.add_argument("--gates", default=None, help="Gates, e.g. 'h0,cx01'")
    report_p.add_argument("--error-rate", type=float, default=0.01, help="Per-gate error rate")

    transpile_p = sub.add_parser("transpile", help="Decompose for Indian QPU")
    transpile_p.add_argument("circuit", nargs="?", default=None, help=".qasm file")
    transpile_p.add_argument("--gates", default=None, help="Gates")
    transpile_p.add_argument("--target", required=True, choices=list(INDIAN_QPU_CONFIGS))

    sub.add_parser("list", help="List backends and techniques")

    cert_p = sub.add_parser("certify", help="Run QPU certification suite")
    cert_p.add_argument("--backend", default="local-simulator", help="Backend name")
    cert_p.add_argument("--shots", type=int, default=4096, help="Shots per benchmark")

    qv_p = sub.add_parser("qv-sweep", help="Quantum Volume sweep across widths")
    qv_p.add_argument("--backend", default="local-simulator", help="Backend name")
    qv_p.add_argument("--start", type=int, default=2, help="Starting width")
    qv_p.add_argument("--end", type=int, default=None, help="Ending width (default: backend qubits)")
    qv_p.add_argument("--trials", type=int, default=10, help="Trials per width")
    qv_p.add_argument("--shots", type=int, default=4096, help="Shots per trial")

    sched_p = sub.add_parser("sched", help="Batch-schedule multiple circuits")
    sched_p.add_argument("--spec", required=True, nargs="+", help="Circuits as label:gates or path/to/file.qasm")
    sched_p.add_argument("--backends", default="local-simulator", help="Comma-separated backends")
    sched_p.add_argument("--shots", type=int, default=2048, help="Shots per circuit")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "mitigate":
        cmd_mitigate(args)
    elif args.command == "batch":
        cmd_batch(args)
    elif args.command == "sched":
        cmd_sched(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "transpile":
        cmd_transpile(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "certify":
        cmd_certify(args)
    elif args.command == "qv-sweep":
        cmd_qv_sweep(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
