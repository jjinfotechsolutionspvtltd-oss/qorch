"""ASCII circuit drawer — renders a Circuit as a Unicode string diagram."""

from __future__ import annotations

from qorch.ir import Circuit

# ─ │ ═ ║ ● ○ ⊕ ⊗ ┼ ┤ ├ ┴ ┬ ┘ └ ┌ ┐

_TDQ = {
    "h": "H",
    "x": "X",
    "y": "Y",
    "z": "Z",
    "sx": "SX",
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
    "id": "ID",
    "cx": "●",
    "swap": "×",
    "ms": "MS",
}

_GATE_WIDTH = 4  # total cells per gate column: " XX "


def _compute_layers(circuit: Circuit) -> list[list[int]]:
    """Assign each gate to a layer (timestep) using greedy scheduling."""
    n = len(circuit.gates)
    if n == 0:
        return []
    last_used: dict[int, int] = {}
    layers: list[list[int]] = []
    gate_to_layer: dict[int, int] = {}
    for i, g in enumerate(circuit.gates):
        start = max(last_used.get(q, -1) for q in g.qubits) + 1
        for q in g.qubits:
            last_used[q] = start
        gate_to_layer[i] = start
    max_layer = max(gate_to_layer.values()) if gate_to_layer else -1
    for _ in range(max_layer + 1):
        layers.append([])
    for i, g in enumerate(circuit.gates):
        layers[gate_to_layer[i]].append(i)
    return layers


def _gate_label(gate_idx: int, circuit: Circuit) -> str:
    """Return display label for a gate at this position."""
    g = circuit.gates[gate_idx]
    label = _TDQ.get(g.name, g.name.upper())
    if g.params:
        p = g.params[0]
        label = f"{label}({p:.3g})"
    return label


def draw_circuit(circuit: Circuit) -> str:
    """Render a circuit as a Unicode string diagram.

    Qubits are shown as horizontal lines.  Single-qubit gates are labeled
    boxes.  Two-qubit gates show connections with vertical lines.
    """
    layers = _compute_layers(circuit)
    nq = circuit.num_qubits
    nlayers = len(layers)

    # For each qubit row, for each layer, store (gate_idx, is_end_of_2q, is_mid)
    grid: list[list[dict]] = []
    for q in range(nq):
        row: list[dict] = []
        for _ in range(nlayers):
            row.append({"gate": None, "ctrl": False, "tgt": False, "mid_2q": False})
        grid.append(row)

    for layer_idx, gidxs in enumerate(layers):
        for gi in gidxs:
            g = circuit.gates[gi]
            if len(g.qubits) == 2:
                q0, q1 = g.qubits
                minq, maxq = min(q0, q1), max(q0, q1)
                for q in range(minq, maxq + 1):
                    if q == q0:
                        grid[q][layer_idx]["ctrl"] = True
                        grid[q][layer_idx]["gate"] = gi
                    elif q == q1:
                        grid[q][layer_idx]["tgt"] = True
                        grid[q][layer_idx]["gate"] = gi
                    else:
                        grid[q][layer_idx]["mid_2q"] = True
            elif len(g.qubits) == 1:
                q = g.qubits[0]
                grid[q][layer_idx]["gate"] = gi

    lines: list[str] = []

    def _cell(layer: dict, q: int, layer_idx: int) -> str:
        """Build the display cell for one qubit at one layer."""
        if layer["mid_2q"]:
            return " │  "
        if layer["ctrl"] and layer["tgt"]:
            label = _gate_label(layer["gate"], circuit)
            return f" {label:<2}"
        if layer["ctrl"]:
            label = _gate_label(layer["gate"], circuit)
            if label == "●":
                return f" {label:<2}"
            label = label[:2]
            return f" {label:<2}"
        if layer["tgt"]:
            label = _gate_label(layer["gate"], circuit)
            label = label[:2]
            return f" {label:<2}"
        if layer["gate"] is not None:
            label = _gate_label(layer["gate"], circuit)
            label = label[:3]
            return f"{label:<4}"
        return " ────"

    def _mid_cell(layer: dict, q: int, layer_idx: int) -> str:
        """Cell for the connector line between qubit rows."""
        if layer["mid_2q"]:
            return " │  "
        if layer["ctrl"] or layer["tgt"]:
            gid = layer["gate"]
            if gid is not None:
                g = circuit.gates[gid]
                if len(g.qubits) == 2:
                    q0, q1 = g.qubits
                    if q0 == q and q1 - q0 > 1:
                        return " │  "
        return "    "

    # Draw each qubit row
    for q in range(nq):
        row_str = f"q[{q}]:"
        for lidx in range(nlayers):
            row_str += _cell(grid[q][lidx], q, lidx)
        row_str += " ═"
        lines.append(row_str)

        if q < nq - 1:
            mid_str = "      "
            has_mid = False
            for lidx in range(nlayers):
                mc = _mid_cell(grid[q][lidx], q, lidx)
                if mc.strip():
                    has_mid = True
                mid_str += mc
            if has_mid:
                lines.append(mid_str)

    # Measurement marker
    measured = circuit.readout_qubits
    if measured:
        meas_row = "     "
        for q in range(nq):
            if q in measured:
                meas_row += " M  "
            else:
                meas_row += "    "
        meas_row += "    "
        lines.append("")

        wire_bottom = ""
        for q in range(nq):
            if q in measured:
                wire_bottom += " ═══"
            else:
                wire_bottom += " ────"
        wire_bottom = "      " + wire_bottom
        lines.append(meas_row)
        lines.append(wire_bottom)

    return "\n".join(lines) + "\n"


def print_circuit(circuit: Circuit) -> None:
    """Print a circuit diagram to stdout."""
    print(draw_circuit(circuit))
