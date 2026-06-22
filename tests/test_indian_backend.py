"""Tests for the Indian QPU backend adapter."""

from qorch import Circuit, IndianQPU, INDIAN_QPU_CONFIGS


def test_all_presets_available():
    assert "iit-jodhpur-ion-trap" in INDIAN_QPU_CONFIGS
    assert "tifr-superconducting" in INDIAN_QPU_CONFIGS
    assert "drdo-mirai" in INDIAN_QPU_CONFIGS


def test_from_preset_creates_backend():
    qpu = IndianQPU.from_preset("tifr-superconducting", seed=42)
    assert qpu.name == "TIFR Superconducting"


def test_from_preset_raises_on_unknown():
    try:
        IndianQPU.from_preset("nonexistent")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_backend_properties():
    qpu = IndianQPU.from_preset("tifr-superconducting")
    props = qpu.properties()
    assert props.num_qubits > 0
    assert len(props.basis_gates) > 0
    assert len(props.readout_fidelity) == props.num_qubits


def test_bell_state_on_tifr():
    """Bell state on noisy Indian QPU — '00' and '11' dominate over errors."""
    qpu = IndianQPU.from_preset("tifr-superconducting", seed=42)
    bell = Circuit(num_qubits=2).h(0).cx(0, 1)
    result = qpu.run(bell, shots=2000)
    total = sum(result.counts.values())
    dominant = result.counts.get("00", 0) + result.counts.get("11", 0)
    assert dominant / total > 0.7


def test_bell_state_on_ion_trap():
    qpu = IndianQPU.from_preset("iit-jodhpur-ion-trap", seed=42)
    bell = Circuit(num_qubits=2).h(0).cx(0, 1)
    result = qpu.run(bell, shots=2000)
    total = sum(result.counts.values())
    dominant = result.counts.get("00", 0) + result.counts.get("11", 0)
    assert dominant / total > 0.7


def test_bell_state_on_drdo():
    qpu = IndianQPU.from_preset("drdo-mirai", seed=42)
    bell = Circuit(num_qubits=2).h(0).cx(0, 1)
    result = qpu.run(bell, shots=2000)
    total = sum(result.counts.values())
    dominant = result.counts.get("00", 0) + result.counts.get("11", 0)
    assert dominant / total > 0.7


def test_x_gate_flips_qubit_on_tifr():
    qpu = IndianQPU.from_preset("tifr-superconducting", seed=1)
    result = qpu.run(Circuit(num_qubits=1).x(0), shots=500)
    assert result.counts.get("1", 0) > result.counts.get("0", 0)
