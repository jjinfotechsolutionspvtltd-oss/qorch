"""Backend API v2: calibration on the HAL + exact T1/T2 noise path (A9)."""

from __future__ import annotations

from qorch import Circuit, IndianQPU, LocalSimulator
from qorch.backends.base import DeviceCalibration


def test_base_backend_calibration_defaults_none():
    """Simple backends stay a 3-method adapter: calibration/coupling default None."""
    sim = LocalSimulator()
    assert sim.calibration() is None
    assert sim.coupling_map() is None


def test_indian_qpu_exposes_calibration():
    qpu = IndianQPU.from_preset("tifr-superconducting")
    cal = qpu.calibration()
    assert isinstance(cal, DeviceCalibration)
    assert cal.num_qubits == 5
    # T1/T2 surfaced from the config (previously stored-but-unused)
    assert cal.qubits[0].t1_us == 50.0
    assert cal.qubits[0].t2_us == 30.0
    assert 0.0 < cal.qubits[0].readout_fidelity <= 1.0
    # per-edge two-qubit error + connectivity present
    assert cal.coupling_map == qpu.coupling_map()
    assert len(cal.two_qubit_error) == len(cal.coupling_map)
    assert cal.gate_durations_us  # non-empty


def test_indian_qpu_exact_noise_runs_and_is_labeled():
    """exact_noise=True drives an exact density-matrix sim from T1/T2 (A9)."""
    qpu = IndianQPU.from_preset("tifr-superconducting", seed=1, exact_noise=True)
    result = qpu.run(Circuit(2).h(0).cx(0, 1), shots=512)
    assert sum(result.counts.values()) == 512
    assert result.metadata["noise_model"] == "density-t1t2"
    # Bell correlation still dominant under realistic noise
    corr = result.counts.get("00", 0) + result.counts.get("11", 0)
    assert corr / 512 > 0.7


def test_density_sim_from_calibration_uses_t1t2():
    qpu = IndianQPU.from_preset("iit-jodhpur-ion-trap")
    from qorch.backends.density_simulator import DensitySimulator
    sim = DensitySimulator.from_calibration(qpu.calibration(), seed=0)
    # T1/T2 produced non-trivial amplitude/phase damping
    assert sim._noise.amplitude_damping_gamma > 0.0
    assert sim._noise.phase_damping_lambda > 0.0
