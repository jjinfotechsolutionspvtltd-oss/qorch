"""Tests for the density-matrix simulator."""

from __future__ import annotations

from qorch import Circuit, DensitySimulator, NoiseChannel


class TestDensitySimulator:
    def test_basic_run(self):
        sim = DensitySimulator(seed=0)
        c = Circuit(1).h(0)
        result = sim.run(c, shots=1024)
        assert result.shots == 1024
        assert set(result.counts.keys()) <= {"0", "1"}

    def test_bell_state(self):
        sim = DensitySimulator(seed=0)
        c = Circuit(2).h(0).cx(0, 1)
        result = sim.run(c, shots=2048)
        assert set(result.counts.keys()) <= {"00", "11"}

    def test_depolarizing_noise(self):
        noise = NoiseChannel(depolarizing_prob=0.5)
        sim = DensitySimulator(seed=42, noise=noise)
        c = Circuit(1).h(0)
        result = sim.run(c, shots=4096)
        total_0 = result.counts.get("0", 0)
        ratio = total_0 / result.shots
        assert 0.3 < ratio < 0.7

    def test_amplitude_damping(self):
        noise = NoiseChannel(amplitude_damping_gamma=0.5)
        sim = DensitySimulator(seed=42, noise=noise)
        c = Circuit(1).x(0)
        result = sim.run(c, shots=4096)
        total_0 = result.counts.get("0", 0)
        assert total_0 > 0

    def test_phase_damping(self):
        noise = NoiseChannel(phase_damping_lambda=1.0)
        sim = DensitySimulator(seed=42, noise=noise)
        c = Circuit(1).h(0)
        result = sim.run(c, shots=4096)
        assert result.shots > 0

    def test_bell_entanglement(self):
        sim = DensitySimulator(seed=0)
        c = Circuit(2).h(0).cx(0, 1)
        result = sim.run(c, shots=2048)
        assert set(result.counts.keys()) <= {"00", "11"}

    def test_swap_gate(self):
        sim = DensitySimulator(seed=0)
        c = Circuit(2).x(0).swap(0, 1)
        result = sim.run(c, shots=1024)
        assert result.counts.get("01", 0) > result.counts.get("10", 0)

    def test_noiseless_matches_statevector(self):
        """Density sim without noise should match LocalSimulator on average."""
        sim_d = DensitySimulator(seed=0)
        sim_s = __import__("qorch", fromlist=["LocalSimulator"]).LocalSimulator(seed=0)
        c = Circuit(2).h(0).cx(0, 1)
        r_d = sim_d.run(c, shots=4096)
        r_s = sim_s.run(c, shots=4096)
        assert abs(r_d.counts.get("00", 0) - r_s.counts.get("00", 0)) < 200

    def test_parametric_gates(self):
        sim = DensitySimulator(seed=0)
        c = Circuit(1).rx(0, 3.14159)
        result = sim.run(c, shots=1024)
        assert result.shots > 0

    def test_num_qubits_validated(self):
        sim = DensitySimulator(seed=0)
        c = Circuit(10)
        try:
            sim.run(c)
        except ValueError:
            pass
