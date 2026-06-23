"""Tests for noise-model builder utilities."""

from __future__ import annotations

from qorch import NoiseChannel
from qorch.backends.simulator import GateNoise, ReadoutNoise


class TestNoiseChannelBuilder:
    def test_from_gate_fidelity_ideal(self):
        nc = NoiseChannel.from_gate_fidelity(1.0)
        assert nc.depolarizing_prob == 0.0
        assert nc.amplitude_damping_gamma == 0.0
        assert nc.phase_damping_lambda == 0.0

    def test_from_gate_fidelity_noisy(self):
        nc = NoiseChannel.from_gate_fidelity(0.99)
        assert abs(nc.depolarizing_prob - 0.01) < 1e-12

    def test_from_gate_fidelity_with_t1(self):
        nc = NoiseChannel.from_gate_fidelity(1.0, t1_us=50.0, t2_us=30.0, gate_time_us=0.5)
        assert nc.depolarizing_prob == 0.0
        assert nc.amplitude_damping_gamma > 0.0
        assert nc.phase_damping_lambda > 0.0


class TestReadoutNoiseBuilder:
    def test_from_readout_fidelity_asymmetric(self):
        rn = ReadoutNoise.from_readout_fidelity(0.95)
        assert rn.p0_given1 > rn.p1_given0
        assert rn.active

    def test_from_readout_fidelity_symmetric(self):
        rn = ReadoutNoise.from_readout_fidelity(0.95, asymmetric=False)
        assert rn.p1_given0 == rn.p0_given1
        assert abs(rn.p1_given0 - 0.025) < 1e-12

    def test_from_readout_fidelity_ideal(self):
        rn = ReadoutNoise.from_readout_fidelity(1.0)
        assert not rn.active


class TestGateNoiseBuilder:
    def test_from_gate_fidelity(self):
        gn = GateNoise.from_gate_fidelity(0.99)
        assert abs(gn.depolarizing_prob - 0.01) < 1e-12
        assert gn.active

    def test_from_gate_fidelity_ideal(self):
        gn = GateNoise.from_gate_fidelity(1.0)
        assert gn.depolarizing_prob == 0.0
        assert not gn.active
