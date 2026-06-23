"""Tests for the mitigation pipeline."""

from __future__ import annotations

from qorch import Circuit, LocalSimulator
from qorch.mitigation.pipeline import MitigationPipeline, PipelineResult
from qorch.mitigation.readout import ReadoutMitigator


class TestMitigationPipeline:
    def test_basic_run(self):
        sim = LocalSimulator(seed=0)
        pipeline = MitigationPipeline(backend=sim, shots=512)
        c = Circuit(1).h(0)
        result = pipeline.run(c)
        assert isinstance(result, PipelineResult)
        assert result.shots == 512
        assert "raw" in result.steps_applied

    def test_with_twirling(self):
        sim = LocalSimulator(seed=0)
        pipeline = MitigationPipeline(backend=sim, shots=512, twirling=True)
        c = Circuit(2).h(0).cx(0, 1)
        result = pipeline.run(c)
        assert "twirling" in result.steps_applied

    def test_with_dd(self):
        sim = LocalSimulator(seed=0)
        pipeline = MitigationPipeline(backend=sim, shots=512, dd_sequence="xy4")
        c = Circuit(2).h(0).cx(0, 1)
        result = pipeline.run(c)
        assert "dd(xy4)" in result.steps_applied

    def test_with_readout(self):
        sim = LocalSimulator(seed=0)
        mitigator = ReadoutMitigator.from_calibration_matrix(
            labels=["0", "1"],
            matrix=[[0.95, 0.15], [0.05, 0.85]],
        )
        pipeline = MitigationPipeline(backend=sim, shots=512, readout_mitigator=mitigator)
        c = Circuit(1).h(0)
        result = pipeline.run(c)
        assert "readout" in result.steps_applied

    def test_with_zne(self):
        sim = LocalSimulator(seed=0)
        pipeline = MitigationPipeline(backend=sim, shots=512, zne_scales=(1, 3, 5))
        c = Circuit(1).h(0)
        result = pipeline.run(c)
        assert result.metadata.get("zne_extrapolated")

    def test_all_steps(self):
        sim = LocalSimulator(seed=0)
        pipeline = MitigationPipeline(
            backend=sim, shots=512, twirling=True, dd_sequence="hahn",
        )
        c = Circuit(2).h(0).cx(0, 1)
        result = pipeline.run(c)
        assert "twirling" in result.steps_applied
        assert "dd(hahn)" in result.steps_applied

    def test_counts_in_bounds(self):
        sim = LocalSimulator(seed=0)
        pipeline = MitigationPipeline(backend=sim, shots=1024)
        c = Circuit(1).h(0)
        result = pipeline.run(c)
        total = sum(result.counts.values())
        assert abs(total - 1024) < 0.1
