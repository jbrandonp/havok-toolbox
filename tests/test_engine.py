"""Tests for HAVOK Engine streaming components."""
import numpy as np
import pytest
from havolib.engine.ring_buffer import RingBuffer
from havolib.engine.incremental_hankel import IncrementalHankel
from havolib.engine.brand_svd import BrandSVD
from havolib.engine.incremental_havok import IncrementalHAVOK
from havolib.engine.risk_engine import RiskEngine, RiskLevel
from havolib.engine.alert_pipeline import AlertPipeline, AlertRule, AlertTarget, AlertLevel


# ── RingBuffer ──────────────────────────────────────────────
class TestRingBuffer:
    def test_push_and_view(self):
        buf = RingBuffer(capacity=100)
        for i in range(50):
            buf.push(float(i))
        view = buf.view(10)
        assert len(view) == 10
        assert view[-1] == 49.0
        assert view[0] == 40.0

    def test_wraparound(self):
        buf = RingBuffer(capacity=10)
        for i in range(25):
            buf.push(float(i))
        assert buf.size == 10
        assert buf[-1] == 24.0
        assert buf[0] == 15.0

    def test_push_many(self):
        buf = RingBuffer(capacity=100)
        buf.push_many(np.arange(150, dtype=float))
        assert buf.size == 100
        assert buf[-1] == 149.0
        assert buf[0] == 50.0

    def test_empty_view(self):
        buf = RingBuffer(capacity=10)
        assert len(buf.view()) == 0

    def test_index_error(self):
        buf = RingBuffer(capacity=10)
        with pytest.raises(IndexError):
            _ = buf[0]


# ── IncrementalHankel ───────────────────────────────────────
class TestIncrementalHankel:
    def test_ready_after_enough_points(self):
        h = IncrementalHankel(m=10, tau=1)
        assert not h.ready
        for i in range(10):
            h.update(float(i))
        assert h.ready

    def test_row_values(self):
        h = IncrementalHankel(m=5, tau=1)
        for i in range(10):
            h.update(float(i))
        row = h.latest_row
        assert len(row) == 5
        assert row[0] == 5.0  # oldest in window
        assert row[-1] == 9.0  # newest

    def test_with_tau(self):
        h = IncrementalHankel(m=3, tau=2)
        for i in range(20):
            h.update(float(i))
        row = h.latest_row
        # Stride = m * tau = 6; last values: 14,15,16,17,18,19
        # Row = [buf[0], buf[2], buf[4]] = [14, 16, 18]
        assert row[0] == 14.0
        assert row[1] == 16.0
        assert row[2] == 18.0


# ── BrandSVD ────────────────────────────────────────────────
class TestBrandSVD:
    def test_initialization_with_matrix(self):
        H = np.random.randn(100, 20)
        svd = BrandSVD(m=20, r=5, initial_matrix=H)
        assert svd.V.shape == (20, 5)
        assert svd.s.shape == (5,)

    def test_update_does_not_crash(self):
        H = np.random.randn(100, 20)
        svd = BrandSVD(m=20, r=5, initial_matrix=H)
        for _ in range(50):
            svd.update(np.random.randn(20))
        assert svd.V is not None
        assert svd.s is not None

    def test_recalibration(self):
        H = np.random.randn(200, 20)
        svd = BrandSVD(m=20, r=5, initial_matrix=H)
        svd._updates_since_calibration = 600  # force recal
        assert svd.needs_recalibration()
        svd.recalibrate(H)
        assert not svd.needs_recalibration()


# ── IncrementalHAVOK ────────────────────────────────────────
class TestIncrementalHAVOK:
    def test_lorenz_produces_forcing(self):
        from havolib.data_loader import generate_lorenz
        _, x = generate_lorenz(n_points=4000)

        havok = IncrementalHAVOK(m=30, tau=1, r=5, batch_stride=10)
        forcings = []
        for v in x:
            f, _ = havok.update(float(v))
            forcings.append(f)

        # After warmup, forcing should be non-zero
        nonzero = [f for f in forcings[1000:] if abs(f) > 1e-6]
        assert len(nonzero) > 0, f"No forcing extracted after warmup (nonzero={len(nonzero)}/{len(forcings[1000:])})"

    def test_risk_escalates_on_spike(self):
        """Inject a spike — risk should be non-zero at some point."""
        x = np.sin(np.linspace(0, 20 * np.pi, 3000)) * 0.5
        x[2200:2400] += 10.0  # big spike

        havok = IncrementalHAVOK(m=20, tau=1, r=3, window=50, threshold_std=1.5, batch_stride=10)
        risks = []
        for v in x:
            _, risk = havok.update(float(v))
            risks.append(risk)

        # At least some risk should be non-zero in the spike region
        spike_risks = risks[2200:2400]
        assert len([r for r in spike_risks if r > 0]) > 0, \
            f"No non-zero risk during spike: max={np.max(spike_risks):.3f}"


# ── RiskEngine ──────────────────────────────────────────────
class TestRiskEngine:
    def test_normal_signal(self):
        engine = RiskEngine()
        forcing = np.sin(np.linspace(0, 10 * np.pi, 200)) * 0.1
        score, level, details = engine.assess(forcing)
        assert level == RiskLevel.NORMAL
        assert score < 0.5

    def test_spiky_signal_critical(self):
        engine = RiskEngine()
        forcing = np.random.randn(500) * 0.1
        forcing[200:250] = 5.0  # huge spike cluster
        score, level, details = engine.assess(forcing)
        assert level in (RiskLevel.WARNING, RiskLevel.CRITICAL, RiskLevel.ELEVATED)
        assert score > 0.1

    def test_returns_all_details(self):
        engine = RiskEngine()
        forcing = np.random.randn(200)
        score, level, details = engine.assess(forcing)
        assert set(details.keys()) == {"surge", "trend", "cluster", "significance"}


# ── AlertPipeline ───────────────────────────────────────────
class TestAlertPipeline:
    def test_fires_alert(self):
        import asyncio
        async def _test():
            pipeline = AlertPipeline()
            pipeline.add_target("stdout", AlertTarget(type="stdout"))
            pipeline.add_rule(AlertRule("test", "risk > 0.5", AlertLevel.WARNING, 0.0, ["stdout"]))
            return await pipeline.check("test", 0.8)
        fired = asyncio.run(_test())
        assert len(fired) == 1

    def test_cooldown_prevents_refire(self):
        import asyncio
        async def _test():
            pipeline = AlertPipeline()
            pipeline.add_target("stdout", AlertTarget(type="stdout"))
            pipeline.add_rule(AlertRule("test", "risk > 0.5", AlertLevel.WARNING, 999.0, ["stdout"]))
            await pipeline.check("test", 0.8)
            return await pipeline.check("test", 0.9)
        fired = asyncio.run(_test())
        assert len(fired) == 0  # cooldown active

    def test_condition_not_met_no_alert(self):
        import asyncio
        async def _test():
            pipeline = AlertPipeline()
            pipeline.add_target("stdout", AlertTarget(type="stdout"))
            pipeline.add_rule(AlertRule("test", "risk > 0.9", AlertLevel.CRITICAL, 0.0, ["stdout"]))
            return await pipeline.check("test", 0.3)
        fired = asyncio.run(_test())
        assert len(fired) == 0
