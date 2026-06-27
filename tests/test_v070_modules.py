"""Tests for v0.7.0 modules: adaptive, attribution, hybrid, federated, arena, multichannel, automl, polars."""
import numpy as np
import pytest

# ── Multichannel ────────────────────────────────────────
class TestMultichannel:
    def test_basic(self):
        from havolib.multichannel import MultichannelHAVOK
        X = np.random.randn(200, 3)
        mh = MultichannelHAVOK(3, tau=1, m=15, r=3)
        r = mh.fit_transform(X, show_progress=False)
        assert r.n_channels == 3
        assert len(r.channels) == 3
        assert r.coupling_matrix.shape == (3, 3)
        assert len(r.joint_forcing) == 200

    def test_summary(self):
        from havolib.multichannel import MultichannelHAVOK
        X = np.random.randn(200, 2)
        mh = MultichannelHAVOK(2, tau=1, m=15, r=3)
        r = mh.fit_transform(X, show_progress=False)
        assert "Multichannel" in r.summary()

# ── Adaptive ────────────────────────────────────────────
class TestAdaptive:
    def test_detects_two_regimes(self):
        from havolib.adaptive import AdaptiveHAVOK
        x = np.concatenate([
            np.sin(np.linspace(0, 10*np.pi, 300)),
            np.sin(np.linspace(0, 10*np.pi, 300)) * 4,
        ])
        adp = AdaptiveHAVOK(min_segment_length=100, detection_method='rolling')
        r = adp.fit_transform(x, show_progress=False)
        assert len(r.segments) >= 1
        assert len(r.full_forcing) == len(x)

    def test_summary_reports_regimes(self):
        from havolib.adaptive import AdaptiveHAVOK
        x = np.concatenate([np.sin(np.linspace(0,10*np.pi,200)),
                            np.sin(np.linspace(0,10*np.pi,200))*3])
        adp = AdaptiveHAVOK(min_segment_length=80, detection_method='rolling')
        r = adp.fit_transform(x, show_progress=False)
        assert "Adaptive HAVOK" in r.summary()

# ── Attribution ─────────────────────────────────────────
class TestAttribution:
    def test_explains_spike(self):
        from havolib.attribution import explain_forcing_spike
        x = np.concatenate([np.zeros(200), np.sin(np.linspace(0,5*np.pi,200))*5])
        result = explain_forcing_spike(x, spike_index=300, tau=1, m=15, r=3)
        assert "cause" in result
        assert "contributions" in result
        assert 0 <= result["confidence"] <= 1

# ── Federated ───────────────────────────────────────────
class TestFederated:
    def test_two_clients(self):
        from havolib.federated import FederatedHAVOK
        fed = FederatedHAVOK()
        fed.add_client("A", np.sin(np.linspace(0,20*np.pi,300)), m=15)
        fed.add_client("B", np.sin(np.linspace(0,20*np.pi,300))*2, m=15)
        model = fed.train(rounds=2, verbose=False)
        assert model.n_clients == 2
        assert model.rounds_trained == 2

# ── Arena ───────────────────────────────────────────────
class TestArena:
    def test_runs_and_ranks(self):
        from havolib.arena import BenchmarkArena
        arena = BenchmarkArena()
        entries = arena.run(methods=['havok_basic', 'rolling_std'], verbose=False)
        assert len(entries) > 0
        lb = arena.print_leaderboard()
        assert "🏆" in lb

# ── AutoML ──────────────────────────────────────────────
class TestAutoML:
    def test_runs_n_trials(self):
        from havolib.automl import auto_optimize
        x = np.sin(np.linspace(0,20*np.pi,200))
        bp = auto_optimize(x, n_trials=3, show_progress=False)
        assert "tau" in bp["best_params"]
        assert bp["n_completed"] > 0

# ── Polars Loader ───────────────────────────────────────
class TestPolarsLoader:
    def test_loads_csv(self):
        from havolib.polars_loader import load_csv_fast
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('v\n1.0\n2.0\n3.0')
            tmp = f.name
        df = load_csv_fast(tmp)
        os.unlink(tmp)
        assert len(df) == 3

# ── Hybrid (skip if no torch) ───────────────────────────
class TestHybrid:
    def test_creates_model(self):
        pytest.importorskip("torch")
        from havolib.hybrid import HavokTransformer
        ht = HavokTransformer(horizon=10, d_model=16, r=3)
        assert ht.horizon == 10
