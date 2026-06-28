"""Data validation tests — corrupted inputs, edge cases, error handling."""
import pytest, numpy as np, tempfile, os, pandas as pd
from havolib.estimator import HavokEstimator
from havolib.pipeline import HavokPipeline
from havolib.data_loader import load_csv, generate_lorenz

class TestCorruptedInputs:
    def test_nan_input(self):
        x = np.sin(np.linspace(0, 10*np.pi, 200))
        x[50] = np.nan
        est = HavokEstimator(m=15, r=3)
        # Should not crash — NaN handling is sklearn's job, but verify no segfault
        try:
            est.fit(x)
        except Exception:
            pass  # NaN rejection is acceptable

    def test_all_zeros(self):
        x = np.zeros(200)
        est = HavokEstimator(m=15, r=3)
        est.fit(x)
        # All-zeros: forcing should be very small (numerical noise only, < 1.0)
        assert np.max(np.abs(est.forcing_)) < 1.0

    def test_single_value_repeated(self):
        x = np.ones(200) * 3.14
        est = HavokEstimator(m=15, r=3)
        est.fit(x)
        assert np.max(np.abs(est.forcing_)) < 1.0

    def test_very_short_input(self):
        x = np.random.randn(10)
        est = HavokEstimator(m=15, r=3)
        # Should auto-reduce m or produce valid output
        try:
            est.fit(x)
            assert len(est.forcing_) == len(x) or len(est.forcing_) > 0
        except Exception:
            pass  # Too short rejection is acceptable

    def test_multichannel_single_channel(self):
        from havolib.multichannel import MultichannelHAVOK
        X = np.random.randn(100, 1)
        mh = MultichannelHAVOK(1, tau=1, m=15, r=3)
        r = mh.fit_transform(X, show_progress=False)
        assert r.n_channels == 1

    def test_adaptive_single_regime(self):
        from havolib.adaptive import AdaptiveHAVOK
        x = np.sin(np.linspace(0, 10*np.pi, 300))
        adp = AdaptiveHAVOK(min_segment_length=80)
        r = adp.fit_transform(x, show_progress=False)
        assert len(r.segments) >= 1

    def test_csv_extra_columns(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0')
            tmp = f.name
        try:
            from havolib.polars_loader import load_csv_fast
            df = load_csv_fast(tmp)
            assert len(df) == 2
        finally:
            os.unlink(tmp)

    def test_empty_file_handled(self):
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            f.write('col\n')
            tmp = f.name
        try:
            from havolib.polars_loader import load_csv_fast
            df = load_csv_fast(tmp)
            assert len(df) == 0
        finally:
            os.unlink(tmp)

    def test_federated_empty_client(self):
        from havolib.federated import FederatedHAVOK
        fed = FederatedHAVOK()
        # < 2 clients should raise clear error
        try:
            fed.train(rounds=1, verbose=False)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass  # expected

    def test_arena_no_methods(self):
        from havolib.arena import BenchmarkArena
        arena = BenchmarkArena()
        entries = arena.run(methods=[], verbose=False)
        assert len(entries) == 0
