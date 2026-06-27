"""Tests for user-facing analysis tools."""
import numpy as np
import tempfile, os, json
from havolib.user import (
    analyze, AnalysisReport, batch_analyze,
    suggest_and_explain, export_csv, export_json, bootstrap_forcing,
)
from havolib.data_loader import generate_lorenz


class TestAnalyze:
    def test_returns_report(self):
        x = np.sin(np.linspace(0, 20*np.pi, 500))
        report = analyze(x, label="test", tau=1, m=20, r=3, show_progress=False)
        assert isinstance(report, AnalysisReport)
        assert report.n_samples == 500
        assert report.max_forcing > 0

    def test_report_summary(self):
        x = np.sin(np.linspace(0, 10*np.pi, 300))
        report = analyze(x, m=15, r=3, show_progress=False)
        s = report.summary()
        assert "HAVOK ANALYSIS REPORT" in s
        assert "FORCING SIGNAL" in s
        assert "EDGE OF CHAOS" in s

    def test_export_csv(self):
        x = np.sin(np.linspace(0, 10*np.pi, 200))
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            tmp = f.name
        report = analyze(x, m=15, r=3, show_progress=False)
        report.export(tmp)
        assert os.path.getsize(tmp) > 0
        os.unlink(tmp)

    def test_export_json(self):
        x = np.sin(np.linspace(0, 10*np.pi, 200))
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            tmp = f.name
        report = analyze(x, m=15, r=3, show_progress=False)
        report.export(tmp)
        with open(tmp) as f:
            data = json.load(f)
        assert "max_forcing" in data
        os.unlink(tmp)

    def test_lorenz_produces_risk(self):
        _, x = generate_lorenz(n_points=3000)
        report = analyze(x, label="lorenz", tau=1, m=50, r=5, show_progress=False)
        assert report.max_forcing > 0.005, f"Max forcing too low: {report.max_forcing:.4f}"
        assert report.edge_score > 0, "Edge score should be positive"


class TestBootstrap:
    def test_bootstrap_returns_ci(self):
        x = np.sin(np.linspace(0, 20*np.pi, 500))
        ci = bootstrap_forcing(x, n_bootstrap=20, m=20, r=3, show_progress=False)
        assert "forcing_mean" in ci
        assert "risk_probability" in ci
        assert len(ci["forcing_mean"]) == len(x)
        # Risk probability should be in [0,1]
        rp = ci["risk_probability"]
        assert np.all((rp >= 0) & (rp <= 1))

    def test_bootstrap_integration(self):
        x = np.sin(np.linspace(0, 10*np.pi, 300))
        report = analyze(x, m=15, r=3, bootstrap_ci=True, n_bootstrap=15, show_progress=False)
        assert report.bootstrap is not None


class TestSuggestAndExplain:
    def test_returns_params(self):
        x = np.sin(np.linspace(0, 20*np.pi, 500))
        result = suggest_and_explain(x)
        assert "tau" in result
        assert "m" in result
        assert "explanation" in result
        assert "quality" in result


class TestBatchAnalyze:
    def test_batch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            # Create 2 CSV files
            for i in range(2):
                df_path = os.path.join(tmp, f"data_{i}.csv")
                import pandas as pd
                x = np.sin(np.linspace(0, 10*np.pi, 200)) + 0.1*np.random.randn(200)
                pd.DataFrame({"value": x}).to_csv(df_path, index=False)

            reports = batch_analyze(
                [os.path.join(tmp, f"data_{i}.csv") for i in range(2)],
                m=15, r=3, show_progress=False)
            assert len(reports) == 2
            for r in reports:
                assert isinstance(r, AnalysisReport)
