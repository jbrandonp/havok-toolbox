"""Tests for data_loader.py — CSV loading, Lorenz generation, EEG loading."""
import os
from pathlib import Path
import numpy as np
from havolib.data_loader import (
    load_csv,
    generate_lorenz,
    generate_eeg_like,
    load_eeg,
    download_chb_sample,
    list_edf_channels,
)


class TestGenerateLorenz:
    def test_returns_correct_shapes(self):
        t, x = generate_lorenz(n_points=2000)
        assert len(t) == 2000
        assert len(x) == 2000
        assert t.dtype == np.float64
        assert x.dtype == np.float64

    def test_produces_chaotic_behavior(self):
        """Lorenz attractor should not be constant or degenerate."""
        t, x = generate_lorenz(n_points=5000)
        assert np.std(x) > 1.0, f"Lorenz std too low: {np.std(x):.4f}"
        assert np.max(x) - np.min(x) > 10.0, "Lorenz range too narrow"

    def test_different_seeds(self):
        """Lorenz is deterministic, but we can test with different dt."""
        t1, x1 = generate_lorenz(n_points=1000, dt=0.01)
        t2, x2 = generate_lorenz(n_points=1000, dt=0.02)
        assert not np.allclose(x1, x2)


class TestGenerateEEGLike:
    def test_returns_correct_shapes(self):
        t, x = generate_eeg_like(n_points=1000, fs=256.0)
        assert len(t) == 1000
        assert len(x) == 1000

    def test_sampling_rate(self):
        t, x = generate_eeg_like(n_points=512, fs=256.0)
        assert abs(t[1] - t[0] - 1.0 / 256.0) < 1e-6


class TestLoadCSV:
    def test_load_existing_sample(self):
        csv_path = Path(__file__).parent.parent / "data" / "chb_sample.csv"
        if not csv_path.exists():
            csv_path = Path(__file__).parent.parent / "data" / "chb_sample_synthetic.csv"
        if csv_path.exists():
            data = load_csv(str(csv_path), column="eeg")
            assert isinstance(data, np.ndarray)
            assert len(data) > 0

    def test_load_synthetic_fallback(self):
        import tempfile
        import pandas as pd
        with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
            pd.DataFrame({'value': np.sin(np.linspace(0, 10, 100))}).to_csv(f, index=False)
            tmp_path = f.name
        try:
            data = load_csv(tmp_path, column='value')
            assert len(data) == 100
            assert data.dtype == float
        finally:
            os.unlink(tmp_path)


class TestEDFChannelListing:
    def test_list_channels_on_existing_edf(self):
        edf_dir = Path(__file__).parent.parent / "data" / "chbmit"
        edf_files = [f for f in os.listdir(edf_dir) if f.endswith('.edf')] if edf_dir.exists() else []
        if edf_files:
            edf_path = os.path.join(str(edf_dir), edf_files[0])
            channels = list_edf_channels(edf_path)
            assert len(channels) > 0
            assert all(isinstance(c, str) for c in channels)

    def test_missing_file_returns_empty(self):
        channels = list_edf_channels("/nonexistent/file.edf")
        assert channels == []
