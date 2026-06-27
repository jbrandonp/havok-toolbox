"""Tests for serialize module — .havok save/load roundtrip."""
import pytest, numpy as np, tempfile, os
from havolib.serialize import save_pipeline, load_pipeline

class TestSerialize:
    def test_roundtrip_single_array(self):
        with tempfile.NamedTemporaryFile(suffix='.havok', delete=False) as f:
            tmp = f.name
        try:
            save_pipeline(tmp, "0.7.1", {"tau": 5}, {"forcing": np.array([1.0, 2.0, 3.0])})
            loaded = load_pipeline(tmp)
            assert loaded["version"] == "0.7.1"
            assert loaded["config"]["tau"] == 5
            assert np.allclose(loaded["arrays"]["forcing"], [1.0, 2.0, 3.0])
        finally:
            os.unlink(tmp)

    def test_roundtrip_multiple_arrays(self):
        with tempfile.NamedTemporaryFile(suffix='.havok', delete=False) as f:
            tmp = f.name
        try:
            save_pipeline(tmp, "0.7.1", {"m": 30}, {
                "forcing": np.random.randn(200),
                "risk": np.ones(200, dtype=int),
                "eigen_coords": np.random.randn(200, 5),
            })
            loaded = load_pipeline(tmp)
            assert len(loaded["arrays"]) == 3
            assert loaded["arrays"]["forcing"].shape == (200,)
            assert loaded["arrays"]["risk"].shape == (200,)
            assert loaded["arrays"]["eigen_coords"].shape == (200, 5)
        finally:
            os.unlink(tmp)

    def test_metadata_preserved(self):
        with tempfile.NamedTemporaryFile(suffix='.havok', delete=False) as f:
            tmp = f.name
        try:
            save_pipeline(tmp, "0.7.1", {"tau": 10},
                          {"x": np.ones(5)},
                          metadata={"dataset": "lorenz", "n_points": 5000})
            loaded = load_pipeline(tmp)
            assert loaded["metadata"]["dataset"] == "lorenz"
            assert loaded["metadata"]["n_points"] == 5000
        finally:
            os.unlink(tmp)

    def test_timestamp_is_iso(self):
        with tempfile.NamedTemporaryFile(suffix='.havok', delete=False) as f:
            tmp = f.name
        try:
            save_pipeline(tmp, "0.7.1", {}, {"x": np.ones(3)})
            loaded = load_pipeline(tmp)
            assert "T" in loaded["timestamp"]
        finally:
            os.unlink(tmp)
