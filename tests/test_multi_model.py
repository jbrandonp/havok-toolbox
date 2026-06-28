"""Multi-model tests — verify HAVOK wrapper and model registry."""
import numpy as np
import pytest
from havolib.config import HavokParams
from havolib.pipeline import HavokPipeline
from havolib.models.registry import ModelRegistry
from havolib.models.base import BaseRegimeModel


class TestModelRegistry:
    def test_havok_registered(self):
        assert "havok" in ModelRegistry.list_models()

    def test_registry_rejects_unknown(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ModelRegistry.get("nonexistent_model_xyz123")

    def test_registry_lists_models(self):
        models = ModelRegistry.list_models()
        assert isinstance(models, list)
        assert len(models) >= 1


class TestHavokWrapper:
    def test_fit_transform_risk(self):
        np.random.seed(42)
        _, x = __import__("havolib.data_loader", fromlist=["generate_lorenz"]).generate_lorenz(2000)
        model = ModelRegistry.get("havok")(tau=1, m=30, r=5)
        model.fit(None, x)
        forcing = model.transform(None, x)
        risk = model.get_risk()
        assert len(forcing) > 0
        assert len(risk) == len(forcing)
        assert set(np.unique(risk)).issubset({0, 1})

    def test_fit_transform_shortcut(self):
        np.random.seed(42)
        _, x = __import__("havolib.data_loader", fromlist=["generate_lorenz"]).generate_lorenz(1000)
        model = ModelRegistry.get("havok")(m=20, r=3)
        forcing = model.fit_transform(None, x)
        assert len(forcing) > 0

    def test_wrapper_repr(self):
        model = ModelRegistry.get("havok")()
        assert "HavokWrapper" in repr(model)

    def test_sindy_unavailable_raises(self):
        """Requesting sindy without pysindy installed should raise."""
        try:
            import pysindy  # noqa
            pytest.skip("pysindy installed — skipping unavailable test")
        except ImportError:
            with pytest.raises(ValueError, match="sindy"):
                config = HavokParams(model_type="sindy")
                HavokPipeline(config)


class TestBaseRegimeModelABC:
    """Cannot instantiate abstract class directly."""

    def test_abstract_class(self):
        with pytest.raises(TypeError):
            BaseRegimeModel()  # noqa
