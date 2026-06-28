"""Model registry — central dispatch for regime-shift detection models.

Usage:
    from havolib.models.registry import ModelRegistry
    MyModel = ModelRegistry.get("havok")  # returns HavokWrapper class
    ModelRegistry.list_models()           # returns ["havok", "sindy", ...]
"""
from typing import Dict, Type
from .base import BaseRegimeModel


class ModelRegistry:
    _models: Dict[str, Type[BaseRegimeModel]] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register a model class under a string name.

        Example:
            @ModelRegistry.register("my_model")
            class MyModel(BaseRegimeModel):
                ...
        """
        def decorator(model_class):
            if not issubclass(model_class, BaseRegimeModel):
                raise TypeError(
                    f"{model_class.__name__} must inherit from BaseRegimeModel"
                )
            cls._models[name] = model_class
            return model_class
        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseRegimeModel]:
        """Return the model class registered under `name`.

        Raises ValueError if the model is not found.
        """
        if name not in cls._models:
            available = ", ".join(sorted(cls._models.keys())) or "none"
            raise ValueError(
                f"Unknown model '{name}'. Available: {available}"
            )
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list:
        """Return a sorted list of all registered model names."""
        return sorted(cls._models.keys())
