"""Multi-model regime detection — pluggable wrappers for HAVOK, SINDy, etc.

Usage:
    from havolib.models import ModelRegistry, BaseRegimeModel
    HavokClass = ModelRegistry.get("havok")
    model = HavokClass(tau=1, m=50, r=5)
"""
from .base import BaseRegimeModel
from .registry import ModelRegistry
from ._utils import is_available

# Load all wrappers so they register themselves
from . import havok_wrapper  # noqa: F401  always available
from . import sindy_wrapper  # noqa: F401  conditional on pysindy

__all__ = ["BaseRegimeModel", "ModelRegistry", "is_available"]
