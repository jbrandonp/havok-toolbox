"""Optional dependency helper for model wrappers.

Usage:
    from ._utils import is_available
    if is_available("pysindy"):
        # register SINDy wrapper
"""
import importlib


def is_available(module_name: str) -> bool:
    """Check whether an optional Python package can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
