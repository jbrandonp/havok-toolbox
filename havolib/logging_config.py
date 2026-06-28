"""
Structured logging setup for HAVOK.

Replaces all print() calls with proper logging.
Call init_logging() once at startup to configure.
"""

import logging
import sys
from typing import Optional


def init_logging(
    level: str = "INFO",
    format_str: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Initialize structured logging for HAVOK.

    Args:
        level: "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL"
        format_str: custom format (default: timestamp + level + module + message)
        log_file: optional file path for persistent logs

    Returns:
        Root HAVOK logger
    """
    if format_str is None:
        format_str = "%(asctime)s [%(levelname)-7s] %(name)s | %(message)s"

    root = logging.getLogger("havok")
    root.setLevel(getattr(logging, level.upper()))

    # Only add handler if none exists (avoid duplicates)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(format_str, datefmt="%H:%M:%S"))
        root.addHandler(handler)

        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S"))
            root.addHandler(fh)

    # Silence noisy external libs
    for lib in ("matplotlib", "PIL", "asyncio"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    return root


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the havok namespace."""
    return logging.getLogger(f"havok.{name}")
