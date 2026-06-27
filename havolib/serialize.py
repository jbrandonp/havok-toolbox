"""
Model serialization — save/load HAVOK pipeline and predictor state.

Uses joblib for numpy arrays and JSON for metadata.
Format: a .havok file is a gzip'd dict with keys:
  - "version": str
  - "config": dict (pipeline parameters)
  - "arrays": dict of numpy arrays (V, forcing, risk)
  - "timestamp": ISO 8601
"""

from __future__ import annotations
import json
import gzip
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger("havok.serialize")


def save_pipeline(
    path: str,
    version: str,
    config: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a HAVOK pipeline result to a .havok file.

    Args:
        path: output file path (.havok extension recommended)
        version: software version string
        config: pipeline configuration dict
        arrays: dict of named numpy arrays (forcing, risk, V, etc.)
        metadata: optional extra metadata
    """
    output = {
        "version": version,
        "config": config,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "arrays": {},
    }

    for name, arr in arrays.items():
        output["arrays"][name] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "data_b64": _array_to_b64(arr),
        }

    serialized = json.dumps(output, ensure_ascii=False)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(serialized)

    logger.info(f"Pipeline saved to {path} ({len(arrays)} arrays)")


def load_pipeline(path: str) -> Dict[str, Any]:
    """Load a HAVOK pipeline result from a .havok file.

    Returns:
        dict with keys: version, config, timestamp, metadata, arrays (dict of ndarray)
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        raw = json.load(f)

    arrays = {}
    for name, arr_info in raw["arrays"].items():
        arrays[name] = _b64_to_array(
            arr_info["data_b64"],
            arr_info["dtype"],
            tuple(arr_info["shape"]),
        )

    logger.info(f"Pipeline loaded from {path} (v{raw['version']}, {len(arrays)} arrays)")
    return {
        "version": raw["version"],
        "config": raw["config"],
        "timestamp": raw["timestamp"],
        "metadata": raw.get("metadata", {}),
        "arrays": arrays,
    }


def _array_to_b64(arr: np.ndarray) -> str:
    import base64
    return base64.b64encode(arr.tobytes()).decode("ascii")


def _b64_to_array(data_b64: str, dtype: str, shape: tuple) -> np.ndarray:
    import base64
    raw = base64.b64decode(data_b64)
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)
