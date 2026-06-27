"""
Polars-accelerated data loading — 10x faster than Pandas for HAVOK workflows.

Usage:
    from havolib.polars_loader import load_csv_fast
    df = load_csv_fast("large_file.csv")  # Polars DataFrame
    x = df["column"].to_numpy()            # Convert to numpy for HAVOK
"""

from __future__ import annotations
import numpy as np
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger("havok.polars_loader")

_POLARS_AVAILABLE = False
try:
    import polars as pl
    _POLARS_AVAILABLE = True
except ImportError:
    pass


def load_csv_fast(
    path: str,
    columns: Optional[List[str]] = None,
    n_rows: Optional[int] = None,
) -> "pl.DataFrame":
    """Load CSV with Polars (10-50x faster than Pandas).

    Falls back to Pandas if Polars is not installed.
    """
    if _POLARS_AVAILABLE:
        return pl.read_csv(path, columns=columns, n_rows=n_rows)
    else:
        logger.info("Polars not installed — falling back to pandas")
        import pandas as pd
        return pd.read_csv(path, usecols=columns, nrows=n_rows)


def load_parquet_fast(path: str, columns: Optional[List[str]] = None) -> "pl.DataFrame":
    """Load Parquet with Polars."""
    if _POLARS_AVAILABLE:
        return pl.read_parquet(path, columns=columns)
    else:
        import pandas as pd
        return pd.read_parquet(path, columns=columns)


def to_numpy(df, column: str) -> np.ndarray:
    """Convert a DataFrame column to numpy array (works with both Polars and Pandas)."""
    if _POLARS_AVAILABLE and isinstance(df, pl.DataFrame):
        return df[column].to_numpy()
    else:
        return df[column].values


def batch_load_csvs(
    paths: List[str],
    column: str,
) -> Tuple[np.ndarray, List[str]]:
    """Load multiple CSV files and stack into a multichannel array.

    Returns:
        X: array of shape (n_samples, n_files)
        labels: list of file stems
    """
    data_list = []
    labels = []
    from pathlib import Path

    for p in paths:
        try:
            df = load_csv_fast(p)
            x = to_numpy(df, column)
            data_list.append(x)
            labels.append(Path(p).stem)
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")

    if not data_list:
        raise ValueError("No files loaded successfully")

    # Pad to common length
    max_len = max(len(x) for x in data_list)
    X = np.zeros((max_len, len(data_list)))
    for i, x in enumerate(data_list):
        X[:len(x), i] = x

    return X, labels
