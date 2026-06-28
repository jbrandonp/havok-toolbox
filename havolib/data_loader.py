import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List

import requests
from tqdm import tqdm

try:
    import pyedflib
    PYEDFLIB_AVAILABLE = True
except ImportError:
    PYEDFLIB_AVAILABLE = False

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

# Portable data directory: works in both editable and pip-installed modes
def _get_package_data_dir() -> Path:
    """Return the package root directory, portable across install modes."""
    try:
        # Python 3.9+: use importlib.resources for installed packages
        from importlib.resources import files as _res_files
        return Path(str(_res_files("havolib")))
    except Exception:
        pass
    # Fallback: sibling to havolib/ (editable install / source layout)
    return Path(__file__).resolve().parent.parent

_BASE_DIR = _get_package_data_dir()
DEFAULT_DATA_DIR = _BASE_DIR / "data"


def generate_eeg_like(n_points=4000, fs=256.0):
    t = np.arange(n_points) / fs
    x = 0.8 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.3, n_points)
    return t, x


def generate_lorenz(n_points=12000, sigma=10.0, rho=28.0, beta=8.0/3.0, dt=0.01):
    """Generate Lorenz attractor x-component (classic chaotic system)."""
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)
    x[0], y[0], z[0] = 1.0, 1.0, 1.0
    for i in range(1, n_points):
        dx = sigma * (y[i-1] - x[i-1])
        dy = x[i-1] * (rho - z[i-1]) - y[i-1]
        dz = x[i-1] * y[i-1] - beta * z[i-1]
        x[i] = x[i-1] + dt * dx
        y[i] = y[i-1] + dt * dy
        z[i] = z[i-1] + dt * dz
    t = np.arange(n_points) * dt
    return t, x


def load_csv(filepath, column=None):
    """Load a time series column from CSV. Supports path or file-like (Streamlit)."""
    if hasattr(filepath, "read"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath)
    if column is None:
        column = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    series = df[column].values.astype(float)
    return series


def load_eeg(csv_path, column="eeg"):
    """Load EEG signal from prepared CSV (column 'eeg')."""
    return load_csv(csv_path, column=column)


def download_chb_sample(dest_dir=None, channel="FP1-F7", duration_sec=180.0):
    """
    Prepare a CSV sample for EEG testing.
    If real EDF exists in data/chbmit/, convert first channel segment to CSV.
    Otherwise fall back to synthetic.
    Returns path to the CSV.
    """
    if dest_dir is None:
        dest_dir = str(DEFAULT_DATA_DIR)
    os.makedirs(dest_dir, exist_ok=True)
    csv_path = os.path.join(dest_dir, "chb_sample.csv")
    edf_dir = os.path.join(dest_dir, "chbmit")
    edf_files = [f for f in os.listdir(edf_dir) if f.endswith(".edf")] if os.path.exists(edf_dir) else []

    if edf_files:
        edf_path = os.path.join(edf_dir, edf_files[0])
        print(f"[download] Using real EDF: {edf_path}")
        t, signal, fs = load_chb_channel(edf_path, channel=channel, duration_sec=duration_sec)
        df = pd.DataFrame({"time": t, "eeg": signal})
        df.to_csv(csv_path, index=False)
        print(f"[download] Wrote {csv_path} ({len(signal)} samples)")
        return csv_path
    else:
        print("[download] No real EDF found, generating synthetic CSV")
        t, x = generate_eeg_like(int(256 * duration_sec))
        df = pd.DataFrame({"time": t, "eeg": x})
        df.to_csv(csv_path, index=False)
        return csv_path


def load_chb_channel(edf_path, channel="FP1-F7", duration_sec=300.0, start_sec=0.0):
    if not edf_path or not Path(edf_path).exists():
        print("[EEG] synthetic fallback")
        t, x = generate_eeg_like(int(256 * duration_sec))
        return t, x, 256.0
    p = Path(edf_path)
    if PYEDFLIB_AVAILABLE:
        try:
            f = pyedflib.EdfReader(str(p))
            labels = [lab.strip() for lab in f.getSignalLabels()]
            ch_idx = 0
            for i, lab in enumerate(labels):
                if channel.lower() in lab.lower():
                    ch_idx = i
                    break
            fs = f.getSampleFrequency(ch_idx)
            n = min(int(duration_sec * fs), f.getNSamples()[ch_idx])
            signal = f.readSignal(ch_idx, start=0, n=n)
            f.close()
            signal = np.asarray(signal, dtype=float) - np.mean(signal)
            t = np.arange(len(signal)) / fs
            print(f"[EEG] loaded {len(signal)} samples from real EDF")
            return t, signal, float(fs)
        except Exception as e:
            print("pyedflib fail", e)
    t, x = generate_eeg_like(int(256 * duration_sec))
    return t, x, 256.0


def list_edf_channels(edf_path):
    if not PYEDFLIB_AVAILABLE or not Path(edf_path).exists():
        return []
    f = pyedflib.EdfReader(str(edf_path))
    labs = [l.strip() for l in f.getSignalLabels()]
    f.close()
    return labs
