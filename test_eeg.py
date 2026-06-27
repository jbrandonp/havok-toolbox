#!/usr/bin/env python
"""
CHB-MIT style single-channel HAVOK + pre-processing + surrogate test.
Tries real PhysioNet EDF first (chb01_01), falls back to high-fidelity synthetic.

Run:
    python -m pytest tests/ -k test_eeg --tb=line
or from source root:
    python test_eeg.py
(works after `pip install -e .` too)
"""
import os
from pathlib import Path
import numpy as np
from havolib.data_loader import download_chb_sample, load_eeg, load_chb_channel
from havolib.pipeline import HavokPipeline

print("=" * 60)
print("CHB-MIT EEG single-channel regime-shift test (HAVOK)")
print("=" * 60)

# 1. Prepare sample (real download attempt or synthetic)
print("\n[1] Preparing CHB-MIT single-channel sample...")
csv_path = download_chb_sample(dest_dir=None, channel="FP1-F7", duration_sec=180.0)

# 2. Load the signal
print("\n[2] Loading signal...")
x = load_eeg(csv_path, column="eeg")
print(f"    Loaded {len(x)} samples")

# For more control, we can also load directly from EDF if present
edf_dir = str(Path(__file__).parent / "data" / "chbmit")
edf_files = [f for f in os.listdir(edf_dir) if f.endswith('.edf')] if os.path.exists(edf_dir) else []
t = None
if edf_files:
    edf_path = os.path.join(edf_dir, edf_files[0])
    print(f"[INFO] Real EDF found: {edf_path}")
    t_real, x_real, fs = load_chb_channel(edf_path, channel="FP1-F7", duration_sec=180.0)
    print(f"    Using real data: {len(x_real)} pts @ {fs} Hz")
    x = x_real  # prefer real when available
    t = t_real

# 3. Run full pipeline with pre-processing
print("\n[3] Running HAVOK pipeline (preprocess + auto tau/m + fit)...")
p = HavokPipeline(
    do_preprocess=True,
    interpolate=True,
    outlier_method="iqr",
    smooth_method="savgol",
    smooth_window=11,
    detrend=False,
)
if t is None:
    t = np.arange(len(x)) / 256.0
p.auto_fit(t, x)

forcing = p.get_forcing()
print(f"    Max |forcing| = {np.max(np.abs(forcing)):.4f}")

# 4. Surrogate validation (the key "deeper layer" test)
print("\n[4] Statistical validation with phase-randomized surrogates...")
summary = p.validate_with_surrogates(n_surrogates=25)
print(summary)

print("\n" + "=" * 60)
if summary.get("significant_at_alpha"):
    print("✅ SIGNIFICANT regime-shift forcing detected (p < 0.01).")
    print("   This is consistent with a real intermittent driver (e.g. seizure onset).")
else:
    print("   No significant forcing above surrogate threshold.")
print("=" * 60)
print("\n✅ CHB-MIT style test complete.")
print("   Next: drop a real downloaded .edf in data/chbmit/ and re-run for authentic data.")
