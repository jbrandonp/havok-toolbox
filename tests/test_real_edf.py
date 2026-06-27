"""Test real EDF loading robustness (task 1)."""
import os
from pathlib import Path
from havolib.data_loader import load_chb_channel, list_edf_channels, generate_eeg_like

def test_edf_fallback_synthetic():
    t, x, fs = load_chb_channel(None, duration_sec=10)
    assert len(x) > 100
    assert fs == 256.0

def test_edf_real_if_present():
    edf = Path("data/chbmit/chb01_01.edf")
    if edf.exists():
        channels = list_edf_channels(str(edf))
        assert len(channels) > 0
        t, x, fs = load_chb_channel(str(edf), "FP1-F7", duration_sec=5)
        assert len(x) > 100
        assert fs > 200
    else:
        print("No real EDF present — test skipped (synthetic fallback works)")

def test_path_robustness():
    # Should not crash on bad path
    t, x, fs = load_chb_channel("nonexistent\\bad\\path.edf", duration_sec=5)
    assert len(x) > 0

if __name__ == "__main__":
    test_edf_fallback_synthetic()
    test_edf_real_if_present()
    test_path_robustness()
    print("test_real_edf.py PASSED")