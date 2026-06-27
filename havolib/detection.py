import numpy as np
import pandas as pd
from typing import List

def threshold_risk(forcing: np.ndarray,
                   window: int = 100,
                   n_std: float = 3.0) -> np.ndarray:
    """
    Binary risk flag: 1 when |forcing| exceeds rolling n_std * std.
    """
    abs_f = np.abs(forcing)
    rolling_std = pd.Series(abs_f).rolling(window=window, min_periods=max(5, window//5)).std().values
    threshold = n_std * rolling_std
    risk = (abs_f > threshold).astype(int)
    # Fill early NaNs
    risk[:window] = 0
    return risk

def bayesian_changepoint(forcing: np.ndarray, penalty: float = 10.0) -> List[int]:
    """
    Offline change-point detection on the forcing signal using ruptures.
    Returns list of change point indices.
    """
    try:
        import ruptures as rpt
    except ImportError:
        raise ImportError("ruptures package required for bayesian_changepoint. "
                          "pip install ruptures")

    algo = rpt.Pelt(model="rbf", min_size=10, jump=5).fit(forcing.reshape(-1, 1))
    return algo.predict(pen=penalty)
