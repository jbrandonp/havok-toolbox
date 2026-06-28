# Contributing to HAVOK Toolbox

## Adding a new regime-shift detection model

HAVOK Toolbox uses a **model registry** so any algorithm can be plugged in
without changing the core pipeline.  Here's how to add yours.

### 1. Create a wrapper file

Create `havolib/models/your_model_wrapper.py`:

```python
from typing import Optional
import numpy as np
from .base import BaseRegimeModel
from .registry import ModelRegistry

@ModelRegistry.register("your_model_name")
class YourModelWrapper(BaseRegimeModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Validate kwargs here — raise TypeError for unknown params

    def fit(self, t: Optional[np.ndarray], x: np.ndarray) -> "YourModelWrapper":
        # Train on (t, x)
        return self

    def transform(self, t: Optional[np.ndarray], x: np.ndarray) -> np.ndarray:
        # Return 1D forcing signal of length n
        return np.zeros(len(x))  # replace with real logic
```

### 2. Understand the contract

- `fit(t, x)` — train the model.  `t` can be `None` (regularly sampled).
- `transform(t, x)` — produce a **1D array** (same length as `x`).
  For multivariate input `(n, d)`, reduce to scalar per step (e.g. L2 norm).
- `get_risk()` — inherited from `BaseRegimeModel`.  Default: min-max
  normalisation to [0, 1].  Override for domain-specific scoring.

### 3. Handle optional dependencies

If your model requires an external library, guard the import:

```python
from ._utils import is_available

if is_available("your_library"):
    @ModelRegistry.register("your_model")
    class YourModelWrapper(BaseRegimeModel):
        ...
# If not installed, the model simply won't appear in the registry.
# ModelRegistry.get("your_model") will raise a clear ValueError.
```

### 4. Add an optional dependency

In `pyproject.toml`, add an extra:

```toml
[project.optional-dependencies]
your_model = ["your_library"]
```

### 5. Write tests

Create `tests/test_your_model.py`.  Test with synthetic data (Lorenz, sine):

```python
from havolib.models.registry import ModelRegistry

def test_your_model():
    model = ModelRegistry.get("your_model")(**params)
    t = np.linspace(0, 10, 500)
    x = np.sin(2 * np.pi * t)
    model.fit(t, x)
    forcing = model.transform(t, x)
    assert len(forcing) == len(x)
    risk = model.get_risk()
    assert np.all((risk >= 0) & (risk <= 1))
```

### 6. Update the README

Add your model to the multi-model table in `README.md`.

## Code style

- Follow PEP 8.  Line length ≤ 100 is acceptable.
- Use type hints on all public methods.
- Prefer pure NumPy over Pandas for core algorithms.
- Add a docstring to every public class and function.

## Testing

```bash
pip install havok-toolbox[dev]
pytest tests/ -v
```

All contributions must keep the full test suite green.
