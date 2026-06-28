"""
Unified configuration system for HAVOK.

Uses dataclasses with strict validation, YAML serialization,
and JSON Schema export for external tooling.

Design principles:
- Immutable where possible (frozen dataclasses)
- Every field has a documented default
- Validation on construction, not on use
- Serializable to/from YAML/JSON
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml
import json
import logging

logger = logging.getLogger("havok.config")


# ── Pre-processing config ──────────────────────────────────────

@dataclass(frozen=True)
class PreprocessingConfig:
    """Pre-processing pipeline parameters."""
    interpolate: bool = True
    smooth_method: Optional[str] = "savgol"  # "savgol", "lowpass", or None
    smooth_window: int = 11
    outlier_method: Optional[str] = "iqr"     # "iqr", "zscore", or None
    detrend: bool = False

    def __post_init__(self):
        if self.smooth_method not in ("savgol", "lowpass", None):
            raise ValueError(f"Invalid smooth_method: {self.smooth_method}")
        if self.smooth_window < 3 or self.smooth_window % 2 == 0:
            raise ValueError(f"smooth_window must be odd and >=3, got {self.smooth_window}")
        if self.outlier_method not in ("iqr", "zscore", None):
            raise ValueError(f"Invalid outlier_method: {self.outlier_method}")


# ── HAVOK parameters ───────────────────────────────────────────

@dataclass(frozen=True)
class HavokParams:
    """HAVOK algorithm parameters with validation."""
    tau: int = 1
    m: int = 50
    r: int = 5
    threshold_std: float = 3.0
    window: int = 100

    def __post_init__(self):
        if self.tau < 1:
            raise ValueError(f"tau must be >= 1, got {self.tau}")
        if self.m < 2:
            raise ValueError(f"m must be >= 2, got {self.m}")
        if self.r < 1:
            raise ValueError(f"r must be >= 1, got {self.r}")
        if self.r > self.m:
            raise ValueError(f"r ({self.r}) must be <= m ({self.m})")
        if self.threshold_std <= 0:
            raise ValueError(f"threshold_std must be > 0, got {self.threshold_std}")
        if self.window < 10:
            raise ValueError(f"window must be >= 10, got {self.window}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Pipeline config ────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    """Full HAVOK pipeline configuration."""
    havok: HavokParams = field(default_factory=HavokParams)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    auto_tune: bool = True
    auto_tune_max_lag: int = 100
    auto_tune_max_m: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "havok": self.havok.to_dict(),
            "preprocessing": asdict(self.preprocessing),
            "auto_tune": self.auto_tune,
            "auto_tune_max_lag": self.auto_tune_max_lag,
            "auto_tune_max_m": self.auto_tune_max_m,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        h = d.get("havok", {})
        p = d.get("preprocessing", {})
        return cls(
            havok=HavokParams(**h) if h else HavokParams(),
            preprocessing=PreprocessingConfig(**p) if p else PreprocessingConfig(),
            auto_tune=d.get("auto_tune", True),
            auto_tune_max_lag=d.get("auto_tune_max_lag", 100),
            auto_tune_max_m=d.get("auto_tune_max_m", 50),
        )


# ── Engine config ──────────────────────────────────────────────

@dataclass(frozen=True)
class StreamConfig:
    """Configuration for one streaming input."""
    id: str
    source: str
    havok: HavokParams = field(default_factory=HavokParams)
    buffer_seconds: float = 300.0
    sample_rate: Optional[float] = None
    batch_stride: int = 20

    def __post_init__(self):
        if not self.id:
            raise ValueError("stream id must be non-empty")
        if self.batch_stride < 1:
            raise ValueError(f"batch_stride must be >= 1, got {self.batch_stride}")


@dataclass(frozen=True)
class AlertTargetConfig:
    """Alert delivery target."""
    type: str
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in ("stdout", "telegram", "webhook", "discord"):
            logger.warning(f"Unknown alert target type: {self.type}")


@dataclass(frozen=True)
class EngineConfig:
    """Full HAVOK Engine configuration."""
    streams: List[StreamConfig] = field(default_factory=list)
    alert_targets: Dict[str, AlertTargetConfig] = field(default_factory=dict)
    log_level: str = "INFO"

    def __post_init__(self):
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log_level: {self.log_level}")

    @classmethod
    def from_yaml(cls, path: str) -> "EngineConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineConfig":
        streams = []
        for s in d.get("streams", []):
            h = s.get("havok", {})
            streams.append(StreamConfig(
                id=s["id"],
                source=s["source"],
                havok=HavokParams(**h) if h else HavokParams(),
                buffer_seconds=s.get("buffer_seconds", 300.0),
                sample_rate=s.get("sample_rate"),
                batch_stride=s.get("batch_stride", 20),
            ))
        targets = {
            name: AlertTargetConfig(type=t["type"], config=t.get("config", {}))
            for name, t in d.get("alert_targets", {}).items()
        }
        return cls(streams=streams, alert_targets=targets, log_level=d.get("log_level", "INFO"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "streams": [
                {
                    "id": s.id, "source": s.source,
                    "havok": s.havok.to_dict(),
                    "buffer_seconds": s.buffer_seconds,
                    "batch_stride": s.batch_stride,
                }
                for s in self.streams
            ],
            "alert_targets": {
                name: {"type": t.type, "config": t.config}
                for name, t in self.alert_targets.items()
            },
            "log_level": self.log_level,
        }


# ── Loader: from YAML profile ──────────────────────────────────

def _get_config_path() -> Path:
    """Return path to havok_config.yaml, portable across install modes."""
    try:
        from importlib.resources import files as _res_files
        p = Path(str(_res_files("havolib"))) / "havok_config.yaml"
        if p.exists():
            return p
    except Exception:
        pass
    # Fallback: sibling to havolib/ (editable install / source layout)
    return Path(__file__).resolve().parent.parent / "havok_config.yaml"

_BUILTIN_CONFIG = _get_config_path()


def load_profile(name: str, config_path: Optional[str] = None) -> PipelineConfig:
    """Load a named configuration profile (eeg, finance, climate, lorenz_demo)."""
    path = Path(config_path) if config_path else _BUILTIN_CONFIG
    if not path.exists():
        logger.warning(f"Config not found at {path}, using defaults")
        return PipelineConfig()

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if name not in cfg:
        raise ValueError(f"Unknown profile '{name}'. Available: {list(cfg.keys())}")

    prof = cfg[name]
    h = HavokParams(
        tau=prof.get("tau", 1),
        m=prof.get("m", 50),
        r=prof.get("r", 5),
        threshold_std=prof.get("threshold_std", 3.0),
        window=prof.get("window", 100),
    )
    pp = prof.get("preprocess", {})
    preproc = PreprocessingConfig(
        interpolate=pp.get("interpolate", True),
        smooth_method=pp.get("smooth_method"),
        smooth_window=pp.get("smooth_window", 11),
        outlier_method=pp.get("outlier_method"),
        detrend=pp.get("detrend", False),
    )
    return PipelineConfig(havok=h, preprocessing=preproc)


def list_profiles(config_path: Optional[str] = None) -> List[str]:
    path = Path(config_path) if config_path else _BUILTIN_CONFIG
    if not path.exists():
        return []
    with open(path) as f:
        return list(yaml.safe_load(f).keys())


# ── Legacy compatibility ───────────────────────────────────────

DEFAULT_TAU = 1
DEFAULT_M = 50
DEFAULT_R = 5
DEFAULT_THRESHOLD_STD = 3.0
DEFAULT_WINDOW = 100

# Keep old function for backward compat
def load_config(profile: Optional[str] = None, config_path: Optional[str] = None) -> Dict[str, Any]:
    """Legacy wrapper — use load_profile() for new code."""
    if profile is None:
        return {"tau": 1, "m": 50, "r": 5, "threshold_std": 3.0, "window": 100}
    cfg = load_profile(profile, config_path)
    return {
        "tau": cfg.havok.tau, "m": cfg.havok.m, "r": cfg.havok.r,
        "threshold_std": cfg.havok.threshold_std, "window": cfg.havok.window,
        "preprocess": asdict(cfg.preprocessing),
    }
