"""
HAVOK Engine — real-time streaming regime-shift detection.

Modules:
- RingBuffer: O(1) circular buffer
- IncrementalHankel: O(m) sliding Hankel matrix
- BrandSVD: incremental truncated SVD
- IncrementalHAVOK: streaming forcing extraction
- RiskEngine: multi-dimensional risk assessor
- AlertPipeline: routing + cooldown dispatcher
- HavokEngine: asyncio orchestrator
"""

from .ring_buffer import RingBuffer
from .incremental_hankel import IncrementalHankel
from .brand_svd import BrandSVD
from .incremental_havok import IncrementalHAVOK
from .risk_engine import RiskEngine, RiskLevel
from .alert_pipeline import AlertPipeline, AlertRule, AlertTarget, AlertLevel
from .engine import HavokEngine, StreamConfig, EngineConfig
