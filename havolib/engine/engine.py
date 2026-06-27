"""
HavokEngine — 24/7 streaming HAVOK orchestration.

Async event loop that ingests from multiple sources (MQTT, WebSocket,
CSV watch, synthetic), runs IncrementalHAVOK per stream, assesses risk,
and dispatches alerts.

Usage:
    engine = HavokEngine(config_path="engine.yaml")
    await engine.start()
"""

import asyncio
import time
import yaml
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
import logging

from .incremental_havok import IncrementalHAVOK
from .risk_engine import RiskEngine, RiskLevel
from .alert_pipeline import AlertPipeline, AlertRule, AlertTarget, AlertLevel

log = logging.getLogger("havok.engine")


@dataclass
class StreamConfig:
    """Configuration for one input stream."""
    id: str
    source: str  # "mqtt://...", "ws://...", "watch://...", "synthetic://lorenz", "synthetic://eeg"
    havok_params: Dict[str, Any] = field(default_factory=dict)
    buffer_seconds: float = 300.0
    sample_rate: Optional[float] = None
    alerts: List[Dict] = field(default_factory=list)


@dataclass
class EngineConfig:
    """Full engine configuration."""
    streams: List[StreamConfig] = field(default_factory=list)
    risk: Dict[str, Any] = field(default_factory=dict)
    alert_targets: Dict[str, Dict] = field(default_factory=dict)
    log_level: str = "INFO"


class HavokEngine:
    """Central orchestrator for streaming HAVOK analysis.

    Manages per-stream IncrementalHAVOK instances, risk assessment,
    and alert dispatch. Designed to run as a long-lived asyncio process.
    """

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        self._streams: Dict[str, IncrementalHAVOK] = {}
        self._configs: Dict[str, StreamConfig] = {}
        self._risk_engine = RiskEngine()
        self._alert_pipeline = AlertPipeline()
        self._tasks: List[asyncio.Task] = []
        self._running = False

        # Load config
        if config_path:
            with open(config_path) as f:
                raw = yaml.safe_load(f)
        elif config_dict:
            raw = config_dict
        else:
            raw = {}

        self._configure(raw)

    def _configure(self, raw: Dict) -> None:
        # Risk engine weights
        risk_cfg = raw.get("risk", {})
        if risk_cfg:
            self._risk_engine = RiskEngine(
                surge_weight=risk_cfg.get("surge_weight", 0.30),
                trend_weight=risk_cfg.get("trend_weight", 0.25),
                cluster_weight=risk_cfg.get("cluster_weight", 0.20),
                significance_weight=risk_cfg.get("significance_weight", 0.25),
            )

        # Alert targets
        for name, tgt in raw.get("alert_targets", {}).items():
            self._alert_pipeline.add_target(name, AlertTarget(
                type=tgt["type"],
                config=tgt.get("config", {}),
            ))

        # Streams
        for stream_raw in raw.get("streams", []):
            cfg = StreamConfig(
                id=stream_raw["id"],
                source=stream_raw["source"],
                havok_params=stream_raw.get("havok", {}),
                buffer_seconds=stream_raw.get("buffer_seconds", 300),
                sample_rate=stream_raw.get("sample_rate"),
                alerts=stream_raw.get("alerts", []),
            )
            self._configs[cfg.id] = cfg

            # Set up alert rules for this stream
            for alert_raw in cfg.alerts:
                rule = AlertRule(
                    stream_id=cfg.id,
                    condition=alert_raw["condition"],
                    level=AlertLevel(alert_raw.get("level", "warning")),
                    cooldown_seconds=alert_raw.get("cooldown", 60),
                    targets=alert_raw.get("targets", []),
                )
                self._alert_pipeline.add_rule(rule)

            # Create HAVOK instance
            params = cfg.havok_params
            self._streams[cfg.id] = IncrementalHAVOK(
                m=params.get("m", 50),
                tau=params.get("tau", 1),
                r=params.get("r", 5),
                threshold_std=params.get("threshold_std", 3.0),
                window=params.get("window", 100),
                batch_stride=params.get("batch_stride", 20),
            )

        log.info(f"Engine configured: {len(self._streams)} streams, {len(self._alert_pipeline._rules)} alert rules")

    async def start(self) -> None:
        """Start all stream ingestion tasks."""
        self._running = True
        for stream_id, cfg in self._configs.items():
            task = asyncio.create_task(self._run_stream(stream_id, cfg))
            self._tasks.append(task)
            log.info(f"Stream '{stream_id}' started ({cfg.source})")

        log.info(f"Engine running with {len(self._tasks)} stream(s)")

    async def stop(self) -> None:
        """Stop all streams gracefully."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        log.info("Engine stopped")

    async def _run_stream(self, stream_id: str, cfg: StreamConfig) -> None:
        """Main loop for one stream."""
        havok = self._streams[stream_id]
        source = cfg.source

        try:
            if source.startswith("synthetic://"):
                await self._run_synthetic(stream_id, havok, source)
            elif source.startswith("csv://"):
                await self._run_csv(stream_id, havok, source)
            elif source.startswith("mqtt://"):
                await self._run_mqtt(stream_id, havok, cfg)
            else:
                log.warning(f"Unknown source type: {source}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error(f"Stream '{stream_id}' crashed: {e}", exc_info=True)

    async def _run_synthetic(self, stream_id: str, havok: IncrementalHAVOK, source: str) -> None:
        """Generate synthetic data for testing."""
        import numpy as np
        from havolib.data_loader import generate_lorenz, generate_eeg_like

        synth_type = source.replace("synthetic://", "")
        if synth_type == "lorenz":
            _, data = generate_lorenz(n_points=100000, dt=0.01)
        else:
            _, data = generate_eeg_like(n_points=60000)

        for i, value in enumerate(data):
            if not self._running:
                break
            forcing, risk = havok.update(value)

            if i % 100 == 0:
                _, level, details = self._risk_engine.assess(havok.get_forcing_history(200))
                await self._alert_pipeline.check(stream_id, risk, details)
                log.debug(f"{stream_id}[{i}] forcing={forcing:.4f} risk={risk:.3f} level={level.value}")

            await asyncio.sleep(0.001)  # ~1000 Hz max for synthetic

    async def _run_csv(self, stream_id: str, havok: IncrementalHAVOK, source: str) -> None:
        """Replay a CSV file."""
        import pandas as pd
        import numpy as np

        path = source.replace("csv://", "")
        col = "eeg" if "eeg" in path.lower() else None
        df = pd.read_csv(path)
        if col and col in df.columns:
            data = df[col].values
        else:
            data = df.iloc[:, -1].values

        for i, value in enumerate(data):
            if not self._running:
                break
            forcing, risk = havok.update(float(value))

            if i % 50 == 0:
                _, level, details = self._risk_engine.assess(havok.get_forcing_history(200))
                await self._alert_pipeline.check(stream_id, risk, details)

            await asyncio.sleep(0.005)

    async def _run_mqtt(self, stream_id: str, havok: IncrementalHAVOK, cfg: StreamConfig) -> None:
        """Ingest from MQTT broker."""
        try:
            import paho.mqtt.client as mqtt
        except ImportError:
            log.error("paho-mqtt not installed. pip install paho-mqtt")
            return

        topic = cfg.source.replace("mqtt://", "").split("/topic/")[-1] if "/topic/" in cfg.source else "#"
        broker = cfg.source.replace("mqtt://", "").split("/")[0] if "mqtt://" in cfg.source else "localhost"

        queue: asyncio.Queue = asyncio.Queue()

        def on_message(client, userdata, msg):
            try:
                value = float(msg.payload.decode())
                asyncio.run_coroutine_threadsafe(queue.put(value), asyncio.get_event_loop())
            except (ValueError, UnicodeDecodeError):
                pass

        client = mqtt.Client()
        client.on_message = on_message
        client.connect(broker, 1883, 60)
        client.subscribe(topic)
        client.loop_start()

        try:
            while self._running:
                try:
                    value = await asyncio.wait_for(queue.get(), timeout=1.0)
                    forcing, risk = havok.update(value)

                    if havok.point_count % 50 == 0:
                        _, level, details = self._risk_engine.assess(havok.get_forcing_history(200))
                        await self._alert_pipeline.check(stream_id, risk, details)

                except asyncio.TimeoutError:
                    continue
        finally:
            client.loop_stop()
            client.disconnect()

    def get_stream_state(self, stream_id: str) -> Optional[Dict]:
        """Get current state of a stream."""
        havok = self._streams.get(stream_id)
        if havok is None:
            return None
        return {
            "points_processed": havok.point_count,
            "latest_forcing": float(havok.get_forcing_history(1)[-1]) if havok.get_forcing_history(1).size > 0 else 0.0,
        }

    def get_all_states(self) -> Dict[str, Dict]:
        return {sid: self.get_stream_state(sid) for sid in self._streams}
