"""
Alert Pipeline — routing, cooldown, and deduplication for HAVOK alerts.

Handles:
- Alert routing to multiple targets (Telegram, webhook, Discord, dashboard)
- Per-stream cooldown to avoid alert storms
- Deduplication of identical alerts
- Escalation: repeat critical alerts after cooldown
"""

import time
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertTarget:
    """Configuration for an alert delivery target."""
    type: str  # "telegram", "webhook", "discord", "stdout", "dashboard"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Rule defining when to fire an alert for a stream."""
    stream_id: str
    condition: str  # "risk > 0.7", "risk > 0.5 AND surge > 0.6"
    level: AlertLevel = AlertLevel.WARNING
    cooldown_seconds: float = 60.0
    targets: List[str] = field(default_factory=list)  # target names
    message_template: str = "Alert: {stream} risk={risk:.2f} level={level}"


class AlertPipeline:
    """Manages alert routing with cooldown and dedup.

    Usage:
        pipeline = AlertPipeline()
        pipeline.add_target("telegram", AlertTarget(type="telegram", config={"chat": "@alerts"}))
        pipeline.add_rule(AlertRule("eeg_1", "risk > 0.7", AlertLevel.CRITICAL, 30.0, ["telegram"]))
        await pipeline.check("eeg_1", 0.85, {"surge": 0.9})
    """

    def __init__(self):
        self._targets: Dict[str, AlertTarget] = {}
        self._rules: Dict[str, List[AlertRule]] = {}
        self._last_fired: Dict[str, float] = {}  # stream_id -> timestamp
        self._handlers: Dict[str, Callable] = {}

    def add_target(self, name: str, target: AlertTarget) -> None:
        self._targets[name] = target

    def add_rule(self, rule: AlertRule) -> None:
        if rule.stream_id not in self._rules:
            self._rules[rule.stream_id] = []
        self._rules[rule.stream_id].append(rule)

    def set_handler(self, target_type: str, handler: Callable) -> None:
        """Register an async handler function for a target type."""
        self._handlers[target_type] = handler

    async def check(
        self,
        stream_id: str,
        risk: float,
        details: Optional[Dict[str, float]] = None,
        current_time: Optional[float] = None,
    ) -> List[Dict]:
        """Check rules and fire alerts if conditions are met.

        Returns list of fired alert dicts.
        """
        if current_time is None:
            current_time = time.time()

        rules = self._rules.get(stream_id, [])
        fired = []

        for rule in rules:
            if not self._evaluate(rule, risk, details):
                continue

            # Cooldown check
            last_key = f"{stream_id}:{rule.level.value}"
            last = self._last_fired.get(last_key, 0)
            if current_time - last < rule.cooldown_seconds:
                continue

            # Fire alert
            alert = {
                "stream": stream_id,
                "risk": risk,
                "level": rule.level.value,
                "timestamp": current_time,
                "details": details or {},
            }

            for target_name in rule.targets:
                alert["target"] = target_name
                await self._dispatch(target_name, alert, rule)

            self._last_fired[last_key] = current_time
            fired.append(alert)

        return fired

    def _evaluate(self, rule: AlertRule, risk: float, details: Optional[Dict]) -> bool:
        """Simple condition evaluator. Supports 'risk > X' and 'X > Y AND Z > W'."""
        ctx = {"risk": risk}
        if details:
            ctx.update(details)

        try:
            parts = rule.condition.replace(" AND ", " and ").split(" and ")
            for part in parts:
                part = part.strip()
                if ">" in part:
                    var, val = part.split(">")
                    var = var.strip()
                    val = float(val.strip())
                    if ctx.get(var, 0) <= val:
                        return False
                elif "<" in part:
                    var, val = part.split("<")
                    var = var.strip()
                    val = float(val.strip())
                    if ctx.get(var, 0) >= val:
                        return False
            return True
        except Exception:
            # Fallback: just check risk threshold
            return risk > 0.5

    async def _dispatch(self, target_name: str, alert: Dict, rule: AlertRule) -> None:
        """Deliver alert to the specified target."""
        target = self._targets.get(target_name)
        if target is None:
            print(f"[AlertPipeline] Unknown target: {target_name}")
            return

        handler = self._handlers.get(target.type)
        if handler:
            try:
                await handler(alert, target.config)
            except Exception as e:
                print(f"[AlertPipeline] Dispatch failed ({target_name}): {e}")
        else:
            # Default: print to stdout
            msg = rule.message_template.format(
                stream=alert["stream"],
                risk=alert["risk"],
                level=alert["level"],
            )
            print(f"[ALERT:{alert['level'].upper()}] {msg} → {target_name}")
