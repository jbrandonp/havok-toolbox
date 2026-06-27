"""
Alert Pipeline — routing, cooldown, and deduplication for HAVOK alerts.

Built-in handlers: stdout (always available).
Extensible: register_handler("webhook", your_async_fn) for webhook/telegram/discord.

Usage:
    pipeline = AlertPipeline()
    pipeline.add_target("console", AlertTarget(type="stdout"))
    pipeline.register_handler("webhook", my_webhook_handler)  # optional
"""

import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio

logger = logging.getLogger("havok.engine.alerts")


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class AlertTarget:
    """Configuration for an alert delivery target."""
    type: str  # "stdout", "webhook", "telegram", "discord", "dashboard"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertRule:
    """Rule that triggers an alert when risk exceeds threshold."""
    name: str
    risk_threshold: float = 0.7
    cooldown_seconds: float = 60.0
    level: AlertLevel = AlertLevel.WARNING
    message_template: str = "[{level}] Stream {stream}: risk={risk:.3f}"


class AlertPipeline:
    """Routes risk scores to alert targets with cooldown and dedup."""

    def __init__(self):
        self._targets: Dict[str, AlertTarget] = {}
        self._handlers: Dict[str, Callable] = {}
        self._last_fired: Dict[str, float] = {}
        self._rules: Dict[str, AlertRule] = {}
        self._register_builtins()

    def add_target(self, name: str, target: AlertTarget) -> None:
        self._targets[name] = target

    def register_handler(self, target_type: str, handler: Callable) -> None:
        """Register an async handler for a target type.

        Handler signature: async def handler(alert: dict, config: dict) -> None
        """
        self._handlers[target_type] = handler

    def _register_builtins(self) -> None:
        """Auto-register stdout handler. Other targets need user handlers."""
        self._handlers["stdout"] = self._deliver_stdout

    async def _deliver_stdout(self, alert: Dict, config: Dict) -> None:
        level = alert.get("level", "?")
        msg = alert.get("message", "")[:200]
        logger.info(f"ALERT [{level}] {msg}")

    def add_rule(self, name_or_rule, rule: AlertRule = None) -> None:
        """Add an alert rule. Accepts add_rule(name, rule) or add_rule(rule)."""
        if rule is None and isinstance(name_or_rule, AlertRule):
            rule = name_or_rule
            name = rule.name
        elif rule is not None:
            name = name_or_rule
        else:
            raise TypeError("add_rule requires (name, rule) or (rule,)")
        self._rules[name] = rule

    async def evaluate(self, stream_id: str, risk: float) -> List[Dict]:
        """Evaluate risk against all rules, fire alerts if thresholds exceeded."""
        fired = []
        for rule_name, rule in self._rules.items():
            if risk >= rule.risk_threshold:
                cooldown_key = f"{stream_id}:{rule.name}"
                now = time.time()
                last = self._last_fired.get(cooldown_key, 0)
                if now - last < rule.cooldown_seconds:
                    continue
                self._last_fired[cooldown_key] = now

                alert = {
                    "stream": stream_id,
                    "risk": risk,
                    "level": rule.level.value,
                    "timestamp": now,
                    "message": rule.message_template.format(
                        stream=stream_id, risk=risk, level=rule.level.value),
                }

                for target_name, target in self._targets.items():
                    await self._dispatch(target_name, alert, rule)

                fired.append(alert)
        return fired

    async def _dispatch(self, target_name: str, alert: Dict, rule: AlertRule) -> None:
        """Deliver alert to the specified target."""
        target = self._targets.get(target_name)
        if target is None:
            logger.warning(f"Unknown alert target: {target_name}")
            return

        # Try registered handler, fall back to stdout
        handler_key = target.type
        handler = self._handlers.get(handler_key, self._handlers.get("stdout"))

        if handler:
            try:
                await handler(alert, target.config)
            except Exception as e:
                logger.error(f"Alert dispatch failed ({target_name}): {e}")
        else:
            logger.warning(f"No handler for target type '{target.type}' — alert dropped")
