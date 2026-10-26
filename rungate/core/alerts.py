"""
Rungate Alert Engine — Real-time notifications on security events.

Fires alerts when:
- Policy violations occur (blocked tool-calls)
- Budget thresholds are crossed
- Kill switch is activated
- Rate limits are hit
- Validation failures occur (possible injection)

Alert channels:
- Webhook (generic HTTP POST)
- Slack (incoming webhook)
- Email (SMTP)
- Callback (custom Python callable)
- Log (writes to audit logger)

Config format:
    alerts:
      channels:
        - type: webhook
          url: "https://hooks.example.com/rungate"
          events: [policy_violation, budget_exceeded, kill_switch]

        - type: slack
          webhook_url: "https://hooks.slack.com/services/T00/B00/xxx"
          channel: "#agent-alerts"
          events: [kill_switch, budget_exceeded]

        - type: email
          smtp_host: "smtp.example.com"
          smtp_port: 587
          from: "rungate@example.com"
          to: ["ops@example.com"]
          events: [kill_switch]

        - type: log
          events: ["*"]  # Log everything
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# Optional imports for alert channels
try:
    import urllib.request
    import urllib.error
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False

try:
    import smtplib
    from email.mime.text import MIMEText
    _HAS_SMTP = True
except ImportError:
    _HAS_SMTP = False


# Standard alert event types
ALERT_EVENTS = {
    "policy_violation",
    "validation_failure",
    "rate_limited",
    "budget_warning",
    "budget_exceeded",
    "kill_switch",
    "execution_error",
}


@dataclass
class Alert:
    """A single alert event."""
    event_type: str
    agent: str = ""
    tool: str = ""
    message: str = ""
    severity: str = "warning"  # info, warning, critical
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "agent": self.agent,
            "tool": self.tool,
            "message": self.message,
            "severity": self.severity,
            "details": self.details,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class AlertChannel:
    """Base class for alert delivery channels."""

    def __init__(self, config: Dict[str, Any]):
        self._events: set = set(config.get("events", ["*"]))
        self._config = config

    def should_fire(self, event_type: str) -> bool:
        """Check if this channel handles this event type."""
        return "*" in self._events or event_type in self._events

    def send(self, alert: Alert) -> bool:
        """Send an alert. Returns True on success."""
        raise NotImplementedError


class WebhookChannel(AlertChannel):
    """HTTP POST webhook alerts."""

    def send(self, alert: Alert) -> bool:
        if not _HAS_URLLIB:
            return False

        url = self._config.get("url", "")
        if not url:
            return False

        try:
            headers = self._config.get("headers", {})
            headers.setdefault("Content-Type", "application/json")

            payload = json.dumps({
                "source": "rungate",
                "alert": alert.to_dict(),
            }).encode("utf-8")

            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            timeout = self._config.get("timeout", 5)

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status < 400
        except Exception:
            return False


class SlackChannel(AlertChannel):
    """Slack incoming webhook alerts."""

    def send(self, alert: Alert) -> bool:
        if not _HAS_URLLIB:
            return False

        webhook_url = self._config.get("webhook_url", "")
        if not webhook_url:
            return False

        severity_emoji = {
            "info": ":information_source:",
            "warning": ":warning:",
            "critical": ":rotating_light:",
        }
        emoji = severity_emoji.get(alert.severity, ":bell:")

        channel = self._config.get("channel", "")
        text = (
            f"{emoji} *Rungate Alert: {alert.event_type}*\n"
            f"Agent: `{alert.agent}` | Tool: `{alert.tool}`\n"
            f"{alert.message}"
        )

        payload: Dict[str, Any] = {"text": text}
        if channel:
            payload["channel"] = channel

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status < 400
        except Exception:
            return False


class EmailChannel(AlertChannel):
    """Email alerts via SMTP."""

    def send(self, alert: Alert) -> bool:
        if not _HAS_SMTP:
            return False

        host = self._config.get("smtp_host", "localhost")
        port = self._config.get("smtp_port", 587)
        username = self._config.get("smtp_user", "")
        password = self._config.get("smtp_pass", "")
        from_addr = self._config.get("from", "rungate@localhost")
        to_addrs = self._config.get("to", [])
        use_tls = self._config.get("use_tls", True)

        if not to_addrs:
            return False

        subject = f"[Rungate {alert.severity.upper()}] {alert.event_type}"
        body = (
            f"Alert Type: {alert.event_type}\n"
            f"Severity: {alert.severity}\n"
            f"Agent: {alert.agent}\n"
            f"Tool: {alert.tool}\n"
            f"Message: {alert.message}\n"
            f"Time: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.timestamp))}\n"
            f"\nDetails:\n{json.dumps(alert.details, indent=2, default=str)}"
        )

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = from_addr
        msg["To"] = ", ".join(to_addrs)

        try:
            server = smtplib.SMTP(host, port, timeout=10)
            if use_tls:
                server.starttls()
            if username and password:
                server.login(username, password)
            server.sendmail(from_addr, to_addrs, msg.as_string())
            server.quit()
            return True
        except Exception:
            return False


class CallbackChannel(AlertChannel):
    """Custom callback function alerts."""

    def __init__(self, config: Dict[str, Any], callback: Callable[[Alert], None]):
        super().__init__(config)
        self._callback = callback

    def send(self, alert: Alert) -> bool:
        try:
            self._callback(alert)
            return True
        except Exception:
            return False


class LogChannel(AlertChannel):
    """Logs alerts to stderr (always available)."""

    def send(self, alert: Alert) -> bool:
        import sys
        try:
            severity_marker = {
                "info": "INFO",
                "warning": "WARN",
                "critical": "CRIT",
            }.get(alert.severity, "ALERT")

            print(
                f"[rungate-alert] [{severity_marker}] {alert.event_type} "
                f"agent={alert.agent} tool={alert.tool} — {alert.message}",
                file=sys.stderr,
            )
            return True
        except Exception:
            return False


class AlertEngine:
    """
    Dispatches alerts to configured channels.

    Usage:
        engine = AlertEngine({
            "channels": [
                {"type": "log", "events": ["*"]},
                {"type": "slack", "webhook_url": "https://...", "events": ["kill_switch"]},
            ]
        })

        engine.fire("policy_violation", call, "Agent tried to access blocked tool")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._channels: List[AlertChannel] = []
        self._history: List[Alert] = []
        self._lock = threading.Lock()
        self._max_history: int = 1000

        if config:
            self._setup_channels(config)

    def _setup_channels(self, config: Dict[str, Any]) -> None:
        """Set up alert channels from config."""
        channels_config = config.get("channels", [])

        # If no channels configured, default to log
        if not channels_config:
            self._channels.append(LogChannel({"events": ["*"]}))
            return

        for ch_config in channels_config:
            if not isinstance(ch_config, dict):
                continue

            ch_type = ch_config.get("type", "log")

            if ch_type == "webhook":
                self._channels.append(WebhookChannel(ch_config))
            elif ch_type == "slack":
                self._channels.append(SlackChannel(ch_config))
            elif ch_type == "email":
                self._channels.append(EmailChannel(ch_config))
            elif ch_type == "log":
                self._channels.append(LogChannel(ch_config))
            elif ch_type == "callback":
                callback = ch_config.get("callback")
                if callback and callable(callback):
                    self._channels.append(CallbackChannel(ch_config, callback))

    def fire(
        self,
        event_type: str,
        call: Any = None,
        message: str = "",
        severity: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Fire an alert to all applicable channels.

        Args:
            event_type: Type of alert (e.g., "policy_violation", "budget_exceeded")
            call: The ToolCall object (if available) — extracts agent/tool from it
            message: Human-readable alert message
            severity: Override auto-severity (info/warning/critical)
            details: Additional context
        """
        # Auto-determine severity if not provided
        if severity is None:
            severity = self._auto_severity(event_type)

        # Extract agent/tool from call object if available
        agent = ""
        tool = ""
        if call is not None:
            agent = getattr(call, "agent", "")
            tool = getattr(call, "tool", "")

        alert = Alert(
            event_type=event_type,
            agent=agent,
            tool=tool,
            message=message,
            severity=severity,
            details=details or {},
        )

        with self._lock:
            self._history.append(alert)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # Dispatch to channels (fire-and-forget, never block the pipeline)
        for channel in self._channels:
            if channel.should_fire(event_type):
                try:
                    channel.send(alert)
                except Exception:
                    pass  # Alerts should never crash the pipeline

    def _auto_severity(self, event_type: str) -> str:
        """Determine severity based on event type."""
        critical_events = {"kill_switch", "budget_exceeded"}
        warning_events = {"policy_violation", "rate_limited", "budget_warning", "validation_failure"}

        if event_type in critical_events:
            return "critical"
        elif event_type in warning_events:
            return "warning"
        return "info"

    def add_channel(self, channel: AlertChannel) -> None:
        """Add a channel at runtime."""
        self._channels.append(channel)

    def add_callback(
        self,
        callback: Callable[[Alert], None],
        events: Optional[List[str]] = None,
    ) -> None:
        """Convenience: add a callback channel at runtime."""
        config = {"events": events or ["*"]}
        self._channels.append(CallbackChannel(config, callback))

    def recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        with self._lock:
            return [a.to_dict() for a in self._history[-limit:]]

    def alert_count(self, event_type: Optional[str] = None) -> int:
        """Count alerts, optionally filtered by type."""
        with self._lock:
            if event_type:
                return sum(1 for a in self._history if a.event_type == event_type)
            return len(self._history)

    def __repr__(self) -> str:
        return f"AlertEngine(channels={len(self._channels)}, history={len(self._history)})"
