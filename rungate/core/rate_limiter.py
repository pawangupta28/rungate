"""
Rungate Rate Limiter — Sliding window rate limiting per agent and tool.

Prevents agents from making too many tool-calls in a given time window.
Supports per-agent, per-tool, and per-agent-tool combination limits.

Config format:
    rate_limits:
      global:
        max_calls: 100
        window_seconds: 60
      per_agent:
        researcher:
          max_calls: 50
          window_seconds: 60
      per_tool:
        execute_code:
          max_calls: 10
          window_seconds: 60
      per_agent_tool:
        researcher:execute_code:
          max_calls: 5
          window_seconds: 60
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RateLimitConfig:
    """Configuration for a single rate limit."""
    max_calls: int = 100
    window_seconds: float = 60.0


class SlidingWindow:
    """Thread-safe sliding window counter."""

    def __init__(self, max_calls: int, window_seconds: float):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._timestamps: List[float] = []
        self._lock = threading.Lock()

    def check(self) -> bool:
        """Check if a new call is allowed (without recording it)."""
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            return len(self._timestamps) < self.max_calls

    def record(self) -> None:
        """Record a new call."""
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            self._timestamps.append(now)

    def check_and_record(self) -> bool:
        """Atomically check and record. Returns True if allowed."""
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if len(self._timestamps) >= self.max_calls:
                return False
            self._timestamps.append(now)
            return True

    def _prune(self, now: float) -> None:
        """Remove timestamps outside the window."""
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    @property
    def current_count(self) -> int:
        with self._lock:
            self._prune(time.monotonic())
            return len(self._timestamps)

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.current_count)


class RateLimiter:
    """
    Sliding window rate limiter for tool-calls.

    Checks limits in order of specificity:
    1. per_agent_tool (most specific)
    2. per_tool
    3. per_agent
    4. global (least specific)

    A call is blocked if ANY applicable limit is exceeded.

    Usage:
        limiter = RateLimiter({
            "global": {"max_calls": 100, "window_seconds": 60},
            "per_tool": {"execute_code": {"max_calls": 10, "window_seconds": 60}},
        })

        if limiter.check("researcher", "execute_code"):
            # Allowed
            limiter.record("researcher", "execute_code")
        else:
            # Rate limited
            pass
    """

    def __init__(self, config: Dict[str, Any]):
        self._global_window: Optional[SlidingWindow] = None
        self._agent_windows: Dict[str, SlidingWindow] = {}
        self._tool_windows: Dict[str, SlidingWindow] = {}
        self._agent_tool_windows: Dict[str, SlidingWindow] = {}
        self._lock = threading.Lock()
        self._config = config
        self._load_config(config)

    def _load_config(self, config: Dict[str, Any]) -> None:
        """Load rate limit configuration."""
        # Global limit
        global_cfg = config.get("global")
        if global_cfg and isinstance(global_cfg, dict):
            self._global_window = SlidingWindow(
                max_calls=global_cfg.get("max_calls", 100),
                window_seconds=global_cfg.get("window_seconds", 60),
            )

        # Per-agent limits
        per_agent = config.get("per_agent", {})
        if isinstance(per_agent, dict):
            for agent, limits in per_agent.items():
                if isinstance(limits, dict):
                    self._agent_windows[agent] = SlidingWindow(
                        max_calls=limits.get("max_calls", 100),
                        window_seconds=limits.get("window_seconds", 60),
                    )

        # Per-tool limits
        per_tool = config.get("per_tool", {})
        if isinstance(per_tool, dict):
            for tool, limits in per_tool.items():
                if isinstance(limits, dict):
                    self._tool_windows[tool] = SlidingWindow(
                        max_calls=limits.get("max_calls", 100),
                        window_seconds=limits.get("window_seconds", 60),
                    )

        # Per-agent-tool limits
        per_agent_tool = config.get("per_agent_tool", {})
        if isinstance(per_agent_tool, dict):
            for key, limits in per_agent_tool.items():
                if isinstance(limits, dict):
                    self._agent_tool_windows[key] = SlidingWindow(
                        max_calls=limits.get("max_calls", 100),
                        window_seconds=limits.get("window_seconds", 60),
                    )

    def check(self, agent: str, tool: str) -> bool:
        """
        Check if a call is allowed under all applicable rate limits.
        Does NOT record the call — call record() separately after execution.
        """
        # Check most specific first
        agent_tool_key = f"{agent}:{tool}"
        if agent_tool_key in self._agent_tool_windows:
            if not self._agent_tool_windows[agent_tool_key].check():
                return False

        if tool in self._tool_windows:
            if not self._tool_windows[tool].check():
                return False

        if agent in self._agent_windows:
            if not self._agent_windows[agent].check():
                return False

        if self._global_window and not self._global_window.check():
            return False

        return True

    def record(self, agent: str, tool: str) -> None:
        """Record a successful call against all applicable windows."""
        agent_tool_key = f"{agent}:{tool}"

        if agent_tool_key in self._agent_tool_windows:
            self._agent_tool_windows[agent_tool_key].record()

        if tool in self._tool_windows:
            self._tool_windows[tool].record()

        if agent in self._agent_windows:
            self._agent_windows[agent].record()

        if self._global_window:
            self._global_window.record()

    def get_status(self, agent: str = "", tool: str = "") -> Dict[str, Any]:
        """Get current rate limit status for debugging."""
        status: Dict[str, Any] = {}

        if self._global_window:
            status["global"] = {
                "remaining": self._global_window.remaining,
                "max": self._global_window.max_calls,
                "window_seconds": self._global_window.window_seconds,
            }

        if agent and agent in self._agent_windows:
            w = self._agent_windows[agent]
            status[f"agent:{agent}"] = {
                "remaining": w.remaining,
                "max": w.max_calls,
            }

        if tool and tool in self._tool_windows:
            w = self._tool_windows[tool]
            status[f"tool:{tool}"] = {
                "remaining": w.remaining,
                "max": w.max_calls,
            }

        key = f"{agent}:{tool}"
        if key in self._agent_tool_windows:
            w = self._agent_tool_windows[key]
            status[f"agent_tool:{key}"] = {
                "remaining": w.remaining,
                "max": w.max_calls,
            }

        return status

    def reset(self, agent: Optional[str] = None, tool: Optional[str] = None) -> None:
        """Reset rate limit counters."""
        if agent is None and tool is None:
            # Reset everything
            if self._global_window:
                self._global_window = SlidingWindow(
                    self._global_window.max_calls,
                    self._global_window.window_seconds,
                )
            self._agent_windows.clear()
            self._tool_windows.clear()
            self._agent_tool_windows.clear()
            self._load_config(self._config)
        elif agent and tool:
            key = f"{agent}:{tool}"
            if key in self._agent_tool_windows:
                w = self._agent_tool_windows[key]
                self._agent_tool_windows[key] = SlidingWindow(w.max_calls, w.window_seconds)

    def __repr__(self) -> str:
        return (
            f"RateLimiter(global={self._global_window is not None}, "
            f"agents={len(self._agent_windows)}, "
            f"tools={len(self._tool_windows)}, "
            f"agent_tools={len(self._agent_tool_windows)})"
        )
