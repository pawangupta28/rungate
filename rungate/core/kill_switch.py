"""
Rungate Kill Switch — Emergency stop for agent tool-calls.

Three kill modes:
- ALL: Stop all tool-calls from all agents (global emergency stop)
- AGENT: Stop all tool-calls from specific agents
- TOOL: Stop specific tools from being called by any agent

Can be triggered manually or automatically (e.g., by budget exceeded).

Usage:
    ks = KillSwitch()

    # Manual kill
    ks.activate(KillMode.ALL, reason="Emergency stop")

    # Check before executing
    if ks.is_killed("researcher", "web_search"):
        # Blocked by kill switch
        pass

    # Selective kill
    ks.activate(KillMode.AGENT, targets=["rogue_agent"], reason="Budget exceeded")
    ks.activate(KillMode.TOOL, targets=["execute_code"], reason="Security incident")

    # Deactivate
    ks.deactivate()
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class KillMode(str, Enum):
    """Kill switch modes."""
    ALL = "all"          # Kill everything
    AGENT = "agent"      # Kill specific agents
    TOOL = "tool"        # Kill specific tools


@dataclass
class KillEvent:
    """Record of a kill switch activation."""
    mode: KillMode
    targets: List[str]
    reason: str
    activated_at: float = field(default_factory=time.time)
    activated_by: str = "manual"  # "manual", "budget", "policy", "alert"


class KillSwitch:
    """
    Emergency stop mechanism for agent tool-calls.

    Supports three granularity levels:
    - ALL: Global kill — nothing gets through
    - AGENT: Block specific agents — all their tools are stopped
    - TOOL: Block specific tools — no agent can use them

    Multiple kills can be active simultaneously (e.g., one agent killed
    AND one tool killed). A call is blocked if ANY active kill applies.

    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._active: bool = False
        self._mode: Optional[KillMode] = None
        self._killed_agents: Set[str] = set()
        self._killed_tools: Set[str] = set()
        self._history: List[KillEvent] = []
        self._lock = threading.Lock()

    def activate(
        self,
        mode: KillMode = KillMode.ALL,
        targets: Optional[List[str]] = None,
        reason: str = "",
        activated_by: str = "manual",
    ) -> None:
        """
        Activate the kill switch.

        Args:
            mode: Kill granularity (ALL, AGENT, or TOOL)
            targets: List of agent names or tool names to kill (for AGENT/TOOL modes)
            reason: Why the kill switch was activated
            activated_by: Who/what triggered it ("manual", "budget", "policy", "alert")
        """
        with self._lock:
            self._active = True

            if mode == KillMode.ALL:
                self._mode = KillMode.ALL
            elif mode == KillMode.AGENT:
                if targets:
                    self._killed_agents.update(targets)
                self._mode = self._mode or KillMode.AGENT
            elif mode == KillMode.TOOL:
                if targets:
                    self._killed_tools.update(targets)
                self._mode = self._mode or KillMode.TOOL

            # If we have both agent and tool kills, mode is mixed
            if self._killed_agents and self._killed_tools:
                pass  # Keep whatever mode, both sets are checked

            self._history.append(KillEvent(
                mode=mode,
                targets=targets or [],
                reason=reason,
                activated_by=activated_by,
            ))

    def deactivate(
        self,
        mode: Optional[KillMode] = None,
        targets: Optional[List[str]] = None,
    ) -> None:
        """
        Deactivate the kill switch.

        With no args: full deactivation (clears everything).
        With mode + targets: selective deactivation (un-kill specific agents/tools).
        """
        with self._lock:
            if mode is None and targets is None:
                # Full deactivation
                self._active = False
                self._mode = None
                self._killed_agents.clear()
                self._killed_tools.clear()
                return

            if mode == KillMode.AGENT and targets:
                self._killed_agents -= set(targets)
            elif mode == KillMode.TOOL and targets:
                self._killed_tools -= set(targets)
            elif mode == KillMode.ALL:
                self._active = False
                self._mode = None
                self._killed_agents.clear()
                self._killed_tools.clear()
                return

            # Check if anything is still killed
            if not self._killed_agents and not self._killed_tools:
                if self._mode != KillMode.ALL:
                    self._active = False
                    self._mode = None

    def is_killed(self, agent: str = "", tool: str = "") -> bool:
        """
        Check if a call would be killed.

        Returns True if the kill switch would block this agent/tool combination.
        """
        with self._lock:
            if not self._active:
                return False

            # Global kill
            if self._mode == KillMode.ALL:
                return True

            # Agent-specific kill
            if agent and agent in self._killed_agents:
                return True

            # Tool-specific kill
            if tool and tool in self._killed_tools:
                return True

            return False

    def is_agent_killed(self, agent: str) -> bool:
        """Check if a specific agent is killed."""
        with self._lock:
            if not self._active:
                return False
            if self._mode == KillMode.ALL:
                return True
            return agent in self._killed_agents

    def is_tool_killed(self, tool: str) -> bool:
        """Check if a specific tool is killed."""
        with self._lock:
            if not self._active:
                return False
            if self._mode == KillMode.ALL:
                return True
            return tool in self._killed_tools

    @property
    def active(self) -> bool:
        """Whether the kill switch is active."""
        with self._lock:
            return self._active

    @property
    def active_mode(self) -> Optional[KillMode]:
        """Current kill mode (None if inactive)."""
        with self._lock:
            return self._mode if self._active else None

    @property
    def killed_agents(self) -> Set[str]:
        """Set of currently killed agents."""
        with self._lock:
            return set(self._killed_agents)

    @property
    def killed_tools(self) -> Set[str]:
        """Set of currently killed tools."""
        with self._lock:
            return set(self._killed_tools)

    def status(self) -> Dict[str, Any]:
        """Get full kill switch status."""
        with self._lock:
            return {
                "active": self._active,
                "mode": self._mode.value if self._mode else None,
                "killed_agents": sorted(self._killed_agents),
                "killed_tools": sorted(self._killed_tools),
                "history_count": len(self._history),
                "last_activation": (
                    {
                        "mode": self._history[-1].mode.value,
                        "targets": self._history[-1].targets,
                        "reason": self._history[-1].reason,
                        "activated_by": self._history[-1].activated_by,
                        "activated_at": self._history[-1].activated_at,
                    }
                    if self._history
                    else None
                ),
            }

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Get kill switch activation history."""
        with self._lock:
            return [
                {
                    "mode": e.mode.value,
                    "targets": e.targets,
                    "reason": e.reason,
                    "activated_at": e.activated_at,
                    "activated_by": e.activated_by,
                }
                for e in self._history
            ]

    def __repr__(self) -> str:
        status = "ACTIVE" if self._active else "inactive"
        mode = self._mode.value if self._mode else "none"
        return f"KillSwitch({status}, mode={mode})"
