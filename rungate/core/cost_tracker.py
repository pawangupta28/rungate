"""
Rungate Cost Tracker — Per-call cost estimation and budget enforcement.

Tracks estimated costs of tool-calls and enforces budget caps per agent.
When a budget is exceeded, can trigger the kill switch.

Config format:
    cost:
      default_cost: 0.001          # Default cost per call if not specified
      tool_costs:
        web_search: 0.01
        execute_code: 0.05
        send_email: 0.001
        gpt4_call: 0.03
      budgets:
        global: 100.0              # Total budget across all agents
        per_agent:
          researcher: 10.0
          code_agent: 20.0
      kill_on_budget: true         # Trigger kill switch when budget exceeded
      warn_at_percent: 80          # Alert at 80% budget usage
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CostEntry:
    """A single cost record."""
    agent: str
    tool: str
    cost: float
    timestamp: float = field(default_factory=time.time)


class CostTracker:
    """
    Tracks tool-call costs and enforces budgets.

    Usage:
        tracker = CostTracker({
            "tool_costs": {"web_search": 0.01, "gpt4_call": 0.03},
            "budgets": {"global": 100.0, "per_agent": {"researcher": 10.0}},
        })

        cost = tracker.estimate("web_search", {})
        # → 0.01

        if tracker.can_afford("researcher", cost):
            tracker.record("researcher", cost)
    """

    def __init__(self, config: Dict[str, Any]):
        self._tool_costs: Dict[str, float] = {}
        self._default_cost: float = 0.001
        self._global_budget: Optional[float] = None
        self._agent_budgets: Dict[str, float] = {}
        self._kill_on_budget: bool = True
        self._warn_at_percent: float = 80.0

        # Running totals
        self._global_spent: float = 0.0
        self._agent_spent: Dict[str, float] = defaultdict(float)
        self._history: List[CostEntry] = []
        self._lock = threading.Lock()

        self._load_config(config)

    def _load_config(self, config: Dict[str, Any]) -> None:
        """Load cost configuration."""
        self._default_cost = config.get("default_cost", 0.001)
        self._tool_costs = config.get("tool_costs", {})
        self._kill_on_budget = config.get("kill_on_budget", True)
        self._warn_at_percent = config.get("warn_at_percent", 80.0)

        budgets = config.get("budgets", {})
        if isinstance(budgets, dict):
            self._global_budget = budgets.get("global")
            per_agent = budgets.get("per_agent", {})
            if isinstance(per_agent, dict):
                self._agent_budgets = per_agent

    def estimate(self, tool: str, args: Dict[str, Any]) -> float:
        """Estimate the cost of a tool-call."""
        base_cost = self._tool_costs.get(tool, self._default_cost)

        # Could add arg-based cost modifiers here
        # e.g., longer queries cost more, larger file operations cost more
        return base_cost

    def can_afford(self, agent: str, cost: float) -> bool:
        """Check if the agent has enough budget for the estimated cost."""
        with self._lock:
            # Check agent budget
            if agent in self._agent_budgets:
                agent_remaining = self._agent_budgets[agent] - self._agent_spent[agent]
                if cost > agent_remaining:
                    return False

            # Check global budget
            if self._global_budget is not None:
                global_remaining = self._global_budget - self._global_spent
                if cost > global_remaining:
                    return False

        return True

    def record(self, agent: str, cost: float) -> None:
        """Record a cost after execution."""
        with self._lock:
            self._global_spent += cost
            self._agent_spent[agent] += cost
            self._history.append(CostEntry(
                agent=agent,
                tool="",  # Tool name not passed here, tracked in audit
                cost=cost,
            ))

    def should_kill(self, agent: str) -> bool:
        """Check if budget is exceeded and kill switch should be triggered."""
        if not self._kill_on_budget:
            return False

        with self._lock:
            # Check agent budget
            if agent in self._agent_budgets:
                if self._agent_spent[agent] >= self._agent_budgets[agent]:
                    return True

            # Check global budget
            if self._global_budget is not None:
                if self._global_spent >= self._global_budget:
                    return True

        return False

    def should_warn(self, agent: str) -> bool:
        """Check if spending is approaching budget (for alerting)."""
        threshold = self._warn_at_percent / 100.0

        with self._lock:
            if agent in self._agent_budgets and self._agent_budgets[agent] > 0:
                usage = self._agent_spent[agent] / self._agent_budgets[agent]
                if usage >= threshold:
                    return True

            if self._global_budget and self._global_budget > 0:
                usage = self._global_spent / self._global_budget
                if usage >= threshold:
                    return True

        return False

    def total_cost(self) -> float:
        """Get total cost across all agents."""
        with self._lock:
            return self._global_spent

    def cost_by_agent(self) -> Dict[str, float]:
        """Get cost breakdown by agent."""
        with self._lock:
            return dict(self._agent_spent)

    def budget_status(self, agent: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed budget status."""
        with self._lock:
            status: Dict[str, Any] = {
                "global": {
                    "budget": self._global_budget,
                    "spent": self._global_spent,
                    "remaining": (self._global_budget - self._global_spent) if self._global_budget else None,
                    "percent_used": (self._global_spent / self._global_budget * 100) if self._global_budget else 0,
                },
            }

            if agent:
                budget = self._agent_budgets.get(agent)
                spent = self._agent_spent[agent]
                status["agent"] = {
                    "name": agent,
                    "budget": budget,
                    "spent": spent,
                    "remaining": (budget - spent) if budget else None,
                    "percent_used": (spent / budget * 100) if budget else 0,
                }

            return status

    def set_budget(self, agent: str, budget: float) -> None:
        """Set or update an agent's budget at runtime."""
        with self._lock:
            self._agent_budgets[agent] = budget

    def set_global_budget(self, budget: float) -> None:
        """Set or update the global budget."""
        with self._lock:
            self._global_budget = budget

    def add_tool_cost(self, tool: str, cost: float) -> None:
        """Set the cost for a tool."""
        self._tool_costs[tool] = cost

    def reset(self, agent: Optional[str] = None) -> None:
        """Reset spending counters."""
        with self._lock:
            if agent:
                self._agent_spent[agent] = 0.0
            else:
                self._global_spent = 0.0
                self._agent_spent.clear()
                self._history.clear()

    def __repr__(self) -> str:
        return (
            f"CostTracker(global_spent={self._global_spent:.4f}, "
            f"global_budget={self._global_budget}, "
            f"agents={len(self._agent_spent)})"
        )
