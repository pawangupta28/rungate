"""
Rungate — Runtime governance for AI agent tool-calls.

The gate every tool-call passes through.
Intercept, validate, rate-limit, cost-track, kill, log, alert —
across any agent framework, fully self-hosted.

Usage:
    from rungate import Rungate

    gate = Rungate.from_yaml("policy.yaml")
    result = gate.call(agent="researcher", tool="web_search", args={"query": "test"})
"""

__version__ = "0.1.0"
__author__ = "Pawan Gupta"
__license__ = "BUSL-1.1"

from rungate.core.interceptor import Rungate
from rungate.core.policy import PolicyEngine, Policy, PolicyRule
from rungate.core.validator import SchemaValidator
from rungate.core.audit import AuditLogger
from rungate.core.rate_limiter import RateLimiter
from rungate.core.cost_tracker import CostTracker
from rungate.core.kill_switch import KillSwitch
from rungate.core.alerts import AlertEngine

__all__ = [
    "Rungate",
    "PolicyEngine",
    "Policy",
    "PolicyRule",
    "SchemaValidator",
    "AuditLogger",
    "RateLimiter",
    "CostTracker",
    "KillSwitch",
    "AlertEngine",
]
