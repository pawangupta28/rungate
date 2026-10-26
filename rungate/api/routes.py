"""
Rungate API Routes — REST endpoint handlers.

Endpoint map:
    POST /v1/validate     — Validate a tool-call (the core endpoint)
    POST /v1/execute      — Validate AND execute a tool-call
    GET  /v1/audit        — Query audit logs
    GET  /v1/status       — System status (kill switch, rate limits, budgets)
    POST /v1/kill         — Activate kill switch
    POST /v1/unkill       — Deactivate kill switch
    GET  /v1/health       — Health check (no auth required)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from rungate.core.interceptor import Rungate, CallStatus
from rungate.api.auth import Tenant
from rungate.api.middleware import APIMiddleware


class RouteHandler:
    """
    Handles API route logic, separated from the web framework.

    Each method returns (status_code, response_dict). The FastAPI/Flask
    server layer maps these to HTTP responses.
    """

    def __init__(self, gate: Rungate, middleware: APIMiddleware):
        self._gate = gate
        self._middleware = middleware

    def health(self) -> tuple:
        """GET /v1/health — No auth required."""
        return 200, {
            "status": "healthy",
            "version": "0.1.0",
            "timestamp": time.time(),
        }

    def validate(self, tenant: Tenant, body: Dict[str, Any]) -> tuple:
        """
        POST /v1/validate — Validate a tool-call without executing.

        Body:
            {
                "agent": "researcher",
                "tool": "web_search",
                "args": {"query": "test"}
            }

        Returns:
            {
                "allowed": true/false,
                "status": "allowed" | "blocked_policy" | ...,
                "pipeline_trace": [...],
                "error": null | "reason"
            }
        """
        scope_err = self._middleware.check_scope(tenant, "validate")
        if scope_err:
            return 403, scope_err

        agent = body.get("agent", "unknown")
        tool = body.get("tool", "unknown")
        args = body.get("args", {})

        # Run through the pipeline without executing
        result = self._gate.call(
            agent=agent,
            tool=tool,
            args=args,
            func=None,  # No execution — validate only
        )

        return 200, {
            "allowed": result.allowed,
            "status": result.status,
            "pipeline_trace": result.call.pipeline_trace if result.call else [],
            "error": result.error,
            "cost_estimate": result.call.cost if result.call else 0,
        }

    def execute(self, tenant: Tenant, body: Dict[str, Any]) -> tuple:
        """
        POST /v1/execute — Validate and execute a tool-call.

        Body:
            {
                "agent": "researcher",
                "tool": "web_search",
                "args": {"query": "test"},
                "webhook_url": "https://..." (optional, for async result)
            }

        Note: The actual tool function must be registered server-side.
        This endpoint is for SaaS mode where the server manages execution.
        """
        scope_err = self._middleware.check_scope(tenant, "execute")
        if scope_err:
            return 403, scope_err

        agent = body.get("agent", "unknown")
        tool = body.get("tool", "unknown")
        args = body.get("args", {})

        result = self._gate.call(
            agent=agent,
            tool=tool,
            args=args,
        )

        response = {
            "allowed": result.allowed,
            "status": result.status,
            "result": str(result.result) if result.result is not None else None,
            "error": result.error,
            "duration_ms": result.call.duration_ms if result.call else 0,
            "cost": result.call.cost if result.call else 0,
        }

        return 200, response

    def audit_query(self, tenant: Tenant, params: Dict[str, Any]) -> tuple:
        """
        GET /v1/audit — Query audit logs.

        Params:
            agent, tool, status, since (unix timestamp), limit
        """
        scope_err = self._middleware.check_scope(tenant, "audit")
        if scope_err:
            return 403, scope_err

        if not self._gate._audit:
            return 200, {"entries": [], "count": 0}

        entries = self._gate._audit.query(
            agent=params.get("agent"),
            tool=params.get("tool"),
            status=params.get("status"),
            since=params.get("since"),
            limit=int(params.get("limit", 100)),
        )

        return 200, {
            "entries": entries,
            "count": len(entries),
        }

    def status(self, tenant: Tenant) -> tuple:
        """GET /v1/status — System status."""
        scope_err = self._middleware.check_scope(tenant, "config")
        if scope_err:
            return 403, scope_err

        return 200, self._gate.stats()

    def kill(self, tenant: Tenant, body: Dict[str, Any]) -> tuple:
        """
        POST /v1/kill — Activate kill switch.

        Body:
            {
                "mode": "all" | "agent" | "tool",
                "targets": ["agent_name"] (for agent/tool modes),
                "reason": "why"
            }
        """
        scope_err = self._middleware.check_scope(tenant, "config")
        if scope_err:
            return 403, scope_err

        from rungate.core.kill_switch import KillMode

        mode_str = body.get("mode", "all")
        targets = body.get("targets", [])
        reason = body.get("reason", f"API kill by tenant {tenant.id}")

        try:
            mode = KillMode(mode_str)
        except ValueError:
            return 400, {"error": f"Invalid mode: {mode_str}. Use: all, agent, tool"}

        self._gate.kill(mode=mode, targets=targets, reason=reason)

        return 200, {
            "killed": True,
            "mode": mode_str,
            "targets": targets,
            "reason": reason,
        }

    def unkill(self, tenant: Tenant, body: Dict[str, Any]) -> tuple:
        """POST /v1/unkill — Deactivate kill switch."""
        scope_err = self._middleware.check_scope(tenant, "config")
        if scope_err:
            return 403, scope_err

        self._gate.unkill()

        return 200, {"killed": False}
