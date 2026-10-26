"""
Rungate Interceptor — The core runtime gate for AI agent tool-calls.

Every tool-call passes through this gate. The interceptor:
1. Captures the call (agent, tool, args, metadata)
2. Runs it through the governance pipeline (policy → validate → rate-limit → cost-check)
3. Either allows execution, blocks it, or triggers the kill switch
4. Logs the result and fires alerts if needed

Supports both sync and async execution.

Usage:
    gate = Rungate.from_yaml("policy.yaml")
    result = gate.call(agent="researcher", tool="web_search", args={"query": "AI safety"})

    # Or wrap any callable
    safe_search = gate.wrap(web_search, tool_name="web_search")
    result = safe_search(query="AI safety")

    # Async support
    result = await gate.acall(agent="coder", tool="execute_code", args={"code": "print(1)"})
"""

from __future__ import annotations

import asyncio
import functools
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from rungate.core.policy import PolicyEngine, PolicyDecision
from rungate.core.validator import SchemaValidator
from rungate.core.rate_limiter import RateLimiter
from rungate.core.cost_tracker import CostTracker
from rungate.core.kill_switch import KillSwitch, KillMode
from rungate.core.audit import AuditLogger, AuditEvent
from rungate.core.alerts import AlertEngine


class CallStatus(str, Enum):
    """Outcome of a tool-call passing through the gate."""
    ALLOWED = "allowed"
    BLOCKED_POLICY = "blocked_policy"
    BLOCKED_VALIDATION = "blocked_validation"
    BLOCKED_RATE_LIMIT = "blocked_rate_limit"
    BLOCKED_BUDGET = "blocked_budget"
    BLOCKED_KILL_SWITCH = "blocked_kill_switch"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class ToolCall:
    """Represents a single tool-call passing through the gate."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent: str = ""
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    status: CallStatus = CallStatus.ALLOWED
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    cost: float = 0.0
    pipeline_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent": self.agent,
            "tool": self.tool,
            "args": self.args,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "result": str(self.result)[:500] if self.result is not None else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "cost": self.cost,
            "pipeline_trace": self.pipeline_trace,
        }


@dataclass
class GateResult:
    """Result returned from the gate after processing a tool-call."""
    allowed: bool
    status: CallStatus
    call: ToolCall
    result: Any = None
    error: Optional[str] = None

    @property
    def blocked(self) -> bool:
        return not self.allowed

    def __repr__(self) -> str:
        if self.allowed:
            return f"GateResult(allowed=True, tool={self.call.tool!r}, status={self.status.value})"
        return f"GateResult(blocked=True, tool={self.call.tool!r}, reason={self.status.value})"


class PipelineStage:
    """A single stage in the governance pipeline."""

    def __init__(self, name: str, handler: Callable, enabled: bool = True):
        self.name = name
        self.handler = handler
        self.enabled = enabled

    def execute(self, call: ToolCall) -> Optional[CallStatus]:
        """Run the stage. Returns a blocking status if the call should be stopped, None if it passes."""
        if not self.enabled:
            call.pipeline_trace.append(f"{self.name}:skipped")
            return None
        result = self.handler(call)
        call.pipeline_trace.append(f"{self.name}:{result or 'passed'}")
        return result

    async def aexecute(self, call: ToolCall) -> Optional[CallStatus]:
        """Async version of execute."""
        if not self.enabled:
            call.pipeline_trace.append(f"{self.name}:skipped")
            return None
        if asyncio.iscoroutinefunction(self.handler):
            result = await self.handler(call)
        else:
            result = self.handler(call)
        call.pipeline_trace.append(f"{self.name}:{result or 'passed'}")
        return result


class Rungate:
    """
    The main runtime gate. Every tool-call passes through here.

    Usage:
        # From YAML config
        gate = Rungate.from_yaml("policy.yaml")

        # Programmatic
        gate = Rungate(
            policies={"default": {"tools": {"web_search": "allow", "execute_code": "block"}}},
            schemas={"web_search": {"type": "object", "properties": {"query": {"type": "string"}}}},
        )

        # Execute a gated call
        result = gate.call(agent="researcher", tool="web_search", args={"query": "test"})
        if result.allowed:
            print(result.result)
        else:
            print(f"Blocked: {result.status}")
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        *,
        policies: Optional[Dict] = None,
        schemas: Optional[Dict] = None,
        rate_limits: Optional[Dict] = None,
        cost_config: Optional[Dict] = None,
        audit_config: Optional[Dict] = None,
        alert_config: Optional[Dict] = None,
        mode: str = "enforce",  # "enforce" | "monitor" | "dry-run"
        api_key: Optional[str] = None,  # Set to switch to SaaS mode
    ):
        # Accept either a single config dict or keyword arguments
        if config is not None:
            policies = policies or config.get("policies", {})
            schemas = schemas or config.get("schemas", {})
            rate_limits = rate_limits or config.get("rate_limits", {})
            cost_config = cost_config or config.get("cost", {})
            audit_config = audit_config or config.get("audit", {})
            alert_config = alert_config or config.get("alerts", {})
            mode = config.get("mode", mode)

        self.mode = mode
        self.api_key = api_key
        self._saas_mode = api_key is not None

        # Initialize all governance modules
        self.policy_engine = PolicyEngine(policies or {})
        self.validator = SchemaValidator(schemas or {})
        self.rate_limiter = RateLimiter(rate_limits or {})
        self.cost_tracker = CostTracker(cost_config or {})
        self.kill_switch = KillSwitch()
        self.audit = AuditLogger(audit_config or {})
        self.alerts = AlertEngine(alert_config or {})

        # Build the pipeline
        self._pipeline: List[PipelineStage] = [
            PipelineStage("kill_switch", self._check_kill_switch),
            PipelineStage("policy", self._check_policy),
            PipelineStage("validation", self._check_validation),
            PipelineStage("rate_limit", self._check_rate_limit),
            PipelineStage("cost", self._check_cost),
        ]

        # Registry of wrapped callables
        self._tool_registry: Dict[str, Callable] = {}

        # Stats
        self._total_calls = 0
        self._blocked_calls = 0
        self._allowed_calls = 0

    @classmethod
    def from_yaml(cls, path: Union[str, Path], **overrides) -> "Rungate":
        """Load configuration from a YAML file."""
        import yaml  # lazy import — yaml is optional for basic usage

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            raise ValueError(f"Invalid config file: expected dict, got {type(config).__name__}")

        kwargs = {
            "policies": config.get("policies", {}),
            "schemas": config.get("schemas", {}),
            "rate_limits": config.get("rate_limits", {}),
            "cost_config": config.get("cost", {}),
            "audit_config": config.get("audit", {}),
            "alert_config": config.get("alerts", {}),
            "mode": config.get("mode", "enforce"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, config: Dict[str, Any], **overrides) -> "Rungate":
        """Load configuration from a dictionary."""
        kwargs = {
            "policies": config.get("policies", {}),
            "schemas": config.get("schemas", {}),
            "rate_limits": config.get("rate_limits", {}),
            "cost_config": config.get("cost", {}),
            "audit_config": config.get("audit", {}),
            "alert_config": config.get("alerts", {}),
            "mode": config.get("mode", "enforce"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    # ── Pipeline checks ──────────────────────────────────────────────

    def _check_kill_switch(self, call: ToolCall) -> Optional[CallStatus]:
        """Check if the kill switch is active for this agent/tool."""
        if self.kill_switch.is_killed(agent=call.agent, tool=call.tool):
            return CallStatus.BLOCKED_KILL_SWITCH
        return None

    def _check_policy(self, call: ToolCall) -> Optional[CallStatus]:
        """Check policy rules."""
        decision = self.policy_engine.evaluate(call.agent, call.tool, call.args)
        if decision == PolicyDecision.BLOCK:
            return CallStatus.BLOCKED_POLICY
        return None

    def _check_validation(self, call: ToolCall) -> Optional[CallStatus]:
        """Validate tool-call arguments against schema."""
        is_valid, error = self.validator.validate(call.tool, call.args)
        if not is_valid:
            call.error = f"Validation failed: {error}"
            return CallStatus.BLOCKED_VALIDATION
        return None

    def _check_rate_limit(self, call: ToolCall) -> Optional[CallStatus]:
        """Check rate limits."""
        allowed = self.rate_limiter.check(call.agent, call.tool)
        if not allowed:
            return CallStatus.BLOCKED_RATE_LIMIT
        return None

    def _check_cost(self, call: ToolCall) -> Optional[CallStatus]:
        """Check cost budget."""
        estimated_cost = self.cost_tracker.estimate(call.tool, call.args)
        call.cost = estimated_cost
        if not self.cost_tracker.can_afford(call.agent, estimated_cost):
            return CallStatus.BLOCKED_BUDGET
        return None

    # ── Core execution ───────────────────────────────────────────────

    def call(
        self,
        agent: str = "default",
        tool: str = "",
        args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        callable: Optional[Callable] = None,
    ) -> GateResult:
        """
        Execute a tool-call through the governance gate.

        Args:
            agent: The agent identifier making the call
            tool: The tool being called
            args: Arguments to the tool
            metadata: Optional metadata (trace_id, session_id, etc.)
            callable: Optional function to execute if the call is allowed

        Returns:
            GateResult with allowed/blocked status and result/error
        """
        call = ToolCall(
            agent=agent,
            tool=tool,
            args=args or {},
            metadata=metadata or {},
        )

        self._total_calls += 1
        start = time.monotonic()

        # Run through pipeline
        blocking_status = self._run_pipeline(call)

        if blocking_status is not None:
            call.status = blocking_status
            call.duration_ms = (time.monotonic() - start) * 1000
            self._blocked_calls += 1

            # In monitor mode, log but don't actually block
            if self.mode == "monitor":
                call.pipeline_trace.append("monitor:would_block")
                # Fall through to execution
            else:
                self._post_call(call)
                return GateResult(
                    allowed=False,
                    status=blocking_status,
                    call=call,
                    error=call.error or f"Blocked by {blocking_status.value}",
                )

        # Execute the actual tool call
        call.status = CallStatus.ALLOWED
        result_value = None
        exec_callable = callable or self._tool_registry.get(tool)

        if exec_callable is not None:
            try:
                result_value = exec_callable(**call.args)
                call.status = CallStatus.EXECUTED
                call.result = result_value
            except Exception as e:
                call.status = CallStatus.FAILED
                call.error = f"{type(e).__name__}: {str(e)}"
                self._post_call(call)
                call.duration_ms = (time.monotonic() - start) * 1000
                return GateResult(
                    allowed=True,
                    status=CallStatus.FAILED,
                    call=call,
                    error=call.error,
                )

        call.duration_ms = (time.monotonic() - start) * 1000
        self._allowed_calls += 1

        # Record cost
        self.cost_tracker.record(call.agent, call.cost)

        # Record rate limit hit
        self.rate_limiter.record(call.agent, call.tool)

        self._post_call(call)

        return GateResult(
            allowed=True,
            status=call.status,
            call=call,
            result=result_value,
        )

    async def acall(
        self,
        agent: str = "default",
        tool: str = "",
        args: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        callable: Optional[Callable] = None,
    ) -> GateResult:
        """Async version of call()."""
        call = ToolCall(
            agent=agent,
            tool=tool,
            args=args or {},
            metadata=metadata or {},
        )

        self._total_calls += 1
        start = time.monotonic()

        # Run through pipeline (async)
        blocking_status = await self._arun_pipeline(call)

        if blocking_status is not None:
            call.status = blocking_status
            call.duration_ms = (time.monotonic() - start) * 1000
            self._blocked_calls += 1

            if self.mode == "monitor":
                call.pipeline_trace.append("monitor:would_block")
            else:
                self._post_call(call)
                return GateResult(
                    allowed=False,
                    status=blocking_status,
                    call=call,
                    error=call.error or f"Blocked by {blocking_status.value}",
                )

        call.status = CallStatus.ALLOWED
        result_value = None
        exec_callable = callable or self._tool_registry.get(tool)

        if exec_callable is not None:
            try:
                if asyncio.iscoroutinefunction(exec_callable):
                    result_value = await exec_callable(**call.args)
                else:
                    result_value = exec_callable(**call.args)
                call.status = CallStatus.EXECUTED
                call.result = result_value
            except Exception as e:
                call.status = CallStatus.FAILED
                call.error = f"{type(e).__name__}: {str(e)}"
                self._post_call(call)
                call.duration_ms = (time.monotonic() - start) * 1000
                return GateResult(
                    allowed=True,
                    status=CallStatus.FAILED,
                    call=call,
                    error=call.error,
                )

        call.duration_ms = (time.monotonic() - start) * 1000
        self._allowed_calls += 1
        self.cost_tracker.record(call.agent, call.cost)
        self.rate_limiter.record(call.agent, call.tool)
        self._post_call(call)

        return GateResult(
            allowed=True,
            status=call.status,
            call=call,
            result=result_value,
        )

    def _run_pipeline(self, call: ToolCall) -> Optional[CallStatus]:
        """Run the call through all pipeline stages."""
        for stage in self._pipeline:
            result = stage.execute(call)
            if result is not None:
                return result
        return None

    async def _arun_pipeline(self, call: ToolCall) -> Optional[CallStatus]:
        """Async pipeline execution."""
        for stage in self._pipeline:
            result = await stage.aexecute(call)
            if result is not None:
                return result
        return None

    def _post_call(self, call: ToolCall) -> None:
        """Post-call processing: audit logging and alerting."""
        # Log to audit
        event = AuditEvent(
            call_id=call.id,
            agent=call.agent,
            tool=call.tool,
            args=call.args,
            status=call.status.value,
            error=call.error,
            duration_ms=call.duration_ms,
            cost=call.cost,
            timestamp=call.timestamp,
            pipeline_trace=call.pipeline_trace,
        )
        self.audit.log(event)

        # Fire alerts if blocked or failed
        if call.status not in (CallStatus.ALLOWED, CallStatus.EXECUTED):
            self.alerts.fire(
                event_type=call.status.value,
                call=call,
                message=call.error or f"Tool-call {call.tool} was {call.status.value}",
            )

        # Check if cost tracker triggers kill switch
        if self.cost_tracker.should_kill(call.agent):
            self.kill_switch.activate(
                mode=KillMode.AGENT,
                targets=[call.agent],
                reason=f"Budget exceeded for agent {call.agent}",
            )
            self.alerts.fire(
                event_type="kill_switch_activated",
                call=call,
                message=f"Kill switch activated: budget exceeded for agent {call.agent}",
            )

    # ── Tool registration ────────────────────────────────────────────

    def register(self, tool_name: str, fn: Callable) -> None:
        """Register a callable as a named tool."""
        self._tool_registry[tool_name] = fn

    def wrap(
        self,
        fn: Callable,
        tool_name: Optional[str] = None,
        agent: str = "default",
    ) -> Callable:
        """
        Wrap a callable so every invocation passes through the gate.

        Usage:
            safe_search = gate.wrap(web_search, tool_name="web_search")
            result = safe_search(query="test")
        """
        name = tool_name or fn.__name__

        import inspect as _inspect
        _sig = _inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            # Bind positional args to named params so the gate sees them
            bound = _sig.bind(*args, **kwargs)
            bound.apply_defaults()
            merged = dict(bound.arguments)
            gate_result = self.call(
                agent=agent,
                tool=name,
                args=merged,
                callable=fn,
            )
            if gate_result.blocked:
                raise ToolCallBlocked(
                    f"Tool-call '{name}' blocked: {gate_result.status.value}",
                    gate_result=gate_result,
                )
            return gate_result.result

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs) -> Any:
            bound = _sig.bind(*args, **kwargs)
            bound.apply_defaults()
            merged = dict(bound.arguments)
            gate_result = await self.acall(
                agent=agent,
                tool=name,
                args=merged,
                callable=fn,
            )
            if gate_result.blocked:
                raise ToolCallBlocked(
                    f"Tool-call '{name}' blocked: {gate_result.status.value}",
                    gate_result=gate_result,
                )
            return gate_result.result

        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return wrapper

    def decorator(
        self,
        tool_name: Optional[str] = None,
        agent: str = "default",
    ) -> Callable:
        """
        Use as a decorator to gate a function.

        Usage:
            @gate.decorator(tool_name="web_search")
            def web_search(query: str) -> str:
                ...
        """
        def _decorator(fn: Callable) -> Callable:
            name = tool_name or fn.__name__
            self.register(name, fn)
            return self.wrap(fn, tool_name=name, agent=agent)
        return _decorator

    # ── Control methods ──────────────────────────────────────────────

    def kill(self, mode: str = "all", targets: Optional[List[str]] = None, reason: str = "") -> None:
        """Activate the kill switch."""
        kill_mode = KillMode(mode)
        self.kill_switch.activate(mode=kill_mode, targets=targets or [], reason=reason)
        self.alerts.fire(
            event_type="kill_switch_manual",
            call=ToolCall(tool="kill_switch"),
            message=f"Kill switch manually activated: mode={mode}, reason={reason}",
        )

    def unkill(self) -> None:
        """Deactivate the kill switch."""
        self.kill_switch.deactivate()

    def stats(self) -> Dict[str, Any]:
        """Return gate statistics."""
        return {
            "total_calls": self._total_calls,
            "allowed_calls": self._allowed_calls,
            "blocked_calls": self._blocked_calls,
            "block_rate": self._blocked_calls / max(self._total_calls, 1),
            "kill_switch_active": self.kill_switch.is_killed(),
            "total_cost": self.cost_tracker.total_cost(),
            "cost_by_agent": self.cost_tracker.cost_by_agent(),
        }

    def __repr__(self) -> str:
        return (
            f"Rungate(mode={self.mode!r}, tools={len(self._tool_registry)}, "
            f"calls={self._total_calls}, blocked={self._blocked_calls})"
        )


class ToolCallBlocked(Exception):
    """Raised when a wrapped tool-call is blocked by the gate."""

    def __init__(self, message: str, gate_result: Optional[GateResult] = None):
        super().__init__(message)
        self.gate_result = gate_result
