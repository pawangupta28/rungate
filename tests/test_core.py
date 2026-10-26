"""
Tests for Rungate core modules.

Run with: python -m pytest tests/ -v
"""

import asyncio
import time
import pytest

from rungate.core.interceptor import Rungate, GateResult, CallStatus, ToolCallBlocked
from rungate.core.policy import PolicyEngine, PolicyDecision
from rungate.core.validator import SchemaValidator
from rungate.core.rate_limiter import RateLimiter, SlidingWindow
from rungate.core.cost_tracker import CostTracker
from rungate.core.kill_switch import KillSwitch, KillMode
from rungate.core.alerts import AlertEngine, Alert
from rungate.core.audit import AuditLogger, AuditEvent


# ─── Policy Engine Tests ───────────────────────────────────────────

class TestPolicyEngine:
    def test_deny_by_default(self):
        engine = PolicyEngine({
            "default": {"mode": "deny", "tools": {"web_search": "allow"}}
        })
        assert engine.evaluate("agent1", "web_search", {}) == PolicyDecision.ALLOW
        assert engine.evaluate("agent1", "delete_file", {}) == PolicyDecision.BLOCK

    def test_allow_by_default(self):
        engine = PolicyEngine({
            "researcher": {"mode": "allow", "tools": {"delete_file": "block"}}
        })
        assert engine.evaluate("researcher", "web_search", {}) == PolicyDecision.ALLOW
        assert engine.evaluate("researcher", "delete_file", {}) == PolicyDecision.BLOCK

    def test_conditions(self):
        engine = PolicyEngine({
            "default": {
                "mode": "allow",
                "conditions": [
                    {
                        "tool": "web_search",
                        "block_if": {"query_contains": ["password", "secret"]},
                    }
                ],
            }
        })
        assert engine.evaluate("a", "web_search", {"query": "AI safety"}) == PolicyDecision.ALLOW
        assert engine.evaluate("a", "web_search", {"query": "get password"}) == PolicyDecision.BLOCK

    def test_inheritance(self):
        engine = PolicyEngine({
            "default": {"mode": "deny", "tools": {"web_search": "allow"}},
            "code_agent": {"inherits": "default", "tools": {"execute_code": "allow"}},
        })
        assert engine.evaluate("code_agent", "web_search", {}) == PolicyDecision.ALLOW
        assert engine.evaluate("code_agent", "execute_code", {}) == PolicyDecision.ALLOW
        assert engine.evaluate("code_agent", "delete_file", {}) == PolicyDecision.BLOCK

    def test_fallback_to_default(self):
        engine = PolicyEngine({
            "default": {"mode": "deny", "tools": {"web_search": "allow"}}
        })
        # Unknown agent falls back to default policy
        assert engine.evaluate("unknown_agent", "web_search", {}) == PolicyDecision.ALLOW
        assert engine.evaluate("unknown_agent", "delete_file", {}) == PolicyDecision.BLOCK

    def test_runtime_add_remove(self):
        engine = PolicyEngine({"default": {"mode": "deny"}})
        engine.add_policy("new_agent", {"mode": "allow"})
        assert engine.evaluate("new_agent", "anything", {}) == PolicyDecision.ALLOW
        engine.remove_policy("new_agent")
        assert engine.evaluate("new_agent", "anything", {}) == PolicyDecision.BLOCK


# ─── Schema Validator Tests ────────────────────────────────────────

class TestSchemaValidator:
    def test_valid_input(self):
        validator = SchemaValidator({
            "search": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string", "minLength": 1}},
            }
        })
        ok, err = validator.validate("search", {"query": "test"})
        assert ok is True
        assert err is None

    def test_missing_required(self):
        validator = SchemaValidator({
            "search": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            }
        })
        ok, err = validator.validate("search", {})
        assert ok is False
        assert "required" in err.lower()

    def test_type_mismatch(self):
        validator = SchemaValidator({
            "tool": {
                "type": "object",
                "properties": {"count": {"type": "integer"}},
            }
        })
        ok, err = validator.validate("tool", {"count": "not_a_number"})
        assert ok is False

    def test_string_constraints(self):
        validator = SchemaValidator({
            "tool": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "minLength": 2, "maxLength": 10}
                },
            }
        })
        ok, _ = validator.validate("tool", {"name": "ab"})
        assert ok is True
        ok, err = validator.validate("tool", {"name": "a"})
        assert ok is False

    def test_no_schema_passes(self):
        validator = SchemaValidator({})
        ok, err = validator.validate("unknown_tool", {"anything": "goes"})
        assert ok is True

    def test_enum_validation(self):
        validator = SchemaValidator({
            "tool": {
                "type": "object",
                "properties": {
                    "lang": {"type": "string", "enum": ["python", "javascript"]}
                },
            }
        })
        ok, _ = validator.validate("tool", {"lang": "python"})
        assert ok is True
        ok, _ = validator.validate("tool", {"lang": "ruby"})
        assert ok is False


# ─── Rate Limiter Tests ────────────────────────────────────────────

class TestRateLimiter:
    def test_global_limit(self):
        limiter = RateLimiter({
            "global": {"max_calls": 3, "window_seconds": 60}
        })
        assert limiter.check("a", "t") is True
        limiter.record("a", "t")
        limiter.record("a", "t")
        limiter.record("a", "t")
        assert limiter.check("a", "t") is False

    def test_per_agent_limit(self):
        limiter = RateLimiter({
            "per_agent": {"agent1": {"max_calls": 2, "window_seconds": 60}}
        })
        limiter.record("agent1", "tool1")
        limiter.record("agent1", "tool1")
        assert limiter.check("agent1", "tool1") is False
        # Different agent is not limited
        assert limiter.check("agent2", "tool1") is True

    def test_per_tool_limit(self):
        limiter = RateLimiter({
            "per_tool": {"dangerous": {"max_calls": 1, "window_seconds": 60}}
        })
        limiter.record("a", "dangerous")
        assert limiter.check("a", "dangerous") is False
        assert limiter.check("a", "safe") is True

    def test_sliding_window(self):
        window = SlidingWindow(max_calls=2, window_seconds=0.1)
        assert window.check_and_record() is True
        assert window.check_and_record() is True
        assert window.check_and_record() is False
        # Wait for window to expire
        time.sleep(0.15)
        assert window.check_and_record() is True


# ─── Cost Tracker Tests ───────────────────────────────────────────

class TestCostTracker:
    def test_estimate(self):
        tracker = CostTracker({
            "tool_costs": {"search": 0.01, "code": 0.05},
            "default_cost": 0.001,
        })
        assert tracker.estimate("search", {}) == 0.01
        assert tracker.estimate("code", {}) == 0.05
        assert tracker.estimate("unknown", {}) == 0.001

    def test_budget_enforcement(self):
        tracker = CostTracker({
            "budgets": {"per_agent": {"agent1": 0.05}},
        })
        assert tracker.can_afford("agent1", 0.03) is True
        tracker.record("agent1", 0.03)
        assert tracker.can_afford("agent1", 0.03) is False
        assert tracker.can_afford("agent1", 0.02) is True

    def test_global_budget(self):
        tracker = CostTracker({
            "budgets": {"global": 0.1},
        })
        tracker.record("a", 0.06)
        tracker.record("b", 0.04)
        assert tracker.can_afford("c", 0.01) is False

    def test_kill_trigger(self):
        tracker = CostTracker({
            "budgets": {"per_agent": {"agent1": 0.05}},
            "kill_on_budget": True,
        })
        tracker.record("agent1", 0.06)
        assert tracker.should_kill("agent1") is True
        assert tracker.should_kill("agent2") is False

    def test_cost_reporting(self):
        tracker = CostTracker({"budgets": {}})
        tracker.record("a", 0.1)
        tracker.record("b", 0.2)
        assert tracker.total_cost() == pytest.approx(0.3)
        by_agent = tracker.cost_by_agent()
        assert by_agent["a"] == pytest.approx(0.1)
        assert by_agent["b"] == pytest.approx(0.2)


# ─── Kill Switch Tests ────────────────────────────────────────────

class TestKillSwitch:
    def test_global_kill(self):
        ks = KillSwitch()
        assert ks.is_killed("any", "any") is False
        ks.activate(KillMode.ALL, reason="test")
        assert ks.is_killed("any", "any") is True
        ks.deactivate()
        assert ks.is_killed("any", "any") is False

    def test_agent_kill(self):
        ks = KillSwitch()
        ks.activate(KillMode.AGENT, targets=["bad_agent"], reason="rogue")
        assert ks.is_killed("bad_agent", "any_tool") is True
        assert ks.is_killed("good_agent", "any_tool") is False

    def test_tool_kill(self):
        ks = KillSwitch()
        ks.activate(KillMode.TOOL, targets=["dangerous_tool"], reason="vuln")
        assert ks.is_killed("any_agent", "dangerous_tool") is True
        assert ks.is_killed("any_agent", "safe_tool") is False

    def test_selective_deactivation(self):
        ks = KillSwitch()
        ks.activate(KillMode.AGENT, targets=["a", "b"], reason="test")
        ks.deactivate(mode=KillMode.AGENT, targets=["a"])
        assert ks.is_agent_killed("a") is False
        assert ks.is_agent_killed("b") is True

    def test_status(self):
        ks = KillSwitch()
        ks.activate(KillMode.ALL, reason="emergency")
        status = ks.status()
        assert status["active"] is True
        assert status["mode"] == "all"


# ─── Alert Engine Tests ───────────────────────────────────────────

class TestAlertEngine:
    def test_fire_alert(self):
        fired = []
        engine = AlertEngine({
            "channels": [
                {"type": "callback", "callback": lambda a: fired.append(a), "events": ["*"]}
            ]
        })
        engine.fire("policy_violation", message="test alert")
        assert len(fired) == 1
        assert fired[0].event_type == "policy_violation"

    def test_event_filtering(self):
        fired = []
        engine = AlertEngine({
            "channels": [
                {
                    "type": "callback",
                    "callback": lambda a: fired.append(a),
                    "events": ["kill_switch"],
                }
            ]
        })
        engine.fire("policy_violation", message="should not fire")
        assert len(fired) == 0
        engine.fire("kill_switch", message="should fire")
        assert len(fired) == 1

    def test_auto_severity(self):
        engine = AlertEngine()
        # Check internal severity mapping
        assert engine._auto_severity("kill_switch") == "critical"
        assert engine._auto_severity("policy_violation") == "warning"
        assert engine._auto_severity("custom_event") == "info"

    def test_alert_history(self):
        engine = AlertEngine({"channels": [{"type": "log", "events": ["*"]}]})
        engine.fire("test1", message="first")
        engine.fire("test2", message="second")
        history = engine.recent_alerts()
        assert len(history) == 2
        assert engine.alert_count() == 2


# ─── Audit Logger Tests ──────────────────────────────────────────

class TestAuditLogger:
    def test_stdout_backend(self, capsys):
        logger = AuditLogger({"backend": "stdout"})
        event = AuditEvent(
            call_id="test-1",
            agent="researcher",
            tool="web_search",
            status="allowed",
        )
        logger.log(event)
        captured = capsys.readouterr()
        assert "researcher" in captured.err
        assert "web_search" in captured.err

    def test_sqlite_backend(self, tmp_path):
        db_path = str(tmp_path / "test_audit.db")
        logger = AuditLogger({"backend": "sqlite", "path": db_path})

        event = AuditEvent(
            call_id="test-1",
            agent="researcher",
            tool="web_search",
            status="allowed",
        )
        logger.log(event)

        results = logger.query(agent="researcher")
        assert len(results) == 1
        assert results[0]["tool"] == "web_search"

        count = logger.count(agent="researcher")
        assert count == 1

        logger.close()

    def test_jsonl_backend(self, tmp_path):
        log_path = str(tmp_path / "test_audit.jsonl")
        logger = AuditLogger({"backend": "jsonl", "path": log_path})

        event = AuditEvent(
            call_id="test-1",
            agent="a",
            tool="t",
            status="blocked",
        )
        logger.log(event)

        results = logger.query()
        assert len(results) == 1


# ─── Interceptor (Rungate) Integration Tests ─────────────────────

class TestRungate:
    def test_basic_allow(self):
        gate = Rungate({
            "policies": {"default": {"mode": "allow"}},
            "audit": {"backend": "none"},
        })
        result = gate.call(
            agent="test",
            tool="search",
            args={"query": "hello"},
            callable=lambda **kw: "found it",
        )
        assert result.allowed is True
        assert result.result == "found it"

    def test_basic_block(self):
        gate = Rungate({
            "policies": {"default": {"mode": "deny"}},
            "audit": {"backend": "none"},
        })
        result = gate.call(
            agent="test",
            tool="search",
            args={},
            callable=lambda **kw: "should not run",
        )
        assert result.allowed is False
        assert result.result is None

    def test_validation_block(self):
        gate = Rungate({
            "policies": {"default": {"mode": "allow"}},
            "schemas": {
                "search": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string"}},
                }
            },
            "audit": {"backend": "none"},
        })
        result = gate.call(agent="a", tool="search", args={}, callable=lambda **kw: "x")
        assert result.allowed is False
        assert result.status == CallStatus.BLOCKED_VALIDATION

    def test_kill_switch(self):
        gate = Rungate({
            "policies": {"default": {"mode": "allow"}},
            "audit": {"backend": "none"},
        })
        gate.kill()
        result = gate.call(agent="a", tool="t", args={}, callable=lambda **kw: "x")
        assert result.allowed is False
        assert result.status == CallStatus.BLOCKED_KILL_SWITCH

        gate.unkill()
        result = gate.call(agent="a", tool="t", args={}, callable=lambda **kw: "y")
        assert result.allowed is True

    def test_monitor_mode(self):
        gate = Rungate({
            "mode": "monitor",
            "policies": {"default": {"mode": "deny"}},
            "audit": {"backend": "none"},
        })
        # In monitor mode, policy block is logged but call proceeds
        result = gate.call(
            agent="a",
            tool="blocked_tool",
            args={},
            callable=lambda **kw: "ran anyway",
        )
        assert result.allowed is True
        assert result.result == "ran anyway"

    def test_stats(self):
        gate = Rungate({
            "policies": {"default": {"mode": "allow"}},
            "audit": {"backend": "none"},
        })
        gate.call(agent="a", tool="t", args={}, callable=lambda **kw: 1)
        gate.call(agent="b", tool="t", args={}, callable=lambda **kw: 2)
        stats = gate.stats()
        assert stats["total_calls"] == 2

    def test_wrap_decorator(self):
        gate = Rungate({
            "policies": {"default": {"mode": "allow"}},
            "audit": {"backend": "none"},
        })

        @gate.decorator(tool_name="my_func", agent="test")
        def my_func(x):
            return x * 2

        assert my_func(5) == 10

    def test_wrap_decorator_blocked(self):
        gate = Rungate({
            "policies": {"default": {"mode": "deny"}},
            "audit": {"backend": "none"},
        })

        @gate.decorator(tool_name="my_func", agent="test")
        def my_func(x):
            return x * 2

        with pytest.raises(ToolCallBlocked):
            my_func(5)

    def test_from_dict(self):
        config = {
            "mode": "enforce",
            "policies": {"default": {"mode": "allow"}},
            "audit": {"backend": "none"},
        }
        gate = Rungate.from_dict(config)
        result = gate.call(agent="a", tool="t", args={}, callable=lambda **kw: "ok")
        assert result.allowed is True


# ─── Async Tests ──────────────────────────────────────────────────

class TestAsync:
    @pytest.mark.asyncio
    async def test_async_call(self):
        gate = Rungate({
            "policies": {"default": {"mode": "allow"}},
            "audit": {"backend": "none"},
        })

        async def async_tool(**kw):
            return "async result"

        result = await gate.acall(
            agent="a",
            tool="t",
            args={},
            callable=async_tool,
        )
        assert result.allowed is True
        assert result.result == "async result"
