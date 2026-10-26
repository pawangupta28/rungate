"""
Rungate Policy Engine — YAML-defined rules for tool-call governance.

Policies define what agents can and cannot do:
- Which tools each agent can call (allow/block)
- Conditional rules (block if args match a pattern)
- Default deny or default allow per agent
- Wildcard patterns for tools and agents

Policy YAML format:
    policies:
      default:
        mode: deny           # deny-by-default: only explicitly allowed tools work
        tools:
          web_search: allow
          execute_code: block

      researcher:
        mode: allow          # allow-by-default: only explicitly blocked tools are stopped
        tools:
          delete_file: block
          send_email: block
        conditions:
          - tool: web_search
            block_if:
              query_contains: ["password", "credential", "secret"]

      code_agent:
        inherits: default
        tools:
          execute_code: allow   # Override the default block
          write_file:
            action: allow
            max_args:
              path_pattern: "^/tmp/.*"   # Only allow writes to /tmp
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    MONITOR = "monitor"  # Allow but flag for review


@dataclass
class PolicyCondition:
    """A conditional rule that blocks/allows based on argument inspection."""
    tool: str
    action: PolicyDecision = PolicyDecision.BLOCK
    arg_rules: Dict[str, Any] = field(default_factory=dict)

    def matches(self, tool: str, args: Dict[str, Any]) -> bool:
        """Check if this condition applies to the given call."""
        if not fnmatch.fnmatch(tool, self.tool):
            return False
        return self._check_arg_rules(args)

    def _check_arg_rules(self, args: Dict[str, Any]) -> bool:
        """Evaluate argument-based conditions."""
        for rule_key, rule_value in self.arg_rules.items():
            if rule_key.endswith("_contains"):
                # Check if an arg value contains any of the specified strings
                arg_name = rule_key.replace("_contains", "")
                arg_val = str(args.get(arg_name, "")).lower()
                if isinstance(rule_value, list):
                    if any(v.lower() in arg_val for v in rule_value):
                        return True
                elif isinstance(rule_value, str):
                    if rule_value.lower() in arg_val:
                        return True

            elif rule_key.endswith("_pattern"):
                # Regex pattern match on arg value
                arg_name = rule_key.replace("_pattern", "")
                arg_val = str(args.get(arg_name, ""))
                if isinstance(rule_value, str):
                    if not re.match(rule_value, arg_val):
                        return True  # Doesn't match the required pattern → trigger

            elif rule_key.endswith("_max"):
                # Numeric max check
                arg_name = rule_key.replace("_max", "")
                arg_val = args.get(arg_name)
                if arg_val is not None:
                    try:
                        if float(arg_val) > float(rule_value):
                            return True
                    except (ValueError, TypeError):
                        pass

            elif rule_key.endswith("_min"):
                # Numeric min check
                arg_name = rule_key.replace("_min", "")
                arg_val = args.get(arg_name)
                if arg_val is not None:
                    try:
                        if float(arg_val) < float(rule_value):
                            return True
                    except (ValueError, TypeError):
                        pass

            elif rule_key.endswith("_in"):
                # Value must be in allowed list
                arg_name = rule_key.replace("_in", "")
                arg_val = args.get(arg_name)
                if isinstance(rule_value, list) and arg_val not in rule_value:
                    return True

            elif rule_key.endswith("_not_in"):
                # Value must NOT be in blocked list
                arg_name = rule_key.replace("_not_in", "")
                arg_val = args.get(arg_name)
                if isinstance(rule_value, list) and arg_val in rule_value:
                    return True

        return False


@dataclass
class PolicyRule:
    """A rule for a specific tool within a policy."""
    action: PolicyDecision = PolicyDecision.ALLOW
    constraints: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, args: Dict[str, Any]) -> PolicyDecision:
        """Evaluate this rule against the given args."""
        # If there are constraints, check them
        for key, value in self.constraints.items():
            if key.endswith("_pattern"):
                arg_name = key.replace("_pattern", "")
                arg_val = str(args.get(arg_name, ""))
                if not re.match(value, arg_val):
                    return PolicyDecision.BLOCK
        return self.action


@dataclass
class Policy:
    """A complete policy for one agent or agent group."""
    name: str = "default"
    mode: str = "deny"  # "deny" = deny-by-default, "allow" = allow-by-default
    tools: Dict[str, Union[str, PolicyRule]] = field(default_factory=dict)
    conditions: List[PolicyCondition] = field(default_factory=list)
    inherits: Optional[str] = None

    def evaluate(self, tool: str, args: Dict[str, Any]) -> PolicyDecision:
        """Evaluate whether a tool-call is allowed under this policy."""
        # Check conditions first (most specific)
        for condition in self.conditions:
            if condition.matches(tool, args):
                return condition.action

        # Check explicit tool rules
        tool_rule = self._match_tool(tool)
        if tool_rule is not None:
            if isinstance(tool_rule, str):
                return PolicyDecision(tool_rule)
            elif isinstance(tool_rule, PolicyRule):
                return tool_rule.evaluate(args)

        # Fall back to default mode
        if self.mode == "deny":
            return PolicyDecision.BLOCK
        return PolicyDecision.ALLOW

    def _match_tool(self, tool: str) -> Optional[Union[str, PolicyRule]]:
        """Find a matching tool rule, supporting wildcards."""
        # Exact match first
        if tool in self.tools:
            return self.tools[tool]

        # Wildcard match
        for pattern, rule in self.tools.items():
            if "*" in pattern or "?" in pattern:
                if fnmatch.fnmatch(tool, pattern):
                    return rule

        return None


class PolicyEngine:
    """
    Evaluates tool-calls against a set of agent policies.

    Usage:
        engine = PolicyEngine({
            "default": {"mode": "deny", "tools": {"web_search": "allow"}},
            "researcher": {"mode": "allow", "tools": {"delete_file": "block"}},
        })

        decision = engine.evaluate("researcher", "web_search", {"query": "test"})
        # → PolicyDecision.ALLOW
    """

    def __init__(self, policies_config: Dict[str, Any]):
        self._policies: Dict[str, Policy] = {}
        self._load_policies(policies_config)

    def _load_policies(self, config: Dict[str, Any]) -> None:
        """Load policies from config dict."""
        for name, policy_data in config.items():
            if not isinstance(policy_data, dict):
                continue

            # Parse tool rules
            tools: Dict[str, Union[str, PolicyRule]] = {}
            raw_tools = policy_data.get("tools", {})
            if isinstance(raw_tools, dict):
                for tool_name, tool_config in raw_tools.items():
                    if isinstance(tool_config, str):
                        tools[tool_name] = tool_config
                    elif isinstance(tool_config, dict):
                        tools[tool_name] = PolicyRule(
                            action=PolicyDecision(tool_config.get("action", "allow")),
                            constraints={k: v for k, v in tool_config.items() if k != "action"},
                        )

            # Parse conditions
            conditions: List[PolicyCondition] = []
            raw_conditions = policy_data.get("conditions", [])
            if isinstance(raw_conditions, list):
                for cond in raw_conditions:
                    if isinstance(cond, dict):
                        block_if = cond.get("block_if", {})
                        allow_if = cond.get("allow_if", {})
                        action = PolicyDecision.BLOCK if block_if else PolicyDecision.ALLOW
                        arg_rules = block_if or allow_if
                        conditions.append(PolicyCondition(
                            tool=cond.get("tool", "*"),
                            action=action,
                            arg_rules=arg_rules,
                        ))

            self._policies[name] = Policy(
                name=name,
                mode=policy_data.get("mode", "deny"),
                tools=tools,
                conditions=conditions,
                inherits=policy_data.get("inherits"),
            )

    def evaluate(self, agent: str, tool: str, args: Dict[str, Any]) -> PolicyDecision:
        """Evaluate a tool-call against the agent's policy."""
        policy = self._resolve_policy(agent)
        return policy.evaluate(tool, args)

    def _resolve_policy(self, agent: str) -> Policy:
        """Resolve the policy for an agent, following inheritance."""
        if agent in self._policies:
            policy = self._policies[agent]
            if policy.inherits and policy.inherits in self._policies:
                return self._merge_policies(
                    self._policies[policy.inherits],
                    policy,
                )
            return policy

        # Try wildcard matching
        for name, policy in self._policies.items():
            if "*" in name and fnmatch.fnmatch(agent, name):
                return policy

        # Fall back to default
        if "default" in self._policies:
            return self._policies["default"]

        # No policy at all — deny by default
        return Policy(name="_fallback", mode="deny")

    def _merge_policies(self, parent: Policy, child: Policy) -> Policy:
        """Merge a child policy with its parent (child overrides parent)."""
        merged_tools = {**parent.tools, **child.tools}
        merged_conditions = parent.conditions + child.conditions
        return Policy(
            name=child.name,
            mode=child.mode if child.mode != parent.mode else parent.mode,
            tools=merged_tools,
            conditions=merged_conditions,
        )

    def add_policy(self, name: str, policy_data: Dict[str, Any]) -> None:
        """Add or update a policy at runtime."""
        self._load_policies({name: policy_data})

    def remove_policy(self, name: str) -> bool:
        """Remove a policy. Returns True if it existed."""
        return self._policies.pop(name, None) is not None

    def list_policies(self) -> List[str]:
        """List all loaded policy names."""
        return list(self._policies.keys())

    def get_policy(self, name: str) -> Optional[Policy]:
        """Get a policy by name."""
        return self._policies.get(name)

    def __repr__(self) -> str:
        return f"PolicyEngine(policies={list(self._policies.keys())})"
