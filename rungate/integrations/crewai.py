"""
Rungate CrewAI Integration — Middleware for CrewAI agent tool-calls.

Wraps CrewAI's tool execution to validate every tool-call through
Rungate's pipeline before execution.

Supports:
- CrewAI BaseTool wrapping
- Agent-level wrapping (wraps all tools assigned to an agent)
- Automatic agent name extraction from CrewAI Agent role

Usage:
    from crewai import Agent, Task, Crew
    from rungate.integrations.crewai import gate_agent, gate_tools

    gate = Rungate(config)

    # Option 1: Wrap an agent's tools
    agent = Agent(role="Researcher", tools=[search_tool, scrape_tool])
    safe_agent = gate_agent(agent, gate)

    # Option 2: Wrap tools directly
    safe_tools = gate_tools([search_tool], gate, agent="researcher")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rungate.core.interceptor import Rungate, GateResult


class RungateCrewTool:
    """
    Wraps a CrewAI tool to pass calls through Rungate's pipeline.

    Preserves the tool's name, description, and interface while adding
    validation, rate limiting, cost tracking, and audit logging.
    """

    def __init__(
        self,
        tool: Any,
        gate: Rungate,
        agent: str = "crewai_agent",
    ):
        self._tool = tool
        self._gate = gate
        self._agent = agent

        # Preserve tool metadata for CrewAI
        self.name = getattr(tool, "name", type(tool).__name__)
        self.description = getattr(tool, "description", "")

        # CrewAI-specific attributes
        if hasattr(tool, "args_schema"):
            self.args_schema = tool.args_schema

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Synchronous tool execution through Rungate."""
        call_args = self._extract_args(args, kwargs)

        result = self._gate.call(
            agent=self._agent,
            tool=self.name,
            args=call_args,
            func=lambda: self._tool._run(*args, **kwargs),
        )

        if not result.allowed:
            from rungate.core.interceptor import ToolCallBlocked
            raise ToolCallBlocked(
                f"Tool '{self.name}' blocked: {result.error}",
                call=result.call,
            )

        return result.result

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """Async tool execution through Rungate."""
        call_args = self._extract_args(args, kwargs)

        if hasattr(self._tool, "_arun"):
            result = await self._gate.acall(
                agent=self._agent,
                tool=self.name,
                args=call_args,
                func=lambda: self._tool._arun(*args, **kwargs),
            )
        else:
            result = self._gate.call(
                agent=self._agent,
                tool=self.name,
                args=call_args,
                func=lambda: self._tool._run(*args, **kwargs),
            )

        if not result.allowed:
            from rungate.core.interceptor import ToolCallBlocked
            raise ToolCallBlocked(
                f"Tool '{self.name}' blocked: {result.error}",
                call=result.call,
            )

        return result.result

    def _extract_args(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Extract and normalize arguments."""
        result: Dict[str, Any] = dict(kwargs)
        if args:
            result["_positional"] = list(args)
        return result

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying tool."""
        return getattr(self._tool, name)

    def __repr__(self) -> str:
        return f"RungateCrewTool({self.name}, agent={self._agent})"


def gate_tools(
    tools: List[Any],
    gate: Rungate,
    agent: str = "crewai_agent",
) -> List[RungateCrewTool]:
    """
    Wrap a list of CrewAI tools with Rungate.

    Usage:
        safe_tools = gate_tools([search_tool, scrape_tool], gate, agent="researcher")
    """
    return [RungateCrewTool(t, gate, agent) for t in tools]


def gate_agent(agent: Any, gate: Rungate) -> Any:
    """
    Wrap all tools on a CrewAI Agent with Rungate.

    Extracts the agent's role as the Rungate agent name.

    Usage:
        agent = Agent(role="Researcher", tools=[search_tool])
        safe_agent = gate_agent(agent, gate)
    """
    agent_name = getattr(agent, "role", "crewai_agent")
    # Normalize role to a valid agent name
    agent_name = agent_name.lower().replace(" ", "_")

    if hasattr(agent, "tools") and agent.tools:
        agent.tools = gate_tools(agent.tools, gate, agent=agent_name)

    return agent
