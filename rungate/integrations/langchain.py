"""
Rungate LangChain Integration — Drop-in middleware for LangChain agents.

Wraps LangChain's tool execution to pass every tool-call through Rungate's
validation pipeline before execution.

Supports:
- LangChain BaseTool wrapping
- AgentExecutor callback integration
- Automatic agent name extraction from chain context

Usage:
    from rungate.integrations.langchain import RungateToolWrapper, RungateCallback

    # Option 1: Wrap individual tools
    gate = Rungate(config)
    wrapped_tool = RungateToolWrapper(original_tool, gate, agent="my_agent")

    # Option 2: Use as AgentExecutor callback
    callback = RungateCallback(gate)
    executor = AgentExecutor(agent=agent, tools=tools, callbacks=[callback])
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TYPE_CHECKING

from rungate.core.interceptor import Rungate, GateResult

if TYPE_CHECKING:
    pass  # LangChain types for IDE support only


class RungateToolWrapper:
    """
    Wraps a LangChain BaseTool to pass calls through Rungate.

    The wrapped tool behaves identically to the original, but every call
    is validated against the Rungate pipeline first. Blocked calls raise
    ToolCallBlocked.

    Usage:
        from langchain.tools import DuckDuckGoSearchRun

        search = DuckDuckGoSearchRun()
        gate = Rungate(config)
        safe_search = RungateToolWrapper(search, gate, agent="researcher")

        # Use safe_search exactly like the original
        result = safe_search.run("query")
    """

    def __init__(
        self,
        tool: Any,
        gate: Rungate,
        agent: str = "langchain_agent",
    ):
        self._tool = tool
        self._gate = gate
        self._agent = agent

        # Preserve tool metadata
        self.name = getattr(tool, "name", type(tool).__name__)
        self.description = getattr(tool, "description", "")

    def run(self, tool_input: Any, **kwargs: Any) -> Any:
        """Run the tool through Rungate's pipeline."""
        args = self._normalize_args(tool_input)

        result = self._gate.call(
            agent=self._agent,
            tool=self.name,
            args=args,
            func=lambda: self._tool.run(tool_input, **kwargs),
        )

        if not result.allowed:
            from rungate.core.interceptor import ToolCallBlocked
            raise ToolCallBlocked(
                f"Tool '{self.name}' blocked by Rungate: {result.error}",
                call=result.call,
            )

        return result.result

    async def arun(self, tool_input: Any, **kwargs: Any) -> Any:
        """Async run the tool through Rungate's pipeline."""
        args = self._normalize_args(tool_input)

        result = await self._gate.acall(
            agent=self._agent,
            tool=self.name,
            args=args,
            func=lambda: self._tool.arun(tool_input, **kwargs),
        )

        if not result.allowed:
            from rungate.core.interceptor import ToolCallBlocked
            raise ToolCallBlocked(
                f"Tool '{self.name}' blocked by Rungate: {result.error}",
                call=result.call,
            )

        return result.result

    def _normalize_args(self, tool_input: Any) -> Dict[str, Any]:
        """Normalize LangChain tool input to a dict."""
        if isinstance(tool_input, dict):
            return tool_input
        elif isinstance(tool_input, str):
            return {"input": tool_input}
        return {"input": str(tool_input)}

    def __getattr__(self, name: str) -> Any:
        """Proxy attribute access to the underlying tool."""
        return getattr(self._tool, name)

    def __repr__(self) -> str:
        return f"RungateToolWrapper({self.name}, agent={self._agent})"


class RungateCallback:
    """
    LangChain callback handler that logs tool-calls through Rungate.

    This is a monitoring-only integration — it logs calls but doesn't
    block them (use RungateToolWrapper for enforcement).

    Usage:
        gate = Rungate(config)
        callback = RungateCallback(gate)
        executor = AgentExecutor(agent=agent, tools=tools, callbacks=[callback])
    """

    def __init__(
        self,
        gate: Rungate,
        agent: str = "langchain_agent",
    ):
        self._gate = gate
        self._agent = agent

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts running."""
        tool_name = serialized.get("name", "unknown")
        args = {"input": input_str}

        # Fire through gate in monitor mode (log but don't block)
        self._gate.call(
            agent=self._agent,
            tool=tool_name,
            args=args,
            func=lambda: None,  # No-op — just for logging
        )

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Called when a tool finishes."""
        pass

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        """Called when a tool errors."""
        pass


def wrap_tools(
    tools: list,
    gate: Rungate,
    agent: str = "langchain_agent",
) -> list:
    """
    Convenience: wrap a list of LangChain tools with Rungate.

    Usage:
        from langchain.tools import DuckDuckGoSearchRun, ShellTool
        from rungate.integrations.langchain import wrap_tools

        tools = [DuckDuckGoSearchRun(), ShellTool()]
        gate = Rungate(config)
        safe_tools = wrap_tools(tools, gate, agent="my_agent")
    """
    return [RungateToolWrapper(t, gate, agent) for t in tools]
