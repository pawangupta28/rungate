"""
Rungate Generic Integration — Framework-agnostic tool-call wrapping.

For any Python callable — works with custom agents, homegrown frameworks,
or direct function calls.

Usage:
    from rungate.integrations.generic import gated, GatedExecutor

    gate = Rungate(config)

    # Option 1: Decorator
    @gated(gate, agent="my_agent")
    def web_search(query: str, max_results: int = 10) -> list:
        return do_search(query, max_results)

    results = web_search("AI safety")  # Passes through Rungate

    # Option 2: Executor class
    executor = GatedExecutor(gate, agent="my_agent")
    executor.register("search", search_func)
    executor.register("write_file", write_file_func)
    result = executor.call("search", {"query": "AI safety"})

    # Option 3: Direct wrapping
    safe_fn = gate.wrap(search_func, agent="my_agent", tool="search")
    result = safe_fn(query="AI safety")
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, Optional

from rungate.core.interceptor import Rungate, GateResult


def gated(
    gate: Rungate,
    agent: str = "default",
    tool: Optional[str] = None,
) -> Callable:
    """
    Decorator that passes function calls through Rungate.

    Usage:
        gate = Rungate(config)

        @gated(gate, agent="researcher")
        def web_search(query: str) -> list:
            return search_api(query)

        # Every call is validated
        results = web_search("AI safety")

        # Blocked calls raise ToolCallBlocked
        try:
            results = web_search("password database dump")
        except ToolCallBlocked:
            print("Call was blocked by policy")
    """
    def decorator(func: Callable) -> Callable:
        tool_name = tool or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Convert args to dict for policy evaluation
            call_args = _extract_args(func, args, kwargs)

            result = gate.call(
                agent=agent,
                tool=tool_name,
                args=call_args,
                func=lambda: func(*args, **kwargs),
            )

            if not result.allowed:
                from rungate.core.interceptor import ToolCallBlocked
                raise ToolCallBlocked(
                    f"Call to '{tool_name}' blocked: {result.error}",
                    call=result.call,
                )

            return result.result

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            call_args = _extract_args(func, args, kwargs)

            result = await gate.acall(
                agent=agent,
                tool=tool_name,
                args=call_args,
                func=lambda: func(*args, **kwargs),
            )

            if not result.allowed:
                from rungate.core.interceptor import ToolCallBlocked
                raise ToolCallBlocked(
                    f"Call to '{tool_name}' blocked: {result.error}",
                    call=result.call,
                )

            return result.result

        if inspect.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


class GatedExecutor:
    """
    Registry-based executor that routes named tool-calls through Rungate.

    Useful for custom agent frameworks that dispatch tools by name.

    Usage:
        gate = Rungate(config)
        executor = GatedExecutor(gate, agent="my_agent")

        # Register tools
        executor.register("search", search_function)
        executor.register("write_file", write_file_function)

        # Execute (validated through Rungate)
        result = executor.call("search", {"query": "AI safety"})

        # Check what's registered
        print(executor.list_tools())
    """

    def __init__(self, gate: Rungate, agent: str = "default"):
        self._gate = gate
        self._agent = agent
        self._tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        """Register a tool function."""
        self._tools[name] = func

    def unregister(self, name: str) -> bool:
        """Unregister a tool. Returns True if it existed."""
        return self._tools.pop(name, None) is not None

    def call(self, tool: str, args: Optional[Dict[str, Any]] = None) -> GateResult:
        """
        Execute a named tool through Rungate.

        Args:
            tool: Registered tool name
            args: Arguments to pass to the tool

        Returns:
            GateResult with execution outcome
        """
        args = args or {}
        func = self._tools.get(tool)

        if func is None:
            # Still validate through the gate (will be logged as execution error)
            return self._gate.call(
                agent=self._agent,
                tool=tool,
                args=args,
                func=lambda: (_ for _ in ()).throw(
                    KeyError(f"Tool '{tool}' not registered")
                ),
            )

        return self._gate.call(
            agent=self._agent,
            tool=tool,
            args=args,
            func=lambda: func(**args) if args else func(),
        )

    async def acall(self, tool: str, args: Optional[Dict[str, Any]] = None) -> GateResult:
        """Async version of call."""
        args = args or {}
        func = self._tools.get(tool)

        if func is None:
            return self._gate.call(
                agent=self._agent,
                tool=tool,
                args=args,
                func=lambda: (_ for _ in ()).throw(
                    KeyError(f"Tool '{tool}' not registered")
                ),
            )

        return await self._gate.acall(
            agent=self._agent,
            tool=tool,
            args=args,
            func=lambda: func(**args) if args else func(),
        )

    def list_tools(self) -> list:
        """List registered tool names."""
        return list(self._tools.keys())

    def __repr__(self) -> str:
        return f"GatedExecutor(agent={self._agent}, tools={list(self._tools.keys())})"


def _extract_args(func: Callable, args: tuple, kwargs: dict) -> Dict[str, Any]:
    """Extract named arguments from positional + keyword args."""
    result = dict(kwargs)

    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        for i, arg in enumerate(args):
            if i < len(params):
                result[params[i]] = arg
            else:
                result[f"_arg{i}"] = arg
    except (ValueError, TypeError):
        # Can't inspect signature — store positional args
        for i, arg in enumerate(args):
            result[f"_arg{i}"] = arg

    return result
