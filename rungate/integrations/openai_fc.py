"""
Rungate OpenAI Function Calling Integration — Middleware for OpenAI's function calling.

Intercepts function calls from OpenAI's chat completions API and validates
them through Rungate before execution.

Supports:
- Function call argument validation
- Tool-call-level interception (parallel tool calls)
- Drop-in replacement for function execution loops

Usage:
    from openai import OpenAI
    from rungate.integrations.openai_fc import gate_function_call, gate_tool_calls

    gate = Rungate(config)
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
    )

    # Validate and execute each tool call
    for tool_call in response.choices[0].message.tool_calls:
        result = gate_function_call(
            gate=gate,
            agent="openai_agent",
            tool_call=tool_call,
            executor=execute_function,
        )
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from rungate.core.interceptor import Rungate, GateResult


def gate_function_call(
    gate: Rungate,
    agent: str,
    tool_call: Any,
    executor: Callable[..., Any],
) -> GateResult:
    """
    Validate and execute a single OpenAI function call through Rungate.

    Args:
        gate: Rungate instance
        agent: Agent identifier
        tool_call: OpenAI tool_call object (from response.choices[0].message.tool_calls)
        executor: Function that executes the tool (receives tool_name, args)

    Returns:
        GateResult with the execution outcome

    Usage:
        def execute_function(name, args):
            if name == "get_weather":
                return get_weather(**args)

        result = gate_function_call(gate, "my_agent", tool_call, execute_function)
        if result.allowed:
            # Add result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result.result),
            })
    """
    # Extract function name and args from the tool_call object
    func_name, func_args = _extract_tool_call(tool_call)

    return gate.call(
        agent=agent,
        tool=func_name,
        args=func_args,
        func=lambda: executor(func_name, func_args),
    )


async def agate_function_call(
    gate: Rungate,
    agent: str,
    tool_call: Any,
    executor: Callable[..., Any],
) -> GateResult:
    """Async version of gate_function_call."""
    func_name, func_args = _extract_tool_call(tool_call)

    return await gate.acall(
        agent=agent,
        tool=func_name,
        args=func_args,
        func=lambda: executor(func_name, func_args),
    )


def gate_tool_calls(
    gate: Rungate,
    agent: str,
    tool_calls: List[Any],
    executor: Callable[..., Any],
) -> List[Dict[str, Any]]:
    """
    Validate and execute all tool calls from an OpenAI response.

    Returns a list of tool result messages ready to append to the
    conversation.

    Usage:
        response = client.chat.completions.create(...)
        tool_calls = response.choices[0].message.tool_calls

        if tool_calls:
            tool_messages = gate_tool_calls(gate, "my_agent", tool_calls, executor)
            messages.extend(tool_messages)
    """
    results = []

    for tool_call in tool_calls:
        tc_id = _get_tool_call_id(tool_call)
        func_name, func_args = _extract_tool_call(tool_call)

        gate_result = gate.call(
            agent=agent,
            tool=func_name,
            args=func_args,
            func=lambda fn=func_name, fa=func_args: executor(fn, fa),
        )

        if gate_result.allowed:
            results.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": str(gate_result.result),
            })
        else:
            # Return error message so the model knows the call was blocked
            results.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": f"[BLOCKED by Rungate] {gate_result.error}",
            })

    return results


async def agate_tool_calls(
    gate: Rungate,
    agent: str,
    tool_calls: List[Any],
    executor: Callable[..., Any],
) -> List[Dict[str, Any]]:
    """Async version of gate_tool_calls."""
    results = []

    for tool_call in tool_calls:
        tc_id = _get_tool_call_id(tool_call)
        func_name, func_args = _extract_tool_call(tool_call)

        gate_result = await gate.acall(
            agent=agent,
            tool=func_name,
            args=func_args,
            func=lambda fn=func_name, fa=func_args: executor(fn, fa),
        )

        if gate_result.allowed:
            results.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": str(gate_result.result),
            })
        else:
            results.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": f"[BLOCKED by Rungate] {gate_result.error}",
            })

    return results


def _extract_tool_call(tool_call: Any) -> tuple:
    """Extract function name and args from an OpenAI tool_call object."""
    # Handle OpenAI SDK objects
    if hasattr(tool_call, "function"):
        func = tool_call.function
        name = getattr(func, "name", "unknown")
        args_str = getattr(func, "arguments", "{}")
    # Handle dict-style tool calls
    elif isinstance(tool_call, dict):
        func = tool_call.get("function", {})
        name = func.get("name", "unknown")
        args_str = func.get("arguments", "{}")
    else:
        return "unknown", {}

    # Parse arguments
    if isinstance(args_str, str):
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            args = {"_raw": args_str}
    elif isinstance(args_str, dict):
        args = args_str
    else:
        args = {}

    return name, args


def _get_tool_call_id(tool_call: Any) -> str:
    """Extract tool_call ID."""
    if hasattr(tool_call, "id"):
        return tool_call.id
    elif isinstance(tool_call, dict):
        return tool_call.get("id", "")
    return ""
