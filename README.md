# Rungate

**Runtime validation middleware for AI agent tool-calls.**

Rungate sits between your AI agents and their tools, validating every tool-call before execution. Policy enforcement, schema validation, rate limiting, cost tracking, kill switches, and full audit trails — in a single pip install with zero required dependencies.

[![License: BSL 1.1](https://img.shields.io/badge/License-BSL%201.1-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## Why Rungate?

AI agents fail 3–15% of tool-calls individually, and 41–87% in multi-agent workflows ([arXiv:2311.05785](https://arxiv.org/abs/2311.05785), [UC Berkeley MAST](https://arxiv.org/abs/2401.15391)). A single unchecked `execute_code` or `send_email` call can leak data, burn budget, or take down production systems.

Existing guardrails validate LLM *outputs* (text/content). Rungate validates LLM *actions* (tool-calls) — the actual function invocations that interact with your systems.

```
┌──────────┐     ┌──────────────────────────────┐     ┌──────────┐
│ AI Agent │────▶│          RUNGATE              │────▶│  Tools   │
│          │     │                               │     │          │
│ LangChain│     │ ┌─────────┐  ┌────────────┐  │     │ APIs     │
│ CrewAI   │     │ │ Policy  │  │ Validation │  │     │ DBs      │
│ OpenAI   │     │ │ Engine  │  │ (Schema)   │  │     │ Files    │
│ Custom   │     │ ├─────────┤  ├────────────┤  │     │ Code     │
│          │     │ │ Rate    │  │ Cost       │  │     │ Email    │
│          │     │ │ Limiter │  │ Tracker    │  │     │          │
│          │     │ ├─────────┤  ├────────────┤  │     │          │
│          │     │ │ Kill    │  │ Audit      │  │     │          │
│          │     │ │ Switch  │  │ Logger     │  │     │          │
│          │◀────│ └─────────┘  └────────────┘  │◀────│          │
└──────────┘     └──────────────────────────────┘     └──────────┘
```

## Quick Start

```bash
pip install rungate-ai
```

```python
from rungate import Rungate

# Define policies
gate = Rungate({
    "policies": {
        "default": {
            "mode": "deny",
            "tools": {
                "web_search": "allow",
                "read_file": "allow",
            }
        },
        "researcher": {
            "mode": "allow",
            "tools": {"delete_file": "block", "execute_code": "block"},
            "conditions": [{
                "tool": "web_search",
                "block_if": {"query_contains": ["password", "credential"]},
            }],
        }
    },
    "schemas": {
        "web_search": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 1000}
            }
        }
    },
    "rate_limits": {
        "global": {"max_calls": 100, "window_seconds": 60}
    },
    "cost": {
        "tool_costs": {"web_search": 0.01, "execute_code": 0.05},
        "budgets": {"global": 50.0, "per_agent": {"researcher": 10.0}},
    },
})

# Validate and execute a tool-call
result = gate.call(
    agent="researcher",
    tool="web_search",
    args={"query": "AI safety research"},
    callable=lambda **kw: search_api(kw["query"]),
)

if result.allowed:
    print(result.result)
else:
    print(f"Blocked: {result.error}")
```

## Features

### 8 Core Modules

| Module | What it does |
|--------|-------------|
| **Interceptor** | Central pipeline — wraps any callable, runs through all checks |
| **Policy Engine** | YAML-defined allow/block rules per agent, per tool, with conditions |
| **Schema Validator** | JSON Schema validation of tool-call arguments |
| **Rate Limiter** | Sliding-window limits: global, per-agent, per-tool, per-agent-tool |
| **Cost Tracker** | Per-call cost estimation, budget caps, auto-kill on overspend |
| **Kill Switch** | Emergency stop — kill all, specific agents, or specific tools |
| **Audit Logger** | Structured logs to SQLite, JSONL, stdout, or custom callbacks |
| **Alert Engine** | Real-time alerts via webhook, Slack, email on violations |

### Zero Required Dependencies

The core library uses only Python stdlib. Optional dependencies unlock extra features:

```bash
pip install rungate-ai              # Core (stdlib only)
pip install rungate-ai[yaml]        # + YAML config support
pip install rungate-ai[schema]      # + jsonschema validation
pip install rungate-ai[server]      # + FastAPI SaaS server
pip install rungate-ai[all]         # Everything
```

### Three Operating Modes

```python
# Enforce: block violations (production)
gate = Rungate({"mode": "enforce", ...})

# Monitor: log violations but don't block (rollout)
gate = Rungate({"mode": "monitor", ...})

# Dry-run: validate but don't execute (testing)
gate = Rungate({"mode": "dry-run", ...})
```

## Framework Integrations

### LangChain

```python
from rungate.integrations.langchain import wrap_tools

gate = Rungate(config)
safe_tools = wrap_tools(tools, gate, agent="researcher")
executor = AgentExecutor(agent=agent, tools=safe_tools)
```

### CrewAI

```python
from rungate.integrations.crewai import gate_agent

gate = Rungate(config)
agent = Agent(role="Researcher", tools=[search, scrape])
safe_agent = gate_agent(agent, gate)
```

### OpenAI Function Calling

```python
from rungate.integrations.openai_fc import gate_tool_calls

response = client.chat.completions.create(model="gpt-4", messages=msgs, tools=tools)
results = gate_tool_calls(gate, "my_agent", response.choices[0].message.tool_calls, executor)
```

### Generic Python

```python
from rungate.integrations.generic import gated

@gated(gate, agent="my_agent")
def web_search(query: str) -> list:
    return search_api(query)

results = web_search("AI safety")  # Passes through Rungate
```

## Kill Switch

Emergency stop for runaway agents:

```python
# Kill everything
gate.kill()

# Kill specific agent
from rungate.core.kill_switch import KillMode
gate.kill(mode=KillMode.AGENT, targets=["rogue_agent"], reason="Budget exceeded")

# Kill specific tool
gate.kill(mode=KillMode.TOOL, targets=["execute_code"], reason="Security incident")

# Resume
gate.unkill()
```

## SaaS API

Run Rungate as a hosted validation service:

```bash
pip install rungate-ai[server]
rungate-server --port 8000
```

```bash
# Validate a tool-call
curl -X POST http://localhost:8000/v1/validate \
  -H "Authorization: Bearer rg_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"agent": "researcher", "tool": "web_search", "args": {"query": "test"}}'
```

Same `Rungate` class, adding `api_key=` switches from self-hosted to SaaS:

```python
# Self-hosted
gate = Rungate(config)

# SaaS (same API)
gate = Rungate(config, api_key="rg_your_key", api_url="https://api.rungate.dev")
```

## Configuration

### YAML Config

```yaml
mode: enforce

policies:
  default:
    mode: deny
    tools:
      web_search: allow

  researcher:
    mode: allow
    tools:
      delete_file: block
    conditions:
      - tool: web_search
        block_if:
          query_contains: ["password", "secret"]

schemas:
  web_search:
    type: object
    required: [query]
    properties:
      query:
        type: string
        minLength: 1

rate_limits:
  global:
    max_calls: 100
    window_seconds: 60

cost:
  tool_costs:
    web_search: 0.01
  budgets:
    global: 100.0
```

Load from YAML:

```python
gate = Rungate.from_yaml("config/policies/default.yaml")
```

## Audit & Observability

Every tool-call is logged with full context:

```python
# Query audit logs
events = gate._audit.query(agent="researcher", limit=10)

# Get system stats
stats = gate.stats()
# → {"total_calls": 1523, "blocked": 47, "kill_switch": {...}, ...}
```

## License

[Business Source License 1.1](LICENSE)

- **Free** for organizations under $1M ARR
- **Paid** for organizations over $1M ARR (contact for pricing)
- **Converts to Apache 2.0** on October 26, 2027

## Contributing

Contributions are welcome. Please open an issue to discuss before submitting a PR.

```bash
git clone https://github.com/pawangupta28/rungate
cd rungate
pip install -e ".[dev]"
pytest tests/ -v
```
