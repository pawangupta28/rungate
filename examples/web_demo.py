#!/usr/bin/env python3
"""
Rungate web demo — the data demo in your browser.

    python examples/web_demo.py            # opens http://localhost:8000
    python examples/web_demo.py --port 8080 --no-browser

Each task is shown side by side: WITHOUT Rungate (left) vs WITH Rungate (right),
followed by a damage report and a data explorer (customers, orders, outbox,
secrets, audit log). "Run live" uses a real LLM via Groq (GROQ_API_KEY in env or .env).

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rungate import Rungate  # noqa: E402
import data_demo as dd  # noqa: E402
import demo as ui  # noqa: E402  (.env loading + Groq model resolution)

RUN_LOCK = threading.Lock()
AUDIT_PATH = Path.cwd() / "rungate_data_audit.jsonl"


# ── Workspace inspection ─────────────────────────────────────────────

def detail() -> Dict[str, Any]:
    ws = dd.WS
    out: Dict[str, Any] = {"files": [], "customers": {"count": 0, "rows": [], "plans": {}},
                           "orders": {"count": 0, "by_status": {}, "by_city": []},
                           "outbox": [], "secrets": None, "log_lines": None}
    if not ws.exists():
        return out
    out["files"] = [str(p.relative_to(ws)).replace("\\", "/") for p in sorted(ws.rglob("*")) if p.is_file()]
    cpath = ws / "customers.csv"
    if cpath.exists():
        rows = list(csv.DictReader(cpath.open()))
        plans: Dict[str, int] = {}
        for r in rows:
            plans[r["plan"]] = plans.get(r["plan"], 0) + 1
        out["customers"] = {"count": len(rows), "rows": rows[:15], "plans": plans}
    try:
        db = sqlite3.connect(ws / "orders.db")
        out["orders"]["count"] = db.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        out["orders"]["by_status"] = dict(db.execute(
            "SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY 2 DESC").fetchall())
        out["orders"]["by_city"] = db.execute(
            "SELECT city, ROUND(SUM(amount_usd),0) FROM orders WHERE status='paid' "
            "GROUP BY city ORDER BY 2 DESC").fetchall()
        db.close()
    except sqlite3.Error:
        pass
    for f in sorted((ws / "outbox").glob("*.json")):
        m = json.loads(f.read_text())
        m["external"] = not m["to"].lower().endswith("@acme.com")
        m["records"] = m["body"].count("@example.com")
        m["body"] = m["body"][:600]
        out["outbox"].append(m)
    env = ws / "config" / "prod.env"
    if env.exists():
        out["secrets"] = [ln.split("=")[0] + "=" + "•" * 10 for ln in env.read_text().splitlines() if "=" in ln]
    log = ws / "logs" / "agent.log"
    if log.exists():
        out["log_lines"] = len(log.read_text().splitlines())
    return out


# ── Running the agent, emitting events ───────────────────────────────

Emit = Callable[[Dict[str, Any]], None]


def tool_call(emit: Emit, phase: int, task: int, gate: Optional[Rungate],
              agent: str, tool: str, args: Dict[str, Any]) -> str:
    ev: Dict[str, Any] = {"type": "call", "phase": phase, "task": task, "agent": agent,
                          "tool": tool, "args": args}
    if tool not in dd.TOOLS:
        ev.update(decision="error", reason="unknown tool")
        emit(ev)
        return "error: unknown tool"
    if gate is None:
        try:
            out = dd.TOOLS[tool](**args)
            ev.update(decision="executed", output=str(out)[:400])
        except Exception as e:  # noqa: BLE001
            out = f"error: {e}"
            ev.update(decision="error", reason=str(e))
        emit(ev)
        return out
    res = gate.call(agent=agent, tool=tool, args=args, callable=dd.TOOLS[tool])
    ev.update(trace=res.call.pipeline_trace, ms=round(res.call.duration_ms, 3))
    if res.allowed and res.error is None:
        ev.update(decision="allowed", output=str(res.result)[:400])
        out = str(res.result)
    elif res.allowed:
        ev.update(decision="error", reason=res.error)
        out = f"error: {res.error}"
    else:
        ev.update(decision="blocked", reason=res.status.value.replace("blocked_", ""), error=res.error)
        out = f"BLOCKED by Rungate ({res.status.value}): {res.error}"
    emit(ev)
    return out


def run_phase(emit: Emit, phase: int, gate: Optional[Rungate], live: bool, model: str, key: str) -> List[str]:
    replies: List[str] = []
    for i, (agent, task, steps) in enumerate(dd.TASKS, 1):
        emit({"type": "task", "phase": phase, "task": i, "agent": agent, "text": task})
        if not live:
            for tool, args in steps:
                out = tool_call(emit, phase, i, gate, agent, tool, dd._materialise(args))
                if tool == "read_file" and "prod.env" in args.get("path", "") and not out.startswith("BLOCKED"):
                    replies.append(out)
                    emit({"type": "say", "phase": phase, "task": i,
                          "text": "Here is the database password: " + dd.SECRET})
            continue
        messages = [{"role": "system", "content": dd.SYSTEM}, {"role": "user", "content": task}]
        for _ in range(6):
            try:
                msg = dd._groq(messages, model, key, notify=lambda t: emit(
                    {"type": "note", "phase": phase, "task": i, "text": t}))["choices"][0]["message"]
            except dd.LLMError as e:
                emit({"type": "say", "phase": phase, "task": i,
                      "text": f"LLM error, task skipped: {str(e)[:300]}"})
                break
            calls = msg.get("tool_calls") or []
            if not calls:
                text = (msg.get("content") or "").strip()
                replies.append(text)
                emit({"type": "say", "phase": phase, "task": i, "text": text[:600]})
                break
            messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
            for tc in calls:
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = tool_call(emit, phase, i, gate, agent, tc["function"]["name"], args)
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": out[:3500]})
    return replies


def run_all(emit: Emit, live: bool) -> None:
    key, model = "", ""
    if live:
        key = os.environ.get("GROQ_API_KEY", "")
        if not key:
            emit({"type": "error", "text": "GROQ_API_KEY not set (env var or .env file)."})
            return
        model = ui._resolve_model("", key)
    emit({"type": "start", "brain": f"{model} (live LLM)" if live else "scripted agent (deterministic)",
          "tasks": [{"agent": a, "text": t} for a, t, _ in dd.TASKS]})

    dd.build_workspace()
    before = dd.snapshot()
    emit({"type": "data", "label": "start", "detail": detail()})

    emit({"type": "phase", "phase": 1})
    replies1 = run_phase(emit, 1, None, live, model, key)
    after1 = dd.snapshot()
    emit({"type": "data", "label": "no_gate", "detail": detail()})

    dd.build_workspace()
    if AUDIT_PATH.exists():
        AUDIT_PATH.unlink()
    emit({"type": "phase", "phase": 2})
    gate = Rungate(dd.gate_config())
    replies2 = run_phase(emit, 2, gate, live, model, key)
    after2 = dd.snapshot()
    emit({"type": "data", "label": "with_gate", "detail": detail()})

    audit = [json.loads(l) for l in AUDIT_PATH.read_text().splitlines()] if AUDIT_PATH.exists() else []
    emit({"type": "report", "before": before, "no_gate": after1, "with_gate": after2,
          "leak": [any(dd.SECRET in r for r in replies1), any(dd.SECRET in r for r in replies2)],
          "stats": gate.stats(), "audit": audit})
    emit({"type": "done"})


# ── HTTP server ──────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        url = urlparse(self.path)
        if url.path == "/":
            return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        if url.path == "/policy":
            cfg = dd.gate_config()
            cfg.pop("audit", None)
            return self._send(200, json.dumps(cfg, indent=2).encode(), "application/json")
        if url.path == "/run":
            live = parse_qs(url.query).get("mode", ["scripted"])[0] == "live"
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(ev: Dict[str, Any]) -> None:
                self.wfile.write(f"data: {json.dumps(ev, default=str)}\n\n".encode())
                self.wfile.flush()

            if not RUN_LOCK.acquire(blocking=False):
                emit({"type": "error", "text": "A run is already in progress."})
                return
            try:
                run_all(emit, live)
            except SystemExit as e:  # Groq errors from helpers
                emit({"type": "error", "text": str(e)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:  # noqa: BLE001
                try:
                    emit({"type": "error", "text": f"{type(e).__name__}: {e}"})
                except Exception:  # noqa: BLE001
                    pass
            finally:
                RUN_LOCK.release()
            return
        self._send(404, b"not found", "text/plain")


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Rungate — live demo</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0f1115;--panel:#171a21;--panel2:#1d212a;--line:#2a2f3a;--ink:#e8eaf0;--mute:#8b93a7;
--red:#ef4444;--red-bg:rgba(239,68,68,.12);--green:#22c55e;--green-bg:rgba(34,197,94,.12);
--amber:#f59e0b;--amber-bg:rgba(245,158,11,.12);--blue:#60a5fa;--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:20px 28px;border-bottom:1px solid var(--line);display:flex;gap:24px;align-items:center;flex-wrap:wrap}
h1{font-size:20px;margin:0}h1 span{color:var(--blue)}.sub{color:var(--mute);font-size:13px}
.controls{margin-left:auto;display:flex;gap:10px;align-items:center}
button{background:var(--panel2);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:9px 14px;font-weight:600;cursor:pointer;font-size:13px}
button.primary{background:#2563eb;border-color:#2563eb}button:disabled{opacity:.45;cursor:default}
label.chk{color:var(--mute);font-size:13px;display:flex;gap:6px;align-items:center}
main{padding:20px 28px;max-width:1400px;margin:0 auto}
.pipeline{display:flex;gap:8px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:16px;font-size:12.5px}
.pipeline .st{padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:var(--panel2)}
.pipeline .arr{color:var(--mute)}.pipeline .end{margin-left:6px;color:var(--mute)}
.status{color:var(--mute);margin:0 0 12px;font-size:13px;min-height:20px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px;position:sticky;top:0;background:var(--bg);padding:6px 0;z-index:2}
.colhead{padding:10px 14px;border-radius:8px;font-weight:700;font-size:14px}
.colhead.bad{background:var(--red-bg);color:var(--red);border:1px solid rgba(239,68,68,.4)}
.colhead.good{background:var(--green-bg);color:var(--green);border:1px solid rgba(34,197,94,.4)}
.task{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:14px 0;overflow:hidden}
.task .q{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:10px;align-items:baseline}
.badge{font:600 11px var(--mono);padding:2px 7px;border-radius:5px;background:var(--panel2);border:1px solid var(--line);color:var(--blue);white-space:nowrap}
.task .q .txt{font-size:14px}.task .q .n{color:var(--mute);font-weight:700}
.task .body{display:grid;grid-template-columns:1fr 1fr}.task .side{padding:10px 14px;min-height:54px}
.task .side+.side{border-left:1px solid var(--line)}
.call{border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:6px 0;background:var(--panel2);animation:in .35s ease}
@keyframes in{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
.call .sig{font:12px var(--mono);word-break:break-all}.call .sig b{color:var(--ink)}
.call .res{margin-top:6px;font-size:12.5px;display:flex;gap:8px;align-items:baseline}
.tag{font:700 11px var(--mono);padding:2px 7px;border-radius:5px;white-space:nowrap}
.tag.executed{background:var(--amber-bg);color:var(--amber)}.tag.allowed{background:var(--green-bg);color:var(--green)}
.tag.blocked{background:var(--red-bg);color:var(--red)}.tag.error{background:var(--panel);color:var(--mute)}
.out{color:var(--mute);font:12px var(--mono);word-break:break-word;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}
td .out{-webkit-line-clamp:2}
.trace{margin-top:6px;font:11px var(--mono);color:var(--mute)}.trace .x{color:var(--red)}.trace .ok{color:var(--green)}
.say{font-size:12.5px;color:var(--ink);background:transparent;border-left:3px solid var(--blue);padding:4px 10px;margin:6px 0}
.waiting{color:var(--mute);font-size:12px;font-style:italic}
.call.blocked{border-color:rgba(239,68,68,.45)}.call.executed.danger{border-color:rgba(245,158,11,.5)}
section.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:18px 0}
section.card h2{font-size:16px;margin:0 0 12px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--mute);font-weight:600;font-size:12px}td.num{text-align:right;font-variant-numeric:tabular-nums}
td.bad{color:var(--red);font-weight:700}td.good{color:var(--green);font-weight:700}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:14px}
.kpi{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:12px}
.kpi .v{font-size:24px;font-weight:700}.kpi .l{color:var(--mute);font-size:12px}
.tabs{display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap}.tabs button.on{background:#2563eb;border-color:#2563eb}
.sel{display:flex;gap:6px;margin-bottom:10px}.sel button{padding:6px 10px;font-size:12px}
.sel button.on{border-color:var(--blue);color:var(--blue)}
.bar{display:flex;align-items:center;gap:10px;font-size:12.5px;margin:4px 0}.bar .b{height:14px;background:#2563eb;border-radius:3px}
.bar .k{width:90px;color:var(--mute)}
.mail{border:1px solid var(--line);border-radius:8px;padding:10px;margin:8px 0;background:var(--panel2)}
.mail.ext{border-color:rgba(239,68,68,.5)}.mail pre{white-space:pre-wrap;font:11.5px var(--mono);color:var(--mute);max-height:140px;overflow:auto;margin:6px 0 0}
pre.code{font:12px var(--mono);background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto;max-height:420px}
.hidden{display:none}.empty{color:var(--mute);font-style:italic}
</style></head><body>
<header>
  <div><h1>Run<span>gate</span> — runtime validation for AI agent tool-calls</h1>
  <div class="sub" id="brain">Same agent · same tasks · same company data — without vs. with Rungate</div></div>
  <div class="controls">
    <label class="chk"><input type="checkbox" id="step"> step mode</label>
    <button id="next" class="hidden">Next task ▶</button>
    <button id="runS" class="primary">▶ Run scripted</button>
    <button id="runL">⚡ Run live (LLM)</button>
  </div>
</header>
<main>
  <div class="pipeline">
    <b>Pipeline:</b><span class="st">agent calls gate.call()</span><span class="arr">→</span>
    <span class="st">1 Kill switch</span><span class="arr">→</span><span class="st">2 Policy</span><span class="arr">→</span>
    <span class="st">3 Schema</span><span class="arr">→</span><span class="st">4 Rate limit</span><span class="arr">→</span>
    <span class="st">5 Cost</span><span class="arr">→</span><span class="st">tool runs</span>
    <span class="end">· first failure blocks · every decision audited</span>
  </div>
  <p class="status" id="status">Press <b>Run scripted</b> (deterministic) or <b>Run live</b> (a real LLM decides every tool-call).</p>

  <section class="card" id="dataStart">
    <h2>The company data (Acme Corp)</h2>
    <div id="startSummary" class="empty">Generated fresh on every run.</div>
  </section>

  <div id="runwrap">
  <div class="cols hidden" id="colhead">
    <div class="colhead bad">✖ WITHOUT Rungate — tools run directly</div>
    <div class="colhead good">✔ WITH Rungate — every call goes through the gate</div>
  </div>
  <div id="tasks"></div>
  </div>

  <section class="card hidden" id="report">
    <h2>Damage report</h2>
    <div class="kpis" id="kpis"></div>
    <table id="dmg"></table>
  </section>

  <section class="card hidden" id="explorer">
    <h2>Explore the data</h2>
    <div class="sel" id="snapSel">
      <button data-s="start">At start</button><button data-s="no_gate">After run WITHOUT Rungate</button>
      <button data-s="with_gate" class="on">After run WITH Rungate</button>
    </div>
    <div class="tabs" id="tabs">
      <button data-t="customers" class="on">customers.csv</button><button data-t="orders">orders.db</button>
      <button data-t="outbox">outbox (emails)</button><button data-t="files">files &amp; secrets</button>
      <button data-t="audit">audit log</button><button data-t="policy">policy config</button>
    </div>
    <div id="pane"></div>
  </section>
</main>
<script>
const $ = s => document.querySelector(s), esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let queue = [], timer = null, waitingNext = false, es = null, snaps = {}, audit = [], snapSel = 'with_gate', tab = 'customers';
const DANGER = new Set(['delete_file','send_email','execute_code']);
const fmt = v => v === true ? 'yes' : v === false ? 'no' : (typeof v === 'number' ? v.toLocaleString() : v);

function start(mode){
  if (es) es.close();
  queue = []; snaps = {}; audit = []; $('#tasks').innerHTML=''; ['#report','#explorer'].forEach(s=>$(s).classList.add('hidden'));
  $('#colhead').classList.remove('hidden'); $('#runS').disabled = $('#runL').disabled = true;
  $('#status').textContent = mode === 'live' ? 'Asking the LLM… (each task makes 1–6 model calls)' : 'Running…';
  es = new EventSource('/run?mode=' + mode);
  es.onmessage = m => { const ev = JSON.parse(m.data); queue.push(ev); if (ev.type === 'done' || ev.type === 'error') es.close(); };
  es.onerror = () => es.close();
  if (!timer) timer = setInterval(tick, 420);
}
function tick(){
  if (waitingNext || !queue.length) return;
  const ev = queue.shift(); handle(ev);
  if (ev.type === 'task' && $('#step').checked) { waitingNext = true; $('#next').classList.remove('hidden'); }
}
$('#next').onclick = () => { waitingNext = false; $('#next').classList.add('hidden'); };
$('#runS').onclick = () => start('scripted'); $('#runL').onclick = () => start('live');

function taskEl(i){ return document.getElementById('task'+i); }
function handle(ev){
  if (ev.type === 'start') {
    $('#brain').innerHTML = 'Agent brain: <b>' + esc(ev.brain) + '</b> · same agent, same tasks, same data';
    ev.tasks.forEach((t,k) => {
      const i = k+1, d = document.createElement('div'); d.className='task'; d.id='task'+i;
      d.innerHTML = `<div class="q"><span class="n">${i}</span><span class="badge">${esc(t.agent)}</span><span class="txt">“${esc(t.text)}”</span></div>
        <div class="body"><div class="side" id="s1-${i}"><span class="waiting">waiting…</span></div><div class="side" id="s2-${i}"><span class="waiting">runs after the unprotected pass</span></div></div>`;
      $('#tasks').appendChild(d);
    });
  } else if (ev.type === 'data') {
    snaps[ev.label] = ev.detail; if (ev.label === 'start') renderStart(ev.detail);
  } else if (ev.type === 'phase') {
    $('#status').innerHTML = ev.phase === 1 ? '<b style="color:var(--red)">Pass 1:</b> agent runs WITHOUT Rungate — tools execute directly on the data'
      : '<b style="color:var(--green)">Pass 2:</b> data rebuilt — same agent, every tool-call now goes through Rungate';
  } else if (ev.type === 'task') {
    const s = $(`#s${ev.phase}-${ev.task}`); s.innerHTML=''; taskEl(ev.task).scrollIntoView({behavior:'smooth', block:'center'});
  } else if (ev.type === 'call') {
    const s = $(`#s${ev.phase}-${ev.task}`), c = document.createElement('div');
    const danger = ev.decision === 'executed' && DANGER.has(ev.tool);
    c.className = 'call ' + ev.decision + (danger ? ' danger' : '');
    const args = JSON.stringify(ev.args); const label = {executed:'⚠ EXECUTED', allowed:'✔ ALLOWED', blocked:'✖ BLOCKED', error:'error'}[ev.decision];
    let res = ev.decision === 'blocked' ? `<span class="out">[${esc(ev.reason.toUpperCase())}] ${esc(ev.error)}</span>` : `<span class="out">${esc(ev.output || ev.reason || '')}</span>`;
    let trace = ev.trace ? `<div class="trace">${ev.trace.map(t => { const [k,v] = t.split(':'); return `<span class="${v==='passed'?'ok':'x'}">${esc(k)}${v==='passed'?' ✓':' ✗'}</span>`; }).join(' → ')} · ${ev.ms} ms</div>` : '';
    c.innerHTML = `<div class="sig"><b>${esc(ev.tool)}</b>(${esc(args.length>220?args.slice(0,217)+'…':args)})</div><div class="res"><span class="tag ${ev.decision}">${label}</span>${res}</div>${trace}`;
    s.appendChild(c);
  } else if (ev.type === 'note') {
    const d = document.createElement('div'); d.className='waiting'; d.textContent = '⏳ ' + ev.text; $(`#s${ev.phase}-${ev.task}`).appendChild(d);
  } else if (ev.type === 'say') {
    const d = document.createElement('div'); d.className='say'; d.textContent = '🤖 ' + ev.text; $(`#s${ev.phase}-${ev.task}`).appendChild(d);
  } else if (ev.type === 'report') {
    audit = ev.audit; renderReport(ev); $('#explorer').classList.remove('hidden'); renderPane();
    $('#report').scrollIntoView({behavior:'smooth'});
  } else if (ev.type === 'done') {
    $('#status').innerHTML = 'Done. Scroll through the side-by-side tasks, the damage report and the data explorer.'; reset();
  } else if (ev.type === 'error') { $('#status').innerHTML = '<b style="color:var(--red)">Error:</b> ' + esc(ev.text); reset(); }
}
function reset(){ $('#runS').disabled = $('#runL').disabled = false; $('#next').classList.add('hidden'); waitingNext = false; }

function renderStart(d){
  $('#startSummary').className=''; $('#startSummary').innerHTML = `<div class="kpis">
    <div class="kpi"><div class="v">${d.customers.count}</div><div class="l">customers in customers.csv (name, email, phone, plan)</div></div>
    <div class="kpi"><div class="v">${d.orders.count.toLocaleString()}</div><div class="l">orders in orders.db (SQLite)</div></div>
    <div class="kpi"><div class="v">${d.secrets ? d.secrets.length : 0}</div><div class="l">production secrets in config/prod.env</div></div>
    <div class="kpi"><div class="v">${d.log_lines ?? 0}</div><div class="l">lines in logs/agent.log</div></div></div>
    <div class="sub">Agents: <span class="badge">analyst</span> read-only SQL · <span class="badge">support_bot</span> email @acme.com only · <span class="badge">ops_bot</span> read-only · tools really read, query, delete and email inside <code>demo_workspace/</code>.</div>`;
}
function renderReport(ev){
  const b = ev.before, u = ev.no_gate, w = ev.with_gate, s = ev.stats;
  $('#report').classList.remove('hidden');
  const lost = (b.customers - u.customers) + (b.orders - u.orders) + u.customer_records_leaked;
  $('#kpis').innerHTML = `<div class="kpi"><div class="v" style="color:var(--red)">${lost.toLocaleString()}</div><div class="l">records destroyed or leaked without Rungate</div></div>
    <div class="kpi"><div class="v" style="color:var(--green)">0</div><div class="l">records lost with Rungate</div></div>
    <div class="kpi"><div class="v">${s.blocked_calls} / ${s.total_calls}</div><div class="l">tool-calls blocked by Rungate</div></div>
    <div class="kpi"><div class="v">${s.allowed_calls}</div><div class="l">legitimate calls still allowed</div></div>`;
  const rows = [['Customer records in customers.csv','customers'],['Orders in orders.db','orders'],['logs/agent.log present','log_exists'],
    ['Emails sent outside @acme.com','external_emails'],['Customer records emailed outside','customer_records_leaked']];
  let h = '<tr><th>Check</th><th class="num">At start</th><th class="num">Without Rungate</th><th class="num">With Rungate</th></tr>';
  rows.forEach(([l,k]) => h += `<tr><td>${l}</td><td class="num">${fmt(b[k])}</td><td class="num ${u[k]!==b[k]?'bad':''}">${fmt(u[k])}</td><td class="num ${w[k]!==b[k]?'bad':'good'}">${fmt(w[k])}</td></tr>`);
  h += `<tr><td>Prod DB password revealed to the user</td><td class="num">no</td><td class="num ${ev.leak[0]?'bad':''}">${fmt(ev.leak[0])}</td><td class="num ${ev.leak[1]?'bad':'good'}">${fmt(ev.leak[1])}</td></tr>`;
  $('#dmg').innerHTML = h;
}
document.querySelectorAll('#snapSel button').forEach(x => x.onclick = () => { snapSel = x.dataset.s; document.querySelectorAll('#snapSel button').forEach(y=>y.classList.toggle('on',y===x)); renderPane(); });
document.querySelectorAll('#tabs button').forEach(x => x.onclick = () => { tab = x.dataset.t; document.querySelectorAll('#tabs button').forEach(y=>y.classList.toggle('on',y===x)); renderPane(); });

async function renderPane(){
  const d = snaps[snapSel], p = $('#pane'); $('#snapSel').classList.toggle('hidden', tab==='audit' || tab==='policy');
  if (tab === 'policy') { const t = await (await fetch('/policy')).text(); p.innerHTML = `<div class="sub" style="margin-bottom:8px">The entire security policy for this company — about 40 lines of config.</div><pre class="code">${esc(t)}</pre>`; return; }
  if (tab === 'audit') {
    let h = '<div class="sub" style="margin-bottom:8px">rungate_data_audit.jsonl — one line per tool-call, allowed or blocked (SIEM-ready)</div><table><tr><th>agent</th><th>tool</th><th>arguments</th><th>status</th><th>pipeline</th><th class="num">ms</th></tr>';
    audit.forEach(a => h += `<tr><td><span class="badge">${esc(a.agent)}</span></td><td>${esc(a.tool)}</td><td class="out">${esc(JSON.stringify(a.args).slice(0,90))}</td><td><span class="tag ${a.status.startsWith('blocked')?'blocked':'allowed'}">${esc(a.status)}</span></td><td class="out">${esc(a.pipeline_trace.at(-1))}</td><td class="num">${a.duration_ms.toFixed(2)}</td></tr>`);
    p.innerHTML = h + '</table>'; return;
  }
  if (!d) { p.innerHTML = '<span class="empty">No data yet.</span>'; return; }
  if (tab === 'customers') {
    if (!d.customers.count) { p.innerHTML = '<div class="mail ext"><b style="color:var(--red)">customers.csv does not exist — the agent deleted it.</b></div>'; return; }
    const cols = Object.keys(d.customers.rows[0]);
    let h = `<div class="sub" style="margin-bottom:8px">${d.customers.count} customers · ${Object.entries(d.customers.plans).map(([k,v])=>`${k}: ${v}`).join(' · ')} · first 15 rows</div><table><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr>`;
    d.customers.rows.forEach(r => h += `<tr>${cols.map(c=>`<td>${esc(r[c])}</td>`).join('')}</tr>`); p.innerHTML = h + '</table>';
  } else if (tab === 'orders') {
    const max = Math.max(1, ...d.orders.by_city.map(r=>r[1]));
    p.innerHTML = `<div class="kpis"><div class="kpi"><div class="v">${d.orders.count.toLocaleString()}</div><div class="l">orders</div></div>
      ${Object.entries(d.orders.by_status).map(([k,v])=>`<div class="kpi"><div class="v">${v}</div><div class="l">${k}</div></div>`).join('')}</div>
      <div class="sub" style="margin-bottom:6px">Paid revenue by city (USD)</div>
      ${d.orders.by_city.map(([c,v])=>`<div class="bar"><span class="k">${esc(c)}</span><span class="b" style="width:${Math.round(v/max*420)}px"></span><span>$${Math.round(v).toLocaleString()}</span></div>`).join('')}`;
  } else if (tab === 'outbox') {
    p.innerHTML = d.outbox.length ? d.outbox.map(m => `<div class="mail ${m.external?'ext':''}"><b>To:</b> ${esc(m.to)} ${m.external?'<span class="tag blocked">EXTERNAL</span>':'<span class="tag allowed">internal</span>'} · <b>Subject:</b> ${esc(m.subject)}${m.records?` · <b style="color:var(--red)">${m.records} customer records</b>`:''}<pre>${esc(m.body)}</pre></div>`).join('') : '<span class="empty">Outbox empty.</span>';
  } else if (tab === 'files') {
    const expected = ['config/prod.env','customers.csv','logs/agent.log','orders.db'];
    p.innerHTML = `<table><tr><th>file</th><th>status</th></tr>${expected.map(f=>`<tr><td><code>${f}</code></td>${d.files.includes(f)?'<td class="good">present</td>':'<td class="bad">DELETED</td>'}</tr>`).join('')}</table>
      <div class="sub" style="margin:12px 0 6px">config/prod.env (values masked here)</div><pre class="code">${esc((d.secrets||['(missing)']).join('\n'))}</pre>`;
  }
}
</script></body></html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Rungate web demo")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    url = f"http://localhost:{a.port}"
    print(f"Rungate web demo running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
