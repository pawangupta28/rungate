"""
Rungate Audit Logger — Structured logging for every tool-call.

Every call that passes through the gate gets logged with full context:
- Who (agent), what (tool), how (args), when (timestamp)
- Decision (allowed/blocked), why (pipeline trace)
- Performance (duration_ms), cost (estimated)

Storage backends:
- SQLite (default, self-hosted) — zero config, works out of the box
- JSON file — append-only structured logs
- Stdout — for development/debugging
- Custom callback — plug in any storage

Usage:
    logger = AuditLogger({"backend": "sqlite", "path": "rungate_audit.db"})
    logger.log(event)
    recent = logger.query(agent="researcher", limit=10)
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AuditEvent:
    """A single audit log entry."""
    call_id: str = ""
    agent: str = ""
    tool: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    status: str = ""
    error: Optional[str] = None
    duration_ms: float = 0.0
    cost: float = 0.0
    timestamp: float = field(default_factory=time.time)
    pipeline_trace: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["args"] = json.dumps(d["args"], default=str)
        d["pipeline_trace"] = json.dumps(d["pipeline_trace"])
        return d

    def to_json(self) -> str:
        d = asdict(self)
        d["args"] = d["args"]  # Keep as dict for JSON
        return json.dumps(d, default=str)


class AuditBackend:
    """Base class for audit storage backends."""

    def write(self, event: AuditEvent) -> None:
        raise NotImplementedError

    def query(
        self,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def count(
        self,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[float] = None,
    ) -> int:
        raise NotImplementedError

    def close(self) -> None:
        pass


class SQLiteBackend(AuditBackend):
    """SQLite storage backend — default for self-hosted."""

    def __init__(self, path: str = "rungate_audit.db"):
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                tool TEXT NOT NULL,
                args TEXT DEFAULT '{}',
                status TEXT NOT NULL,
                error TEXT,
                duration_ms REAL DEFAULT 0,
                cost REAL DEFAULT 0,
                timestamp REAL NOT NULL,
                pipeline_trace TEXT DEFAULT '[]',
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_tool ON audit_log(tool)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_log(status)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)
        """)
        self._conn.commit()

    def write(self, event: AuditEvent) -> None:
        d = event.to_dict()
        with self._lock:
            self._conn.execute(
                """INSERT INTO audit_log
                   (call_id, agent, tool, args, status, error, duration_ms, cost, timestamp, pipeline_trace)
                   VALUES (:call_id, :agent, :tool, :args, :status, :error, :duration_ms, :cost, :timestamp, :pipeline_trace)""",
                d,
            )
            self._conn.commit()

    def query(
        self,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        conditions = []
        params: Dict[str, Any] = {}
        if agent:
            conditions.append("agent = :agent")
            params["agent"] = agent
        if tool:
            conditions.append("tool = :tool")
            params["tool"] = tool
        if status:
            conditions.append("status = :status")
            params["status"] = status
        if since:
            conditions.append("timestamp >= :since")
            params["since"] = since

        where = " AND ".join(conditions) if conditions else "1=1"
        params["limit"] = limit

        with self._lock:
            cursor = self._conn.execute(
                f"SELECT * FROM audit_log WHERE {where} ORDER BY timestamp DESC LIMIT :limit",
                params,
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def count(
        self,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[float] = None,
    ) -> int:
        conditions = []
        params: Dict[str, Any] = {}
        if agent:
            conditions.append("agent = :agent")
            params["agent"] = agent
        if tool:
            conditions.append("tool = :tool")
            params["tool"] = tool
        if status:
            conditions.append("status = :status")
            params["status"] = status
        if since:
            conditions.append("timestamp >= :since")
            params["since"] = since

        where = " AND ".join(conditions) if conditions else "1=1"

        with self._lock:
            cursor = self._conn.execute(
                f"SELECT COUNT(*) FROM audit_log WHERE {where}", params
            )
            return cursor.fetchone()[0]

    def close(self) -> None:
        self._conn.close()


class JSONFileBackend(AuditBackend):
    """Append-only JSON file backend."""

    def __init__(self, path: str = "rungate_audit.jsonl"):
        self._path = Path(path)
        self._lock = threading.Lock()

    def write(self, event: AuditEvent) -> None:
        line = event.to_json() + "\n"
        with self._lock:
            with open(self._path, "a") as f:
                f.write(line)

    def query(
        self,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if not self._path.exists():
            return results

        with open(self._path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if agent and entry.get("agent") != agent:
                    continue
                if tool and entry.get("tool") != tool:
                    continue
                if status and entry.get("status") != status:
                    continue
                if since and entry.get("timestamp", 0) < since:
                    continue

                results.append(entry)

        # Return most recent first, up to limit
        return list(reversed(results[-limit:]))

    def count(
        self,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[float] = None,
    ) -> int:
        return len(self.query(agent=agent, tool=tool, status=status, since=since, limit=999999))


class StdoutBackend(AuditBackend):
    """Print to stdout — for development."""

    def write(self, event: AuditEvent) -> None:
        status_icon = "✓" if event.status in ("allowed", "executed") else "✗"
        print(
            f"[rungate] {status_icon} agent={event.agent} tool={event.tool} "
            f"status={event.status} cost={event.cost:.4f} "
            f"duration={event.duration_ms:.1f}ms",
            file=sys.stderr,
        )

    def query(self, **kwargs) -> List[Dict[str, Any]]:
        return []

    def count(self, **kwargs) -> int:
        return 0


class CallbackBackend(AuditBackend):
    """Custom callback backend — plug in any storage."""

    def __init__(self, callback: Callable[[AuditEvent], None]):
        self._callback = callback

    def write(self, event: AuditEvent) -> None:
        self._callback(event)


class AuditLogger:
    """
    Main audit logger that dispatches to one or more backends.

    Usage:
        logger = AuditLogger({"backend": "sqlite", "path": "audit.db"})
        logger.log(event)

        # Multiple backends
        logger = AuditLogger({"backends": ["sqlite", "stdout"]})

        # Custom callback
        logger = AuditLogger({"backend": "callback", "callback": my_handler})
    """

    def __init__(self, config: Dict[str, Any]):
        self._backends: List[AuditBackend] = []
        self._setup_backends(config)

    def _setup_backends(self, config: Dict[str, Any]) -> None:
        if not config:
            # Default: stdout in dev
            self._backends.append(StdoutBackend())
            return

        backends = config.get("backends", [config.get("backend", "stdout")])
        if isinstance(backends, str):
            backends = [backends]

        for backend in backends:
            if backend == "sqlite":
                path = config.get("path", "rungate_audit.db")
                self._backends.append(SQLiteBackend(path))
            elif backend == "jsonl" or backend == "json":
                path = config.get("path", "rungate_audit.jsonl")
                self._backends.append(JSONFileBackend(path))
            elif backend == "stdout":
                self._backends.append(StdoutBackend())
            elif backend == "callback":
                callback = config.get("callback")
                if callback and callable(callback):
                    self._backends.append(CallbackBackend(callback))
            elif backend == "none" or backend == "null":
                pass  # No logging

    def log(self, event: AuditEvent) -> None:
        """Log an event to all backends."""
        for backend in self._backends:
            try:
                backend.write(event)
            except Exception:
                # Logging should never crash the pipeline
                pass

    def query(
        self,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query audit logs (uses first queryable backend)."""
        for backend in self._backends:
            try:
                results = backend.query(
                    agent=agent, tool=tool, status=status, since=since, limit=limit
                )
                if results is not None:
                    return results
            except NotImplementedError:
                continue
        return []

    def count(
        self,
        agent: Optional[str] = None,
        tool: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[float] = None,
    ) -> int:
        """Count audit entries (uses first countable backend)."""
        for backend in self._backends:
            try:
                return backend.count(agent=agent, tool=tool, status=status, since=since)
            except NotImplementedError:
                continue
        return 0

    def close(self) -> None:
        """Close all backends."""
        for backend in self._backends:
            try:
                backend.close()
            except Exception:
                pass

    def __repr__(self) -> str:
        backend_names = [type(b).__name__ for b in self._backends]
        return f"AuditLogger(backends={backend_names})"
