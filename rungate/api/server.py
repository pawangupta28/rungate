"""
Rungate API Server — FastAPI-based SaaS API.

Provides a REST API for validating tool-calls remotely. Designed so that
self-hosted users can migrate to SaaS by just adding an api_key parameter.

Start the server:
    rungate-server --port 8000
    # or
    uvicorn rungate.api.server:create_app --factory --port 8000

Environment variables:
    RUNGATE_CONFIG      — Path to YAML config file
    RUNGATE_ADMIN_KEY   — Admin API key (for tenant management)
    RUNGATE_PORT        — Server port (default 8000)
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from rungate.core.interceptor import Rungate
from rungate.api.auth import APIKeyAuth
from rungate.api.middleware import APIMiddleware
from rungate.api.routes import RouteHandler


def create_app(config: Optional[Dict[str, Any]] = None) -> Any:
    """
    Create the Rungate API server application.

    Tries FastAPI first, falls back to a minimal WSGI app.

    Args:
        config: Rungate configuration dict. If None, loads from
                RUNGATE_CONFIG env var or uses defaults.
    """
    # Load config
    if config is None:
        config = _load_config()

    # Initialize core components
    gate = Rungate(config)
    auth = APIKeyAuth()
    middleware = APIMiddleware(auth)
    handler = RouteHandler(gate, middleware)

    # Create admin key if configured
    admin_key = os.environ.get("RUNGATE_ADMIN_KEY", "")
    if admin_key:
        # Register the admin key
        auth.create_key("admin", name="Admin", scopes=["*"])

    try:
        return _create_fastapi_app(gate, auth, middleware, handler)
    except ImportError:
        return _create_minimal_app(gate, auth, middleware, handler)


def _create_fastapi_app(
    gate: Rungate,
    auth: APIKeyAuth,
    middleware: APIMiddleware,
    handler: RouteHandler,
) -> Any:
    """Create a FastAPI application."""
    from fastapi import FastAPI, Request, HTTPException
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Rungate",
        description="Runtime validation API for AI agent tool-calls",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store components on app state
    app.state.gate = gate
    app.state.auth = auth
    app.state.middleware = middleware
    app.state.handler = handler

    async def _get_tenant(request: Request):
        """Extract and validate tenant from request."""
        headers = dict(request.headers)
        tenant, error = middleware.authenticate_request(headers)
        if error:
            status = error.get("error", {}).get("status", 401)
            raise HTTPException(status_code=status, detail=error)
        return tenant

    @app.get("/v1/health")
    async def health():
        status_code, data = handler.health()
        return JSONResponse(content=data, status_code=status_code)

    @app.post("/v1/validate")
    async def validate(request: Request):
        tenant = await _get_tenant(request)
        body = await request.json()
        status_code, data = handler.validate(tenant, body)
        return JSONResponse(content=data, status_code=status_code)

    @app.post("/v1/execute")
    async def execute(request: Request):
        tenant = await _get_tenant(request)
        body = await request.json()
        status_code, data = handler.execute(tenant, body)
        return JSONResponse(content=data, status_code=status_code)

    @app.get("/v1/audit")
    async def audit(request: Request):
        tenant = await _get_tenant(request)
        params = dict(request.query_params)
        status_code, data = handler.audit_query(tenant, params)
        return JSONResponse(content=data, status_code=status_code)

    @app.get("/v1/status")
    async def status(request: Request):
        tenant = await _get_tenant(request)
        status_code, data = handler.status(tenant)
        return JSONResponse(content=data, status_code=status_code)

    @app.post("/v1/kill")
    async def kill(request: Request):
        tenant = await _get_tenant(request)
        body = await request.json()
        status_code, data = handler.kill(tenant, body)
        return JSONResponse(content=data, status_code=status_code)

    @app.post("/v1/unkill")
    async def unkill(request: Request):
        tenant = await _get_tenant(request)
        body = await request.json()
        status_code, data = handler.unkill(tenant, body)
        return JSONResponse(content=data, status_code=status_code)

    return app


def _create_minimal_app(
    gate: Rungate,
    auth: APIKeyAuth,
    middleware: APIMiddleware,
    handler: RouteHandler,
) -> Any:
    """Create a minimal WSGI app (no FastAPI dependency)."""
    import http.server
    import json as _json

    class RungateHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/v1/health":
                code, data = handler.health()
                self._respond(code, data)
            elif self.path.startswith("/v1/audit"):
                tenant, err = self._authenticate()
                if err:
                    return
                code, data = handler.audit_query(tenant, {})
                self._respond(code, data)
            elif self.path == "/v1/status":
                tenant, err = self._authenticate()
                if err:
                    return
                code, data = handler.status(tenant)
                self._respond(code, data)
            else:
                self._respond(404, {"error": "Not found"})

        def do_POST(self):
            tenant, err = self._authenticate()
            if err:
                return

            body = self._read_body()

            if self.path == "/v1/validate":
                code, data = handler.validate(tenant, body)
            elif self.path == "/v1/execute":
                code, data = handler.execute(tenant, body)
            elif self.path == "/v1/kill":
                code, data = handler.kill(tenant, body)
            elif self.path == "/v1/unkill":
                code, data = handler.unkill(tenant, body)
            else:
                code, data = 404, {"error": "Not found"}

            self._respond(code, data)

        def _authenticate(self):
            headers = {k: v for k, v in self.headers.items()}
            tenant, error = middleware.authenticate_request(headers)
            if error:
                status = error.get("error", {}).get("status", 401)
                self._respond(status, error)
                return None, error
            return tenant, None

        def _read_body(self):
            length = int(self.headers.get("Content-Length", 0))
            if length:
                raw = self.rfile.read(length)
                return _json.loads(raw)
            return {}

        def _respond(self, code, data):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(_json.dumps(data).encode())

        def log_message(self, format, *args):
            pass  # Suppress default logging

    return RungateHandler


def _load_config() -> Dict[str, Any]:
    """Load configuration from file or environment."""
    config_path = os.environ.get("RUNGATE_CONFIG", "")

    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        except ImportError:
            # Try JSON
            with open(config_path) as f:
                return json.load(f)

    # Default config
    return {
        "mode": "enforce",
        "policies": {
            "default": {"mode": "deny"},
        },
        "audit": {"backend": "sqlite", "path": "rungate_audit.db"},
    }


def main():
    """CLI entry point for running the server."""
    import argparse

    parser = argparse.ArgumentParser(description="Rungate API Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("RUNGATE_PORT", 8000)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--config", default=os.environ.get("RUNGATE_CONFIG", ""))
    args = parser.parse_args()

    if args.config:
        os.environ["RUNGATE_CONFIG"] = args.config

    try:
        import uvicorn
        app = create_app()
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        # Fallback to stdlib server
        import http.server
        handler_class = create_app()
        server = http.server.HTTPServer((args.host, args.port), handler_class)
        print(f"Rungate API server running on {args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
