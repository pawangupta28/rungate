"""
Rungate API Middleware — Request-level auth, rate limiting, and error handling.

Extracts API keys, resolves tenants, enforces request-level rate limits,
and provides consistent error responses.
"""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from typing import Any, Dict, Optional

from rungate.api.auth import APIKeyAuth, Tenant


class RequestRateLimiter:
    """Per-tenant request rate limiter for the API layer."""

    def __init__(self):
        self._windows: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, tenant_id: str, limit: int, window: float = 60.0) -> bool:
        """Check if a request is allowed under the tenant's rate limit."""
        with self._lock:
            now = time.monotonic()
            timestamps = self._windows[tenant_id]

            # Prune old entries
            cutoff = now - window
            self._windows[tenant_id] = [t for t in timestamps if t > cutoff]

            if len(self._windows[tenant_id]) >= limit:
                return False

            self._windows[tenant_id].append(now)
            return True

    def remaining(self, tenant_id: str, limit: int, window: float = 60.0) -> int:
        """Get remaining requests for a tenant."""
        with self._lock:
            now = time.monotonic()
            cutoff = now - window
            timestamps = [t for t in self._windows.get(tenant_id, []) if t > cutoff]
            return max(0, limit - len(timestamps))


class APIMiddleware:
    """
    Request processing middleware for the Rungate API.

    Handles:
    - API key extraction and authentication
    - Tenant resolution
    - Request-level rate limiting
    - Error response formatting
    """

    def __init__(self, auth: APIKeyAuth):
        self._auth = auth
        self._rate_limiter = RequestRateLimiter()

    def authenticate_request(self, headers: Dict[str, str]) -> tuple:
        """
        Authenticate a request from its headers.

        Returns:
            (tenant, None) on success
            (None, error_dict) on failure
        """
        api_key = self._extract_api_key(headers)
        if not api_key:
            return None, self.error_response(401, "Missing API key. Set Authorization: Bearer <key>")

        tenant = self._auth.authenticate(api_key)
        if tenant is None:
            return None, self.error_response(401, "Invalid or revoked API key")

        # Check request rate limit
        if not self._rate_limiter.check(tenant.id, tenant.rate_limit):
            remaining = self._rate_limiter.remaining(tenant.id, tenant.rate_limit)
            return None, self.error_response(
                429,
                f"Rate limit exceeded. Limit: {tenant.rate_limit}/min, remaining: {remaining}",
                headers={"Retry-After": "60"},
            )

        return tenant, None

    def check_scope(self, tenant: Tenant, scope: str) -> Optional[Dict]:
        """Check if tenant has required scope. Returns error dict or None."""
        if not self._auth.has_scope(tenant, scope):
            return self.error_response(403, f"Insufficient scope. Required: {scope}")
        return None

    @staticmethod
    def _extract_api_key(headers: Dict[str, str]) -> Optional[str]:
        """Extract API key from Authorization header."""
        auth = headers.get("Authorization", headers.get("authorization", ""))
        if auth.startswith("Bearer "):
            return auth[7:].strip()
        # Also support X-API-Key header
        return headers.get("X-API-Key", headers.get("x-api-key", "")) or None

    @staticmethod
    def error_response(
        status: int,
        message: str,
        details: Optional[Dict] = None,
        headers: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a standard error response."""
        resp: Dict[str, Any] = {
            "error": {
                "status": status,
                "message": message,
            }
        }
        if details:
            resp["error"]["details"] = details
        if headers:
            resp["_headers"] = headers
        return resp

    @staticmethod
    def success_response(data: Any, status: int = 200) -> Dict[str, Any]:
        """Create a standard success response."""
        return {
            "status": status,
            "data": data,
        }
