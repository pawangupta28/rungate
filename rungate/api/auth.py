"""
Rungate API Authentication — API key management and tenant isolation.

Simple API key auth for the SaaS layer. Each key maps to a tenant
with its own policies, budgets, and audit trail.

Usage:
    auth = APIKeyAuth()
    auth.create_key("tenant_acme", scopes=["validate", "audit"])

    # In request handler:
    tenant = auth.authenticate(request_api_key)
    if tenant is None:
        return 401
"""

from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Tenant:
    """A Rungate SaaS tenant."""
    id: str
    name: str = ""
    api_key_hash: str = ""
    scopes: Set[str] = field(default_factory=lambda: {"validate", "audit", "config"})
    rate_limit: int = 1000  # Requests per minute
    created_at: float = field(default_factory=time.time)
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class APIKeyAuth:
    """
    API key authentication and tenant management.

    Keys are stored as SHA-256 hashes — the plaintext key is returned
    only once at creation time.
    """

    def __init__(self):
        self._tenants: Dict[str, Tenant] = {}
        self._key_to_tenant: Dict[str, str] = {}  # key_hash → tenant_id
        self._lock = threading.Lock()

    def create_key(
        self,
        tenant_id: str,
        name: str = "",
        scopes: Optional[List[str]] = None,
        rate_limit: int = 1000,
    ) -> str:
        """
        Create a new API key for a tenant.

        Returns the plaintext API key (shown only once).
        """
        api_key = f"rg_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(api_key)

        tenant = Tenant(
            id=tenant_id,
            name=name or tenant_id,
            api_key_hash=key_hash,
            scopes=set(scopes) if scopes else {"validate", "audit", "config"},
            rate_limit=rate_limit,
        )

        with self._lock:
            self._tenants[tenant_id] = tenant
            self._key_to_tenant[key_hash] = tenant_id

        return api_key

    def authenticate(self, api_key: str) -> Optional[Tenant]:
        """
        Authenticate an API key and return the tenant.

        Returns None if the key is invalid or the tenant is inactive.
        """
        if not api_key:
            return None

        key_hash = self._hash_key(api_key)

        with self._lock:
            tenant_id = self._key_to_tenant.get(key_hash)
            if tenant_id is None:
                return None

            tenant = self._tenants.get(tenant_id)
            if tenant is None or not tenant.active:
                return None

            return tenant

    def has_scope(self, tenant: Tenant, scope: str) -> bool:
        """Check if a tenant has a specific scope."""
        return scope in tenant.scopes or "*" in tenant.scopes

    def revoke_key(self, tenant_id: str) -> bool:
        """Revoke a tenant's API key."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                return False

            # Remove key mapping
            self._key_to_tenant.pop(tenant.api_key_hash, None)
            tenant.active = False
            return True

    def rotate_key(self, tenant_id: str) -> Optional[str]:
        """Rotate a tenant's API key. Returns the new plaintext key."""
        with self._lock:
            tenant = self._tenants.get(tenant_id)
            if tenant is None:
                return None

            # Remove old key
            self._key_to_tenant.pop(tenant.api_key_hash, None)

            # Generate new key
            new_key = f"rg_{secrets.token_urlsafe(32)}"
            new_hash = self._hash_key(new_key)

            tenant.api_key_hash = new_hash
            self._key_to_tenant[new_hash] = tenant_id

            return new_key

    def list_tenants(self) -> List[Dict[str, Any]]:
        """List all tenants (without key hashes)."""
        with self._lock:
            return [
                {
                    "id": t.id,
                    "name": t.name,
                    "active": t.active,
                    "scopes": sorted(t.scopes),
                    "rate_limit": t.rate_limit,
                    "created_at": t.created_at,
                }
                for t in self._tenants.values()
            ]

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash an API key with SHA-256."""
        return hashlib.sha256(key.encode()).hexdigest()

    def __repr__(self) -> str:
        active = sum(1 for t in self._tenants.values() if t.active)
        return f"APIKeyAuth(tenants={len(self._tenants)}, active={active})"
