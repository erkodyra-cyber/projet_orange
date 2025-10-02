# orchestrator/src/security/vault_auth.py
from __future__ import annotations
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx, os, time
from typing import Optional, Dict, Any, Tuple

bearer = HTTPBearer(auto_error=False)

class VaultPrincipal(BaseModel):
    token: str
    policies: list[str]
    meta: Dict[str, Any] = {}
    ttl: Optional[int] = None

# Petit cache en mémoire
_cache: Dict[str, Tuple[float, VaultPrincipal]] = {}

def _env_bool(name: str, default: bool=False) -> bool:
    v = os.getenv(name)
    if v is None: return default
    return v.lower() in ("1","true","yes","on")

VAULT_ADDR = os.getenv("VAULT_ADDR", "http://localhost:8200").rstrip("/")
VAULT_NAMESPACE = os.getenv("VAULT_NAMESPACE") or None
VAULT_TLS_VERIFY = _env_bool("VAULT_TLS_VERIFY", False)
VAULT_TIMEOUT = int(os.getenv("VAULT_HTTP_TIMEOUT_MS", "2500"))
CACHE_TTL = int(os.getenv("VAULT_LOOKUP_CACHE_TTL_SEC", "45"))
REQUIRED_POLICIES = [p.strip() for p in os.getenv("VAULT_REQUIRED_POLICIES","").split(",") if p.strip()]

async def _lookup_token(token: str) -> VaultPrincipal:
    now = time.time()
    cached = _cache.get(token)
    if cached and (now - cached[0] < CACHE_TTL):
        return cached[1]

    headers = {"X-Vault-Token": token}
    if VAULT_NAMESPACE:
        headers["X-Vault-Namespace"] = VAULT_NAMESPACE

    url = f"{VAULT_ADDR}/v1/auth/token/lookup-self"
    async with httpx.AsyncClient(timeout=VAULT_TIMEOUT/1000, verify=VAULT_TLS_VERIFY) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Vault token")

    data = r.json().get("data", {})
    policies = data.get("policies") or []
    meta = data.get("meta") or {}
    ttl = data.get("ttl")

    principal = VaultPrincipal(token=token, policies=policies, meta=meta, ttl=ttl)
    _cache[token] = (now, principal)
    return principal

async def require_vault_token(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> VaultPrincipal:
    if not creds or creds.scheme.lower() != "bearer" or not creds.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    token = creds.credentials.strip()
    return await _lookup_token(token)

def require_policies(required: list[str]):
    async def dep(principal: VaultPrincipal = Depends(require_vault_token)) -> VaultPrincipal:
        missing = [p for p in required if p not in principal.policies]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required Vault policies: {', '.join(missing)}"
            )
        return principal
    return dep
