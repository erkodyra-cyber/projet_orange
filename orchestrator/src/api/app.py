from __future__ import annotations

import os
import uuid
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import httpx

# -----------------------------------------------------------------------------
# .env (chargé explicitement depuis le dossier orchestrator/)
# -----------------------------------------------------------------------------
_ORCH_ROOT = Path(__file__).resolve().parents[2]  # .../orchestrator
load_dotenv(dotenv_path=_ORCH_ROOT / ".env", override=False)

# -----------------------------------------------------------------------------
# Config ENV
# -----------------------------------------------------------------------------
# Vault
VAULT_ADDR = (os.getenv("VAULT_ADDR") or "http://127.0.0.1:8200").rstrip("/")
VAULT_NAMESPACE = os.getenv("VAULT_NAMESPACE") or None
VAULT_TLS_VERIFY = (os.getenv("VAULT_TLS_VERIFY", "false").lower() in ("1", "true", "yes", "on"))
VAULT_HTTP_TIMEOUT = float(os.getenv("VAULT_HTTP_TIMEOUT_MS", "2500")) / 1000.0
VAULT_LOOKUP_CACHE_TTL = int(os.getenv("VAULT_LOOKUP_CACHE_TTL_SEC", "45"))
REQUIRED_POLICIES = [p.strip() for p in (os.getenv("VAULT_REQUIRED_POLICIES", "sensitive-reader")).split(",") if p.strip()]

# Vault admin (POC) — NE JAMAIS EXPO DANS LE FRONT
VAULT_ADMIN_TOKEN = os.getenv("VAULT_ADMIN_TOKEN") or None

# Admin UI secret (protège les endpoints /admin/* du POC)
ADMIN_UI_SECRET = os.getenv("ADMIN_UI_SECRET") or None

# Agents (endpoints AtoA)
AGENT_PUBLIC_URL = os.getenv("AGENT_PUBLIC_URL", "http://127.0.0.1:8101/atoa")
AGENT_SENSITIVE_URL = os.getenv("AGENT_SENSITIVE_URL", "http://127.0.0.1:8102/atoa")

# Mistral
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral-small-latest")
MISTRAL_TIMEOUT = float(os.getenv("MISTRAL_HTTP_TIMEOUT_S", "45"))
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

def _mask(v: Optional[str], n: int = 6) -> str:
    if not v:
        return "MISSING"
    return f"SET({v[:n]}…)"  # affiche juste le début

print(
    "[BOOT]",
    "VAULT_ADDR=", VAULT_ADDR,
    "| REQUIRED_POLICIES=", REQUIRED_POLICIES,
    "| AGENT_PUBLIC_URL=", AGENT_PUBLIC_URL,
    "| AGENT_SENSITIVE_URL=", AGENT_SENSITIVE_URL,
    "| MISTRAL_API_KEY=", _mask(MISTRAL_API_KEY),
    "| VAULT_ADMIN_TOKEN=", _mask(VAULT_ADMIN_TOKEN),
)

# -----------------------------------------------------------------------------
# Modèles d'E/S (frontend)
# -----------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(..., description="Question utilisateur")
    top_k: int = 10
    initial_k: int = 200
    max_per_source: Optional[int] = None

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]] = []
    used_agents: List[str]
    sensitive_included: bool
    trace_id: str
    notice: Optional[str] = None

# -----------------------------------------------------------------------------
# Auth Vault (lookup-self)
# -----------------------------------------------------------------------------
bearer = HTTPBearer(auto_error=False)

class VaultPrincipal(BaseModel):
    token: str
    policies: List[str]
    meta: Dict[str, Any] = {}
    ttl: Optional[int] = None

_cache: Dict[str, Tuple[float, VaultPrincipal]] = {}

async def _vault_lookup(token: str) -> VaultPrincipal:
    now = time.time()
    cached = _cache.get(token)
    if cached and (now - cached[0] < VAULT_LOOKUP_CACHE_TTL):
        return cached[1]

    headers = {"X-Vault-Token": token}
    if VAULT_NAMESPACE:
        headers["X-Vault-Namespace"] = VAULT_NAMESPACE

    url = f"{VAULT_ADDR}/v1/auth/token/lookup-self"
    async with httpx.AsyncClient(timeout=VAULT_HTTP_TIMEOUT, verify=VAULT_TLS_VERIFY) as client:
        r = await client.get(url, headers=headers)
    if r.status_code != 200:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Vault token")

    data = r.json().get("data", {})
    principal = VaultPrincipal(
        token=token,
        policies=data.get("policies") or [],
        meta=data.get("meta") or {},
        ttl=data.get("ttl"),
    )
    _cache[token] = (now, principal)
    return principal

async def require_vault_token(creds: HTTPAuthorizationCredentials = Depends(bearer)) -> VaultPrincipal:
    if not creds or creds.scheme.lower() != "bearer" or not creds.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    return await _vault_lookup(creds.credentials.strip())

def require_policies(required: List[str]):
    async def dep(principal: VaultPrincipal = Depends(require_vault_token)) -> VaultPrincipal:
        missing = [p for p in required if p not in principal.policies]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing Vault policies: {', '.join(missing)}"
            )
        return principal
    return dep

# -----------------------------------------------------------------------------
# Aides Vault
# -----------------------------------------------------------------------------
def _vault_headers(token: str) -> Dict[str, str]:
    h = {"X-Vault-Token": token}
    if VAULT_NAMESPACE:
        h["X-Vault-Namespace"] = VAULT_NAMESPACE
    return h

def _is_valid_username(u: str) -> bool:
    import re
    return bool(re.fullmatch(r"[A-Za-z0-9._-]{3,32}", u))

# -----------------------------------------------------------------------------
# Aides AtoA + Agents
# -----------------------------------------------------------------------------
def build_atoa_request(req: AskRequest, to_agent: str) -> Dict[str, Any]:
    return {
        "from_agent": "orchestrator",
        "to_agent": to_agent,
        "type": "query",
        "payload": {
            "query": req.question,
            "top_k": req.top_k,
            "initial_k": req.initial_k,
            "max_per_source": req.max_per_source,
        },
    }

async def call_agent(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Ajoute un trace-id utile au debug
        trace_id = payload.get("_trace_id") or str(uuid.uuid4())
        headers = {"X-Trace-Id": trace_id}
        r = await client.post(url, json=payload, headers=headers)
    try:
        return r.json()
    except Exception:
        return {"error": f"bad agent response {r.status_code}", "text": r.text[:500]}

def fuse_naive(agent_payloads: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
    answers: List[str] = []
    sources: List[Dict[str, Any]] = []

    for pl in agent_payloads:
        if not isinstance(pl, dict):
            continue

        payload = pl.get("payload") if isinstance(pl.get("payload"), dict) else {}
        a = payload.get("answer")
        if a is None:
            a = pl.get("answer") or (pl.get("data", {}) if isinstance(pl.get("data"), dict) else {}).get("answer")
        if isinstance(a, str) and a.strip():
            answers.append(a.strip())

        ctx_list = payload.get("contexts", [])
        if not isinstance(ctx_list, list):
            ctx_list = []

        misc_lists: List[Any] = []
        for key in ("sources", "citations", "docs"):
            v = pl.get(key) or payload.get(key)
            if isinstance(v, list):
                misc_lists.extend(v)

        for s in list(ctx_list) + misc_lists:
            if isinstance(s, dict):
                sources.append(s)
            else:
                sources.append({"raw": s})

    final_answer = "\n\n---\n\n".join(answers) if answers else "(Pas de réponse consolidée — voir détails par agent)"
    return final_answer, sources

# -----------------------------------------------------------------------------
# Réécriture Mistral
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "Tu es l'orchestrateur. Tu reçois des réponses partielles de plusieurs agents RAG. "
    "Synthétise une réponse unique, concise et correcte. "
    "S'il y a contradiction, indique l'incertitude et privilégie l'explication la plus probable. "
    "Inclue un court paragraphe 'Sources' avec les références si disponibles. "
    "Réponds en français, clair et structuré."
)

async def rewrite_with_mistral(question: str, agent_payloads: List[Dict[str, Any]]) -> Optional[str]:
    if not MISTRAL_API_KEY:
        return None

    chunks: List[str] = [f"Q: {question}"]
    for i, pl in enumerate(agent_payloads, start=1):
        payload = pl.get("payload") if isinstance(pl.get("payload"), dict) else {}

        a = payload.get("answer")
        if a is None:
            a = pl.get("answer") or (pl.get("data", {}) if isinstance(pl.get("data"), dict) else {}).get("answer")
        if a:
            chunks.append(f"Agent {i} → Réponse:\n{a}")

        ctxs = payload.get("contexts")
        if isinstance(ctxs, list) and ctxs:
            chunks.append(f"Agent {i} → Contexts (JSON): {ctxs}")

    user_prompt = "\n\n".join(chunks)

    headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}
    body = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }
    try:
        async with httpx.AsyncClient(timeout=MISTRAL_TIMEOUT) as client:
            resp = await client.post(MISTRAL_URL, headers=headers, json=body)
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content")
    except Exception:
        return None

# -----------------------------------------------------------------------------
# App FastAPI
# -----------------------------------------------------------------------------
app = FastAPI(title="Orchestrator (Vault + Mistral + AtoA)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # POC
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---- Santé
@app.get("/healthz")
async def healthz():
    return {"ok": True}

# ---- Qui suis-je (via token Vault)
@app.get("/whoami")
async def whoami(p: VaultPrincipal = Depends(require_vault_token)):
    return {"policies": p.policies, "meta": p.meta, "ttl": p.ttl}

# ---- Endpoint sensible protégé
@app.get("/sensitive/ping")
async def sensitive_ping(_p: VaultPrincipal = Depends(require_policies(["sensitive-reader"]))):
    return {"ok": True, "policies": _p.policies}

# ---- Chat /ask
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, p: VaultPrincipal = Depends(require_vault_token)):
    trace_id = str(uuid.uuid4())

    # Public
    public_atoa = build_atoa_request(req, "agent_public")
    public_atoa["_trace_id"] = trace_id
    public_payload = await call_agent(AGENT_PUBLIC_URL, public_atoa)

    used = ["public"]
    sensitive_included = False
    agent_payloads = [public_payload]

    # Sensitive si policies OK
    if REQUIRED_POLICIES and all(pol in p.policies for pol in REQUIRED_POLICIES):
        sensitive_atoa = build_atoa_request(req, "agent_sensitive")
        sensitive_atoa["_trace_id"] = trace_id
        sensitive_payload = await call_agent(AGENT_SENSITIVE_URL, sensitive_atoa)
        agent_payloads.append(sensitive_payload)
        used.append("sensitive")
        sensitive_included = True

    # Fusion
    fallback_answer, sources = fuse_naive(agent_payloads)

    # Mistral
    mistral_answer = await rewrite_with_mistral(req.question, agent_payloads)
    final_answer = mistral_answer or fallback_answer

    notice_parts: List[str] = []
    if not sensitive_included and REQUIRED_POLICIES:
        miss = ", ".join([pol for pol in REQUIRED_POLICIES if pol not in p.policies])
        notice_parts.append(f"Agent sensible non utilisé (policies manquantes: {miss}).")
    if MISTRAL_API_KEY and not mistral_answer:
        notice_parts.append("Reformulation Mistral indisponible, réponse fusionnée naïve.")
    if not MISTRAL_API_KEY:
        notice_parts.append("MISTRAL_API_KEY absente: utilisation de la fusion naïve.")
    notice = " ".join(notice_parts) if notice_parts else None

    return AskResponse(
        answer=final_answer,
        sources=sources,
        used_agents=used,
        sensitive_included=sensitive_included,
        trace_id=trace_id,
        notice=notice,
    )

# -----------------------------------------------------------------------------
# AUTH (Front-friendly) — POC
# -----------------------------------------------------------------------------
@app.post("/auth/signup")
async def auth_signup(
    username: str = Body(..., embed=True),
    password: str = Body(..., embed=True),
):
    """
    POC: crée un utilisateur Vault (userpass) avec policy 'default' uniquement.
    L'ajout de 'sensitive-reader' est réservé à l'admin via /admin/grant-sensitive.
    """
    if not VAULT_ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="Server not configured for signup (missing VAULT_ADMIN_TOKEN)")
    if not _is_valid_username(username):
        raise HTTPException(status_code=400, detail="Invalid username (3-32 chars: letters, digits, . _ -)")
    body = {"password": password, "policies": "default"}
    url = f"{VAULT_ADDR}/v1/auth/userpass/users/{username}"
    async with httpx.AsyncClient(timeout=VAULT_HTTP_TIMEOUT, verify=VAULT_TLS_VERIFY) as client:
        r = await client.post(url, headers=_vault_headers(VAULT_ADMIN_TOKEN), json=body)
    if r.status_code not in (200, 204):
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        raise HTTPException(status_code=r.status_code, detail={"vault_error": err})
    return {"ok": True, "username": username, "policies": ["default"]}

@app.post("/auth/login")
async def auth_login(
    username: str = Body(..., embed=True),
    password: str = Body(..., embed=True),
):
    if not _is_valid_username(username):
        raise HTTPException(status_code=400, detail="Invalid username")
    url = f"{VAULT_ADDR}/v1/auth/userpass/login/{username}"
    async with httpx.AsyncClient(timeout=VAULT_HTTP_TIMEOUT, verify=VAULT_TLS_VERIFY) as client:
        r = await client.post(url, json={"password": password})
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        raise HTTPException(status_code=401, detail={"vault_error": err})
    data = r.json().get("auth", {})
    return {
        "token": data.get("client_token"),
        "policies": data.get("policies"),
        "ttl": data.get("lease_duration"),
        "renewable": data.get("renewable"),
    }

@app.get("/auth/whoami")
async def auth_whoami(p: VaultPrincipal = Depends(require_vault_token)):
    return {"policies": p.policies, "meta": p.meta, "ttl": p.ttl}

# -----------------------------------------------------------------------------
# ADMIN (POC) — nécessite ADMIN_UI_SECRET côté serveur (dans .env)
# -----------------------------------------------------------------------------
def require_admin(secret: Optional[str]) -> None:
    if not ADMIN_UI_SECRET or secret != ADMIN_UI_SECRET:
        raise HTTPException(status_code=403, detail="forbidden")

@app.get("/admin/users")
async def admin_list_users(admin_secret: str):
    """
    Liste les utilisateurs userpass (POC).
    Utilise la méthode LIST de Vault: /v1/auth/userpass/users
    """
    require_admin(admin_secret)
    if not VAULT_ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="missing VAULT_ADMIN_TOKEN")
    url = f"{VAULT_ADDR}/v1/auth/userpass/users"
    # Vault utilise la méthode LIST (HTTP override)
    async with httpx.AsyncClient(timeout=VAULT_HTTP_TIMEOUT, verify=VAULT_TLS_VERIFY) as client:
        r = await client.request("LIST", url, headers=_vault_headers(VAULT_ADMIN_TOKEN))
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        raise HTTPException(status_code=r.status_code, detail={"vault_error": err})
    return r.json()  # ex: {"data":{"keys":["alice","bob"]}}

@app.post("/admin/grant-sensitive")
async def admin_grant_sensitive(admin_secret: str = Body(...), username: str = Body(...)):
    """
    Ajoute la policy 'sensitive-reader' à un utilisateur existant.
    """
    require_admin(admin_secret)
    if not VAULT_ADMIN_TOKEN:
        raise HTTPException(status_code=500, detail="missing VAULT_ADMIN_TOKEN")
    if not _is_valid_username(username):
        raise HTTPException(status_code=400, detail="Invalid username")

    # pour modifier les policies: POST /v1/auth/userpass/users/:name
    url = f"{VAULT_ADDR}/v1/auth/userpass/users/{username}"
    body = {"policies": "default,sensitive-reader"}
    async with httpx.AsyncClient(timeout=VAULT_HTTP_TIMEOUT, verify=VAULT_TLS_VERIFY) as client:
        r = await client.post(url, headers=_vault_headers(VAULT_ADMIN_TOKEN), json=body)
    if r.status_code not in (200, 204):
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        raise HTTPException(status_code=r.status_code, detail={"vault_error": err})
    return {"ok": True, "username": username, "policies": ["default", "sensitive-reader"]}

# -----------------------------------------------------------------------------
# Static front (sert orchestrator/web/index.html)
# -----------------------------------------------------------------------------
WEB_DIR = _ORCH_ROOT / "web"
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
