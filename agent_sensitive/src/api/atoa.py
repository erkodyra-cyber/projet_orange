import os, sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

THIS = Path(__file__).resolve()
AGENT_ROOT = THIS.parents[2]
PROJECT_ROOT = AGENT_ROOT.parent
PROTOS = PROJECT_ROOT / "protocols"
for p in [PROJECT_ROOT, PROTOS]:
    if str(p) not in sys.path: sys.path.insert(0, str(p))

from mcp_atoa import AtoARequest, MCPResponse, MCPContext

# charge .env de l’agent sensible avant d’importer le RAG (même logique que public)
load_dotenv(AGENT_ROOT / ".env")

# ---- RAG sensible (miroir de l’agent public)
from agent_sensitive.src.rag.qa_sensitive import load_resources, retrieve, build_messages, call_mistral

app = FastAPI(title="agent_sensitive (AtoA→MCP)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# charge à démarrage (et plante proprement si index absent)
try:
    STATE = load_resources()
    STARTUP_ERROR = None
except Exception as e:
    STATE = None
    STARTUP_ERROR = str(e)

@app.get("/health")
def health():
    return {"status": "ok" if STARTUP_ERROR is None else "error", "error": STARTUP_ERROR, "name":"agent_sensitive"}

@app.post("/atoa", response_model=MCPResponse)
def atoa(req: AtoARequest):
    if STARTUP_ERROR is not None:
        return MCPResponse(protocol="MCP", type="error", status="error",
                           from_agent="agent_sensitive", to_agent=req.from_agent,
                           error={"code":"index_missing","message":STARTUP_ERROR})

    q = req.payload.query
    k = req.payload.top_k or 10
    ik = req.payload.initial_k or 200
    mps = req.payload.max_per_source if (req.payload.max_per_source and req.payload.max_per_source > 0) else None

    passages = retrieve(q, k, ik, mps, STATE)
    msgs = build_messages(q, passages)
    answer = call_mistral(msgs)

    ctx = []
    for p in passages:
        p.get("meta", {}).update({"agent":"sensitive"})
        ctx.append(MCPContext(score=float(p["score"]), text=p.get("text"), meta=p.get("meta", {})).dict())

    return MCPResponse(from_agent="agent_sensitive", to_agent=req.from_agent, payload={"answer": answer, "contexts": ctx})
import os, sys, json
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Header
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

THIS = Path(__file__).resolve()
AGENT_ROOT = THIS.parents[2]
PROJECT_ROOT = AGENT_ROOT.parent
PROTOS = PROJECT_ROOT / "protocols"
for p in [PROJECT_ROOT, PROTOS]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from mcp_atoa import AtoARequest, MCPResponse, MCPContext

# ---- Config .env (celui de l'agent SENSIBLE)
load_dotenv(AGENT_ROOT / ".env")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
CORS_ORIGINS = ["*"] if CORS_ORIGINS == ["*"] else CORS_ORIGINS
MAX_CONTEXTS_RETURNED = int(os.getenv("MAX_CONTEXTS_RETURNED", "200"))

# ---- RAG sensible (miroir du public mais corpus sensible)
from agent_sensitive.src.rag.qa_sensitive import (
    load_resources, retrieve, build_messages, call_mistral
)

app = FastAPI(title="agent_sensitive")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Startup: charge l'index et expose l'erreur au /health
try:
    STATE = load_resources()
    STARTUP_ERROR = None
except Exception as e:
    STATE = None
    STARTUP_ERROR = str(e)

def _model_dump(obj) -> Dict[str, Any]:
    return obj.model_dump() if hasattr(obj, "model_dump") else obj.dict()

def _warn_ctx(text: str) -> Dict[str, Any]:
    # Warning léger renvoyé sous forme d'un contexte "meta"
    return MCPContext(score=0.0, text="", meta={"agent": "sensitive", "warning": text})

@app.get("/health")
def health():
    return {
        "status": "ok" if STARTUP_ERROR is None else "error",
        "error": STARTUP_ERROR,
        "name": "agent_sensitive"
    }

@app.get("/capabilities")
def capabilities():
    # pour que l’orchestrateur sache quoi envoyer
    return {
        "name": "agent_sensitive",
        "version": "1.0.0",
        "supports": {
            "top_k": True,
            "initial_k": True,
            "max_per_source": True
        }
    }

@app.post("/atoa", response_model=MCPResponse)
def atoa(req: AtoARequest, x_trace_id: Optional[str] = Header(default=None)):
    if STARTUP_ERROR is not None:
        return MCPResponse(
            protocol="MCP", type="error", status="error",
            from_agent="agent_sensitive", to_agent=req.from_agent,
            error={"code": "index_missing", "message": STARTUP_ERROR}
        )

    q = req.payload.query
    k = (req.payload.top_k or 10)
    ik = (req.payload.initial_k or 200)
    # garde : initial_k >= top_k
    warnings: List[str] = []
    if ik < k:
        ik = k
        warnings.append(f"initial_k < top_k → initial_k corrigé à {k}")

    mps = req.payload.max_per_source if (req.payload.max_per_source and req.payload.max_per_source > 0) else None

    # --- retrieve
    try:
        passages = retrieve(q, k, ik, mps, STATE)
    except Exception as e:
        return MCPResponse(
            protocol="MCP", type="error", status="error",
            from_agent="agent_sensitive", to_agent=req.from_agent,
            error={"code": "retrieve_error", "message": str(e)}
        )

    # --- prompt build + LLM
    try:
        msgs = build_messages(q, passages)
    except Exception as e:
        return MCPResponse(
            protocol="MCP", type="error", status="error",
            from_agent="agent_sensitive", to_agent=req.from_agent,
            error={"code": "prompt_build_error", "message": str(e)}
        )

    try:
        # support x_trace_id si la fonction l'accepte (miroir du public)
        answer = call_mistral(msgs, trace_id=x_trace_id) if "trace_id" in call_mistral.__code__.co_varnames else call_mistral(msgs)
    except Exception as e:
        return MCPResponse(
            protocol="MCP", type="error", status="error",
            from_agent="agent_sensitive", to_agent=req.from_agent,
            error={"code": "llm_error", "message": str(e)}
        )

    # --- contexts (cap & pas de mutation in-place)
    ctx: List[Dict[str, Any]] = []
    for p in passages[:MAX_CONTEXTS_RETURNED]:
        meta = dict(p.get("meta") or {})
        meta.setdefault("agent", "sensitive")
        try:
            ctx_obj = MCPContext(
                score=float(p.get("score", 0.0)),
                text=p.get("text", ""),
                meta=meta
            )
            ctx.append(_model_dump(ctx_obj))
        except Exception:
            # On skippe les mauvais items au lieu de casser toute la réponse
            continue

    # ajouter warnings éventuels dans un contexte méta
    for w in warnings:
        ctx.append(_model_dump(_warn_ctx(w)))

    return MCPResponse(
        from_agent="agent_sensitive",
        to_agent=req.from_agent,
        payload={"answer": answer, "contexts": ctx}
    )
