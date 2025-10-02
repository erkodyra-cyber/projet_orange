# protocols/mcp_atoa.py
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

# Compat Pydantic v1/v2 pour extra="forbid"
try:
    from pydantic import ConfigDict
    _MODEL_CFG = dict(model_config=ConfigDict(extra="forbid"))
except Exception:
    class _Cfg: extra = "forbid"
    _MODEL_CFG = dict(Config=_Cfg)

class AtoAQueryPayload(BaseModel):
    query: str
    top_k: Optional[int] = Field(default=10, ge=1, le=100)
    initial_k: Optional[int] = Field(default=200, ge=1, le=5000)
    max_per_source: Optional[int] = Field(default=None, ge=0)  # None/0 = pas de cap
    protocol_version: int = 1
    model_config = _MODEL_CFG


class AtoARequest(BaseModel):
    from_agent: str
    to_agent: str
    payload: AtoAQueryPayload
    model_config = _MODEL_CFG


class MCPContext(BaseModel):
    score: float = 0.0
    text: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)
    model_config = _MODEL_CFG


class MCPResponse(BaseModel):
    protocol: str = "MCP"
    type: str = "data"   # "data" | "error"
    status: str = "ok"   # "ok" | "error"
    from_agent: Optional[str] = None
    to_agent: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None  # {"answer": str, "contexts": [MCPContext]}
    error: Optional[Dict[str, Any]] = None    # {"code": str, "message": str}
    model_config = _MODEL_CFG


class Capabilities(BaseModel):
    name: str
    version: str
    supports: Dict[str, bool]  # {"top_k": True, "initial_k": True, "max_per_source": True}
    model_config = _MODEL_CFG

