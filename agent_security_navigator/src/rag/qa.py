import os
import pickle
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# --- Config
INDEX_DIR = Path("agent_security_navigator/index")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "mistral-small-latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-base")

# Limites & timeouts
CTX_CHAR_LIMIT = int(os.getenv("CTX_CHAR_LIMIT", "1200"))  # tronquage par passage
MISTRAL_TIMEOUT_S = float(os.getenv("MISTRAL_TIMEOUT_S", "45"))
MISTRAL_RETRIES = int(os.getenv("MISTRAL_RETRIES", "1"))  # 0 = pas de retry
FAISS_NPROBE = int(os.getenv("FAISS_NPROBE", "10"))  # utile si IVF

# --------- Chargement des ressources ---------
def load_resources() -> Dict[str, Any]:
    store_p = INDEX_DIR / "store.pkl"
    faiss_p = INDEX_DIR / "vectors.faiss"
    if not store_p.exists() or not faiss_p.exists():
        raise FileNotFoundError("Index absent. Lance pdf_to_chunks.py puis build_index.py")

    with open(store_p, "rb") as f:
        store = pickle.load(f)

    index = faiss.read_index(str(faiss_p))

    # Réglage nprobe si applicable (IVF*)
    try:
        if hasattr(index, "nprobe"):
            index.nprobe = FAISS_NPROBE
    except Exception:
        pass

    embed_model = SentenceTransformer(EMBED_MODEL)

    texts: List[str] = store.get("texts", [])
    metas: List[dict] = store.get("metas", [{} for _ in texts])
    # sécurise la longueur metas = longueur texts
    if len(metas) != len(texts):
        # réaligne
        m2 = [{} for _ in texts]
        for i, m in enumerate(metas[: len(texts)]):
            m2[i] = m or {}
        metas = m2

    return {
        "index": index,
        "texts": texts,
        "metas": metas,
        "embed_model": embed_model,
        "embed_model_name": EMBED_MODEL,
    }

# --------- Embeddings ---------
def _l2(v: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(v, axis=1, keepdims=True)
    denom[denom == 0] = 1e-12
    return v / denom

def embed_query(q: str, model: SentenceTransformer, name: str) -> np.ndarray:
    # E5: préfixe "query:"
    if "e5" in (name or "").lower():
        q = f"query: {q}"
    v = model.encode([q], convert_to_numpy=True).astype("float32")
    return _l2(v)

# --------- Retrieval ---------
def _dedup_key(text: str, meta: dict) -> str:
    src = (meta or {}).get("source", "")
    page = (meta or {}).get("page", "")
    return f"{src}|{page}|{(text or '')[:80]}"

def _truncate_text(t: str) -> str:
    if not t:
        return ""
    if len(t) <= CTX_CHAR_LIMIT:
        return t
    return t[:CTX_CHAR_LIMIT] + " ..."

def retrieve(
    question: str,
    k: int,
    initial_k: int,
    max_per_source: Optional[int],
    state: Dict[str, Any],
) -> List[Dict[str, Any]]:
    texts: List[str] = state["texts"]
    metas: List[dict] = state["metas"]
    index = state["index"]

    if not texts:
        return []  # corpus vide

    # garde: initial_k >= k et <= taille corpus
    if initial_k < k:
        initial_k = k
    initial_k = min(initial_k, len(texts))

    # embed & search
    qv = embed_query(question, state["embed_model"], state["embed_model_name"])
    scores, idxs = index.search(qv, initial_k)

    # candidats triés par score (faiss les renvoie déjà triés), assemblage
    candidates: List[Dict[str, Any]] = []
    for s, i in zip(scores[0], idxs[0]):
        if i < 0:
            continue
        meta = metas[i] or {}
        candidates.append(
            {"score": float(s), "text": texts[i], "meta": meta}
        )

    # déduplication légère + cap par source
    out: List[Dict[str, Any]] = []
    seen = set()
    per_src: Dict[str, int] = {}

    for c in candidates:
        text = c.get("text") or ""
        meta = c.get("meta") or {}
        key = _dedup_key(text, meta)
        if key in seen:
            continue
        seen.add(key)

        if max_per_source and max_per_source > 0:
            src = meta.get("source", "document")
            if per_src.get(src, 0) >= max_per_source:
                continue
            per_src[src] = per_src.get(src, 0) + 1

        c["text"] = _truncate_text(text)
        out.append(c)
        if len(out) >= k:
            break

    return out

# --------- Prompt construction ---------
def build_messages(question: str, passages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not passages:
        sys = (
            "Tu es un assistant RAG. Aucun extrait n'est disponible.\n"
            "- Réponds honnêtement : indique que les extraits sont absents.\n"
            "- Sois bref, en français."
        )
        user = f"Question : {question}\n\nAucun extrait.\n"
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    # Affiche [i] + conserve la possibilité d’insérer [doc,page] via le LLM si meta fournie
    blocks = []
    for i, p in enumerate(passages):
        meta = p.get("meta") or {}
        doc = meta.get("source", "document")
        page = meta.get("page")
        tag = f"[{i+1}]"
        ref = f" ({doc}{', p.'+str(page) if page not in (None, '') else ''})"
        blocks.append(f"{tag} {p.get('text','')}{ref}")

    ctx = "\n\n".join(blocks)

    sys = (
        "Tu es un assistant RAG sur les rapports Security Navigator.\n"
        "- Utilise UNIQUEMENT les extraits fournis.\n"
        "- À chaque fois que tu utilises une info d’un extrait, ajoute entre crochets la référence [doc,page].\n"
        "- Si l'information exacte manque, dis-le clairement (« D'après les extraits, ... »).\n"
        "- Réponds en français, clair et concis (5–8 lignes)."
    )
    user = (
        f"Question : {question}\n\n"
        f"EXTRAITS :\n{ctx}\n\n"
        "Consignes :\n- Appuie chaque affirmation sur les extraits. Si c'est partiel, dis-le."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

# --------- LLM call ---------
def _post_with_retries(url: str, headers: Dict[str, str], json_payload: Dict[str, Any], timeout_s: float, retries: int) -> requests.Response:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=json_payload, timeout=timeout_s)
            return r
        except Exception as e:
            last_exc = e
            # backoff léger
            time.sleep(0.25 * (attempt + 1))
    # raise pour que l’agent remonte une erreur claire (catché en amont si besoin)
    raise RuntimeError(f"mistral_request_failed: {last_exc}")

def _extract_mistral_text(j: Dict[str, Any]) -> str:
    # Mistral renvoie généralement choices[0].message.content
    try:
        return (j["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        # fallback sur un éventuel format alternatif
        return (j.get("output_text") or "").strip()

def call_mistral(messages: List[Dict[str, str]], trace_id: Optional[str] = None) -> str:
    if not MISTRAL_API_KEY:
        return "⚠️ MISTRAL_API_KEY manquante."

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json",
        **({"X-Trace-Id": trace_id} if trace_id else {}),
    }
    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "temperature": 0.2,
    }

    r = _post_with_retries(url, headers, payload, timeout_s=MISTRAL_TIMEOUT_S, retries=MISTRAL_RETRIES)

    if r.status_code >= 400:
        # retourne un message exploitable mais non bloquant
        try:
            body = r.json()
        except Exception:
            body = {"text": r.text}
        return f"Erreur Mistral {r.status_code}: {body}"

    j = r.json()
    answer = _extract_mistral_text(j)
    # Optionnel : usage (tokens), si tu veux l’exposer plus tard
    # usage = j.get("usage", {})  # {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}
    return answer or "(réponse vide)"
