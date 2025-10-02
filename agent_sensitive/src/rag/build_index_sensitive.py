import json, pickle
from pathlib import Path
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

PROCESSED = Path("agent_sensitive/data_sensitive/processed/chunks.jsonl")
INDEX_DIR = Path("agent_sensitive/index_sensitive")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
EMBED_MODEL = "intfloat/multilingual-e5-base"

def l2(x: np.ndarray)->np.ndarray:
    if x.ndim != 2 or x.size == 0:
        raise ValueError("Vecteurs vides : lance pdf_to_chunks_sensitive.py et vérifie les PDFs.")
    n = np.linalg.norm(x, axis=1, keepdims=True); n[n==0]=1e-12; return x/n

def main():
    texts, metas = [], []
    if not PROCESSED.exists():
        raise FileNotFoundError(f"{PROCESSED} introuvable. Lance d’abord pdf_to_chunks_sensitive.py")

    with PROCESSED.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            t = (j.get("text") or "").strip()
            if not t:
                continue
            texts.append(t)
            metas.append({"source": j.get("source"), "page": j.get("page")})

    if not texts:
        raise ValueError("Aucun passage utilisable dans le .jsonl")

    model = SentenceTransformer(EMBED_MODEL)
    inp = [f"passage: {t}" for t in texts]
    vecs = model.encode(inp, convert_to_numpy=True).astype("float32")
    vecs = l2(vecs)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(INDEX_DIR / "vectors.faiss"))
    with open(INDEX_DIR / "store.pkl", "wb") as f:
        pickle.dump({"texts": texts, "metas": metas}, f)
    print(f"OK: {len(texts)} passages sensibles indexés → {INDEX_DIR}")
if __name__ == "__main__":
    main()
