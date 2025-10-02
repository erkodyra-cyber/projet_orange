import json
from pathlib import Path
from pypdf import PdfReader

RAW_DIR = Path("agent_security_navigator/data/raw")
OUT     = Path("agent_security_navigator/data/processed/chunks.jsonl")
OUT.parent.mkdir(parents=True, exist_ok=True)

def chunk_text(t: str, size=900, overlap=150):
    t = " ".join((t or "").split())
    start = 0
    while start < len(t):
        yield t[start:start+size]
        start += max(1, size - overlap)

def main():
    count_pdf, count_chunks = 0, 0
    with OUT.open("w", encoding="utf-8") as w:
        for pdf in RAW_DIR.glob("*.pdf"):
            count_pdf += 1
            reader = PdfReader(str(pdf))
            for i, page in enumerate(reader.pages, start=1):
                txt = page.extract_text() or ""
                for ch in chunk_text(txt):
                    w.write(json.dumps({"source": pdf.name, "page": i, "text": ch}, ensure_ascii=False) + "\n")
                    count_chunks += 1
    print(f"PDF trouvés: {count_pdf} | Chunks générés: {count_chunks}")
    if count_pdf == 0:
        print(f"⚠️  Aucun PDF dans {RAW_DIR.resolve()}")
    if count_chunks == 0:
        print("⚠️  Aucun texte extrait (PDF scannés ?)")
if __name__ == "__main__":
    main()
