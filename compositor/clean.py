"""Optional LLM-based OCR cleanup per article."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3.5:9b"

_SYSTEM = (
    "You correct OCR errors in digitized historical newspaper text.\n"
    "Rules:\n"
    "1. Fix only obvious OCR mistakes: garbled or split words, stray characters,\n"
    "   spacing errors, stray fragments (e.g. ': irst' -> 'first').\n"
    "2. Preserve every name, number, date and fact exactly. Never guess or invent.\n"
    "3. Do not add, remove, summarize or reorder content.\n"
    "4. Keep paragraph breaks. Do not add any preamble or commentary.\n"
    "   Begin directly with the corrected text.\n"
    "When TWO readings are provided (primary + secondary), reconcile word by word:\n"
    "   choose the reading that forms a real word or makes sense in context;\n"
    "   otherwise keep the primary reading.\n"
)

# Leading preamble lines the model may add despite instructions
_PREAMBLE = re.compile(
    r"^(?:here(?:'s| is| are| you go)|below is|the corrected|corrected text|"
    r"output:?|response:?|sure(?:, here|,)?|ok(?:ay)?|result:?)[:\-]?\s*$",
    re.IGNORECASE,
)


def correct_text(text: str, model: str = DEFAULT_MODEL, ollama_url: str = DEFAULT_OLLAMA_URL, timeout: int = 300, embedded_alt: str = "") -> str:
    """Correct OCR errors in `text` using the local Ollama model.

    `embedded_alt` (optional) is the PDF embedded-text reading of the same
    region; when provided the model reconciles the two readings, which resolves
    words where one OCR engine beat the other (e.g. a brand name).
    """
    import requests

    if embedded_alt and len(embedded_alt.strip()) >= 20:
        prompt = (
            f"{_SYSTEM}\n"
            f"Primary OCR reading:\n{text}\n"
            f"\nSecondary embedded-layer reading:\n{embedded_alt}\n"
            f"\nCorrected text:"
        )
    else:
        prompt = f"{_SYSTEM}\nText:\n{text}"

    n_tokens = max(1500, int(len(text.split()) * 1.6) + 600)
    resp = requests.post(
        f"{ollama_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "think": False,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": n_tokens},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    corrected = resp.json().get("response", "")
    corrected = re.sub(r"^```(?:text|txt)?\s*", "", corrected.strip(), flags=re.I)
    corrected = re.sub(r"\s*```$", "", corrected).strip()
    # Drop any short preamble line(s) the model prepended
    lines = corrected.splitlines()
    while lines and _PREAMBLE.match(lines[0].strip()):
        lines.pop(0)
    corrected = "\n".join(lines).strip()
    if len(corrected) < max(5, len(text) // 3):
        raise ValueError("LLM response too short to be a faithful correction")
    # Small local models under greedy decoding can fall into a repetition
    # loop and regenerate the same paragraph until num_predict runs out.
    # 1.4x sits between the longest legit correction (1.26x) and the
    # shortest observed repetition loop (1.52x) in a real test run.
    if len(corrected) > max(80, int(len(text) * 1.4)):
        raise ValueError(
            "LLM response too long relative to the original (likely a repetition loop)"
        )
    return corrected


def _load_embedded_lines_for_page(doc, page_index: int, page_width_px: float) -> list:
    page = doc[page_index]
    scale = page_width_px / page.rect.width if page.rect.width else 1.0
    out = []
    for blk in page.get_text("dict").get("blocks", []):
        for ln in blk.get("lines", []):
            text = "".join(s["text"] for s in ln["spans"])
            if not text.strip():
                continue
            out.append({"text": text, "bbox": [v * scale for v in ln["bbox"]]})
    return out


def _embedded_alt_for_article(pages: list, doc, article: dict, page_lines_cache: dict) -> str:
    page = next((p for p in pages if p.get("page_number") == article.get("page_number")), None)
    if not page or doc is None:
        return ""
    boxes = [b["bbox"] for b in page.get("blocks", [])
             if b.get("block_id") in (article.get("block_ids") or [])
             and b.get("label") != "Unassigned"]
    if not boxes:
        return ""
    page_index = article["page_number"] - 1
    if page_index not in page_lines_cache:
        page_lines_cache[page_index] = _load_embedded_lines_for_page(doc, page_index, page["width"])
    sel = []
    for l in page_lines_cache[page_index]:
        cx = (l["bbox"][0] + l["bbox"][2]) / 2
        cy = (l["bbox"][1] + l["bbox"][3]) / 2
        for bb in boxes:
            if bb[0] - 10 <= cx <= bb[2] + 10 and bb[1] - 10 <= cy <= bb[3] + 10:
                sel.append(l)
                break
    sel.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    return "\n".join(l["text"] for l in sel)


def clean_issue(
    out_dir: Path,
    source_pdf=None,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    force: bool = False,
    min_words: int = 15,
) -> Path:
    """Correct all articles in an issue, caching to corrections.json."""
    articles_path = out_dir / "articles.ndjson"
    if not articles_path.exists():
        raise FileNotFoundError(f"{articles_path} not found (run process/rebuild first)")

    corrections_path = out_dir / "corrections.json"
    corrections = {}
    if corrections_path.exists():
        corrections = json.loads(corrections_path.read_text(encoding="utf-8"))

    articles = [
        json.loads(line)
        for line in articles_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    correctable = {
        a["article_id"] for a in articles
        if a.get("word_count", 0) >= min_words and (a.get("text_normalized") or "").strip()
    }
    corrections = {k: v for k, v in corrections.items() if k in correctable}

    pages = []
    for page_json in sorted((out_dir / "pages").glob("page_*/page.json")):
        pages.append(json.loads(page_json.read_text(encoding="utf-8")))

    doc = None
    if source_pdf and Path(source_pdf).exists():
        import fitz
        doc = fitz.open(str(source_pdf))
    page_lines_cache: dict = {}

    try:
        done = skipped = 0
        for art in articles:
            aid = art["article_id"]
            if aid in corrections and not force:
                skipped += 1
                continue
            text = art.get("text_normalized", "")
            if not text.strip() or art.get("word_count", 0) < min_words:
                skipped += 1
                continue
            alt = _embedded_alt_for_article(pages, doc, art, page_lines_cache)
            try:
                corrected = correct_text(text, model=model, ollama_url=ollama_url, embedded_alt=alt)
            except Exception as e:
                corrections.pop(aid, None)
                print(f"  [warn] {aid}: LLM failed ({e}); leaving uncorrected")
                continue
            corrections[aid] = {
                "corrected_text": corrected,
                "model": model,
                "corrected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            done += 1
            print(f"  {aid}: {art.get('word_count', 0)} words -> corrected")
            time.sleep(0.1)
    finally:
        if doc is not None:
            doc.close()

    corrections_path.write_text(
        json.dumps(corrections, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Clean: {done} corrected, {skipped} skipped/cached -> {corrections_path}")
    return corrections_path


def load_corrections(out_dir: Path) -> dict:
    p = out_dir / "corrections.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}