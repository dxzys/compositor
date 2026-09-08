"""Merge the PDF's embedded text layer with Surya OCR lines."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

MIN_EMBEDDED_LINES = 10 # Min number of clean embedded lines for a page to be considered usable

MIN_ALPHA_RATIO = 0.25 # Min letter ratio for an embedded line to be considered real text
TEXT_REGION_MARGIN = 40.0 # px
LONG_LINE_FRACTION = 0.5 # A line is "long" if wider than this fraction of the median line width
SURYA_MIN_CONF = 0.35 # Surya lines below this confidence are unreliable


def _alpha_proxy(text: str) -> float:
    if not text:
        return 0.0
    alnum = sum(c.isalnum() for c in text)
    return round(max(0.3, alnum / len(text)), 4)


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isalpha() for c in text) / len(text)


# Chars treated as "ordinary" punctuation
_ORDINARY = re.compile(r"[\w\s.,;:'\"\-()]")


def _is_repetitive(text: str) -> bool:
    words = re.findall(r"[A-Za-z]+", text)
    if len(words) < 8:
        return False
    tris = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if not tris:
        return False
    top = Counter(tris).most_common(1)[0][1]
    return top >= 3 and top >= 0.3 * len(tris)


def _is_garbage(text: str) -> bool:
    s = re.sub(r"^[.\s]+|[.\s]+$", "", text.strip())
    if len(s) < 4:
        return True
    if _alpha_ratio(s) < MIN_ALPHA_RATIO:
        return True
    if not re.search(r"[A-Za-z]{3,}", s):
        return True
    specials = len(_ORDINARY.sub("", s))
    if specials > 0.3 * len(s):
        return True
    return _is_repetitive(s)


def extract_embedded_lines(pdf_path, scale: float = 1.0) -> List[List[Dict]]:
    import fitz

    doc = fitz.open(str(pdf_path))
    pages: List[List[Dict]] = []
    try:
        for page in doc:
            lines: List[Dict] = []
            data = page.get_text("dict")
            for blk in data.get("blocks", []):
                if "lines" not in blk:
                    continue
                for ln in blk["lines"]:
                    text = "".join(s["text"] for s in ln["spans"])
                    if not text.strip():
                        continue
                    bbox = [v * scale for v in ln["bbox"]]
                    lines.append({
                        "text": text,
                        "bbox": bbox,
                        "confidence": _alpha_proxy(text),
                        "source": "embedded",
                        "words": [],
                    })
            pages.append(lines)
    finally:
        doc.close()
    return pages


def scale_lines(lines: List[Dict], factor: float) -> List[Dict]:
    for l in lines:
        l["bbox"] = [v * factor for v in l["bbox"]]
    return lines


def _x_center(b) -> float:
    return (b[0] + b[2]) / 2


def _median(values):
    if not values:
        return 0.0
    return sorted(values)[len(values) // 2]


def _text_region(lines: List[Dict]) -> Optional[Tuple[float, float]]:
    widths = [l["bbox"][2] - l["bbox"][0] for l in lines]
    med = _median(widths)
    if med <= 0:
        return None
    long_lines = [l for l in lines if (l["bbox"][2] - l["bbox"][0]) > LONG_LINE_FRACTION * med]
    if not long_lines:
        return None
    x0 = min(l["bbox"][0] for l in long_lines) - TEXT_REGION_MARGIN
    x1 = max(l["bbox"][2] for l in long_lines) + TEXT_REGION_MARGIN
    return (x0, x1)


def clean_embedded_lines(lines: List[Dict]) -> List[Dict]:
    region = _text_region(lines)
    if region is None:
        return []
    x0, x1 = region
    return [
        l for l in lines
        if not _is_garbage(l.get("text", "")) and x0 <= _x_center(l["bbox"]) <= x1
    ]


def _surya_fits_slot(sb: List[float], eb: List[float], eh_cap: float = 60.0) -> bool:
    syc = (sb[1] + sb[3]) / 2
    eyc = (eb[1] + eb[3]) / 2
    eh = min(eb[3] - eb[1], eh_cap)
    if abs(syc - eyc) > 0.6 * max(eh, 20.0):
        return False
    ew = max(eb[2] - eb[0], 30.0)
    sc = (sb[0] + sb[2]) / 2
    ec = (eb[0] + eb[2]) / 2
    if abs(sc - ec) > 0.5 * ew:
        return False
    if (sb[2] - sb[0]) > 1.5 * max(ew, 30.0):
        return False
    return True


def _overlaps(a, b) -> bool:
    y_over = min(a[3], b[3]) - max(a[1], b[1])
    x_over = min(a[2], b[2]) - max(a[0], b[0])
    return y_over > 0 and x_over > 0


def _drop_unreliable_surya(surya_lines: List[Dict]) -> List[Dict]:
    return [
        s for s in surya_lines
        if (s.get("confidence") or 0) >= SURYA_MIN_CONF and not _is_repetitive(s.get("text", ""))
    ]


def merge_line_sources(
    embedded_lines: List[Dict],
    surya_lines: List[Dict],
) -> List[Dict]:
    clean = clean_embedded_lines(embedded_lines)
    if len(clean) < MIN_EMBEDDED_LINES:
        return _drop_unreliable_surya(surya_lines)

    slots = [dict(l) for l in clean]
    slots.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))

    surya = _drop_unreliable_surya(surya_lines)
    med_h = _median([s["bbox"][3] - s["bbox"][1] for s in slots])
    eh_cap = max(1.5 * med_h, 30.0)

    used = set()
    for s in surya:
        best_idx, best_wr = None, None
        for i, slot in enumerate(slots):
            if i in used or not _surya_fits_slot(s["bbox"], slot["bbox"], eh_cap):
                continue
            wr = (s["bbox"][2] - s["bbox"][0]) / max(slot["bbox"][2] - slot["bbox"][0], 1.0)
            if best_wr is None or wr < best_wr:
                best_wr, best_idx = wr, i
        if best_idx is not None:
            slots[best_idx].update({
                "text": s["text"],
                "confidence": round(float(s.get("confidence") or 0.5), 4),
                "source": "surya",
                "words": s.get("words") or [],
            })
            used.add(best_idx)

    med_w = _median([s["bbox"][2] - s["bbox"][0] for s in slots])
    region = _text_region(clean)
    for s in surya:
        if s["bbox"][2] - s["bbox"][0] > 1.5 * med_w:
            continue
        if any(_overlaps(s["bbox"], slot["bbox"]) for slot in slots):
            continue
        if _is_garbage(s.get("text", "")):
            continue
        if region is not None and not (region[0] <= _x_center(s["bbox"]) <= region[1]):
            continue
        x = dict(s)
        x["source"] = "surya"
        x.setdefault("words", [])
        slots.append(x)

    slots.sort(key=lambda s: (s["bbox"][1], s["bbox"][0]))
    return slots