"""Group page blocks into articles by reading order and geometry.

Articles start at SectionHeaders, column changes, or large vertical gaps.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# Page furniture, plus Unassigned (text structure.py couldn't confidently
# place in a layout box); never grouped into an article.
NON_ARTICLE_LABELS = {"PageHeader", "PageFooter", "Unassigned"}

_PAGE_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_CONTINUATION_RE = re.compile(
    r"continued\s+(on|from)\s+page\s+([a-z]+|\d+)", re.IGNORECASE
)


def _resolve_page_number(token: str) -> Optional[int]:
    if token.isdigit():
        return int(token)
    return _PAGE_NUMBER_WORDS.get(token.lower())


def _continuation_pages(text: str) -> Dict[str, Optional[int]]:
    """Find "Continued on/from Page N" markers (digit or word form)."""
    on_page: Optional[int] = None
    from_page: Optional[int] = None
    for direction, token in _CONTINUATION_RE.findall(text):
        page_num = _resolve_page_number(token)
        if page_num is None:
            continue
        if direction.lower() == "on":
            on_page = page_num
        else:
            from_page = page_num
    return {"continued_on_page": on_page, "continued_from_page": from_page}


def _x_overlap(a: Dict, b: Dict) -> float:
    return max(
        0.0,
        min(a["bbox"][2], b["bbox"][2]) - max(a["bbox"][0], b["bbox"][0]),
    )


def _same_column(a: Dict, b: Dict, min_overlap: float = 0.35) -> bool:
    a_w = max(a["bbox"][2] - a["bbox"][0], 1.0)
    b_w = max(b["bbox"][2] - b["bbox"][0], 1.0)
    return _x_overlap(a, b) >= min_overlap * min(a_w, b_w)


def _gap_ok(prev: Dict, block: Dict, gap_ratio: float = 0.5, min_gap_px: float = 40.0) -> bool:
    gap = block["bbox"][1] - prev["bbox"][3]
    max_gap = max(min_gap_px, gap_ratio * max(
        prev["bbox"][3] - prev["bbox"][1],
        block["bbox"][3] - block["bbox"][1],
    ))
    return gap <= max_gap


def segment_page(page: Dict, gap_ratio: float = 0.5, min_gap_px: float = 40.0) -> List[Dict]:
    """Segment one page into article records (mutates page['blocks'] too)."""
    blocks = [
        b for b in page["blocks"]
        if b["label"] not in NON_ARTICLE_LABELS
    ]
    blocks.sort(key=lambda b: (b["reading_order"], b["bbox"][1], b["bbox"][0]))

    articles: List[Dict] = []
    current: Dict | None = None
    prev: Dict | None = None

    for block in blocks:
        is_start = current is None
        if block["label"] == "SectionHeader":
            is_start = True
        if prev is not None and not is_start:
            if not _same_column(prev, block) or not _gap_ok(prev, block, gap_ratio, min_gap_px):
                is_start = True

        if is_start:
            current = {
                "article_id": "",
                "title": block["text"]["normalized"].strip() if block["label"] == "SectionHeader" else "Untitled",
                "_blocks": [],
            }
            articles.append(current)
        assert current is not None
        current["_blocks"].append(block)
        prev = block

    out: List[Dict] = []
    for i, art in enumerate(articles):
        bs = art.pop("_blocks")
        art["article_id"] = f"p{page['page_number']}a{i + 1:03d}"
        art["page_number"] = page["page_number"]
        art["issue_id"] = page["issue_id"]
        art["block_ids"] = [b["block_id"] for b in bs]
        art["reading_order_start"] = min(b["reading_order"] for b in bs)
        art["reading_order_end"] = max(b["reading_order"] for b in bs)
        label_counts: Dict[str, int] = {}
        for b in bs:
            label_counts[b["label"]] = label_counts.get(b["label"], 0) + 1
        art["label_counts"] = label_counts
        art["text_normalized"] = "\n\n".join(
            b["text"]["normalized"] for b in bs if b["text"]["normalized"]
        )
        art["word_count"] = len(art["text_normalized"].split())
        art.update(_continuation_pages(art["text_normalized"]))
        art["bbox"] = [
            min(b["bbox"][0] for b in bs),
            min(b["bbox"][1] for b in bs),
            max(b["bbox"][2] for b in bs),
            max(b["bbox"][3] for b in bs),
        ]

        for b in bs:
            b["article_id"] = art["article_id"]
            b["article_title"] = art["title"]
        out.append(art)

    page["articles"] = out
    return out
