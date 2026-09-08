"""Assemble page/issue documents from OCR lines and layout boxes.

Inputs are plain dicts, so the same path runs on live Surya results and on
cached layout.json (rebuild without re-OCR). Also emits blocks.ndjson records
and the index.json manifest.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .normalize import join_lines, normalize_line

SCHEMA_VERSION = 2

NON_TEXT_LABELS = {"Picture", "Figure", "Table", "Equation", "Form", "Code"}

TEXT_LABELS = {
    "Text", "SectionHeader", "ListItem", "Caption",
    "PageHeader", "PageFooter", "Table", "Equation",
}
PICTURE_LABELS = {"Picture", "Figure"}

RELABEL_MIN_LINES = 4

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}


def _is_weekday_token(token: str) -> bool:
    """True for a weekday, written out or abbreviated ("Monday", "Mon.")."""
    t = token.strip().rstrip(".,").lower()
    if len(t) < 3:
        return False
    return any(day.startswith(t) for day in _WEEKDAYS)


_NUMBER_WORDS = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth",
}
_HEADER_STOPWORDS = {
    "of", "and", "for", "vol", "no", "num", "number", "page", "edition",
    "price", "cents", "weather",
} | _WEEKDAYS | set(MONTHS) | _NUMBER_WORDS
_HEADER_STOPWORDS_UPPER = {w.upper() for w in _HEADER_STOPWORDS}

# Archival scan-quality stamps repeat across pages like a real nameplate would
_ARCHIVAL_PHRASES = {"poor document"}

# Bulk-mail indicia
_POSTAL_PHRASES = {
    "presorted", "standard", "first-class", "first class",
    "presorted standard", "presorted first-class",
}

# Structural section labels ("FIRST SECTION", "PART ONE"): real PageHeader text, but not a tagline
_SECTION_LABEL_RE = re.compile(
    r"^(?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|[0-9]+|ONE|TWO|THREE|FOUR|FIVE|SIX)"
    r"\s+SECTION$"
    r"|^(?:SECTION|PART)\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|[0-9]+)$",
    re.IGNORECASE,
)


# URLs/social handles repeat across pages like a real nameplate would
_URL_RE = re.compile(
    r"\b(?:https?://|www\.)\S+"
    r"|\b[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9-]+)*"
    r"\.(?:com|org|net|edu|gov|info|biz|uk|ca|us|au|nz|ie)\b(?:/\S*)?",
    re.IGNORECASE,
)


def _strip_urls(text: str) -> str:
    return re.sub(r"\s+", " ", _URL_RE.sub(" ", text)).strip()


def _is_structural_artifact(text: str) -> bool:
    """True for structural/archival marker rather than real content."""
    t = text.strip().lower()
    return (
        t in _ARCHIVAL_PHRASES
        or t in _POSTAL_PHRASES
        or bool(_SECTION_LABEL_RE.match(text.strip()))
    )


def ocr_lines_from_pydantic(ocr_result) -> List[Dict]:
    out = []
    for line in ocr_result.text_lines:
        words = [
            {"text": w.text, "bbox": [float(v) for v in w.bbox], "confidence": round(float(w.confidence or 0), 4)}
            for w in (line.words or [])
        ]
        out.append({
            "text": line.text,
            "confidence": float(line.confidence or 0),
            "bbox": [float(v) for v in line.bbox],
            "words": words,
        })
    return out


def layout_boxes_from_pydantic(layout_result) -> List[Dict]:
    out = []
    for b in layout_result.bboxes:
        out.append({
            "label": b.label,
            "position": b.position,
            "confidence": float(b.confidence or 0),
            "bbox": [float(v) for v in b.bbox],
            "polygon": [[float(v) for v in pt] for pt in b.polygon],
        })
    return out


def ocr_lines_from_json(text_lines: List[Dict]) -> List[Dict]:
    out = []
    for line in text_lines or []:
        words = [
            {"text": w.get("text", ""), "bbox": w.get("bbox", [0, 0, 0, 0]), "confidence": round(float(w.get("confidence") or 0), 4)}
            for w in (line.get("words") or [])
        ]
        out.append({
            "text": line.get("text", ""),
            "confidence": float(line.get("confidence") or 0),
            "bbox": line.get("bbox", [0, 0, 0, 0]),
            "words": words,
        })
    return out


def layout_boxes_from_json(bboxes: List[Dict]) -> List[Dict]:
    out = []
    for b in bboxes or []:
        out.append({
            "label": b.get("label", "Text"),
            "position": b.get("position", 0),
            "confidence": float(b.get("confidence") or 0),
            "bbox": b.get("bbox", [0, 0, 0, 0]),
            "polygon": b.get("polygon", []),
        })
    return out


def _date_from_filename(issue_id: str) -> Optional[str]:
    suffix = r"(?:_[a-zA-Z]+)?$"
    for sep in ("", "-", r"\.", "_"):
        m = re.search(rf"(\d{{4}}){sep}(\d{{2}}){sep}(\d{{2}}){suffix}", issue_id)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _day_of_week(date: Optional[str]) -> Optional[str]:
    if not date or date == "unknown":
        return None
    try:
        return datetime.strptime(date, "%Y-%m-%d").strftime("%A")
    except ValueError:
        return None


def _denoise_year(token: str) -> Optional[str]:
    """Reconstruct a 4-digit year from OCR-garbled text ('Tl954' -> '1954', 'I95O' -> '1950')."""
    t = (
        token.replace("l", "1").replace("L", "1").replace("I", "1").replace("|", "1")
        .replace("O", "0").replace("o", "0")
    )
    m = re.search(r"(\d{4})", t)
    return m.group(1) if m else None


_DATELINE_PATTERN = (
    r"((?-i:[A-Z])[A-Za-z .\-]{2,})\s*,\s*((?-i:[A-Z])[A-Za-z .\-]{1,20})\s*,\s*"
    r"(?:(?:mon|tues|wednes|thurs|fri|satur|sun)[a-z]*\s*,?\s*)?"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+"
    r"(\d{1,2})([^\n]*)"
)
_DATELINE_RE = re.compile("^" + _DATELINE_PATTERN, re.IGNORECASE | re.MULTILINE)
_DATELINE_RE_ANYWHERE = re.compile(_DATELINE_PATTERN, re.IGNORECASE)

_DATELINE_PATTERN_DMY = (
    r"((?-i:[A-Z])[A-Za-z .\-]{2,})\s*,\s*((?-i:[A-Z])[A-Za-z .\-]{1,20})\s*,\s*"
    r"(?:(?:mon|tues|wednes|thurs|fri|satur|sun)[a-z]*\s*,?\s*)?"
    r"(\d{1,2})(?:st|nd|rd|th)?\s+"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s*"
    r"([^\n]*)"
)
_DATELINE_RE_DMY = re.compile("^" + _DATELINE_PATTERN_DMY, re.IGNORECASE | re.MULTILINE)


def _dateline_from_match(city_raw: str, province_raw: str, day: int, month_token: str, rest: str) -> Optional[Dict]:
    if not (1 <= day <= 31):
        return None
    city = city_raw.strip()
    if city.lower() in _WEEKDAYS:
        return None
    province_raw = province_raw.strip()
    province_words = province_raw.split()
    if province_words and _is_weekday_token(province_words[-1]):
        province_raw = " ".join(province_words[:-1])
    if not province_raw or _is_weekday_token(province_raw):
        return None
    province = (
        re.sub(r"\s+", "", province_raw) if "." in province_raw
        else re.sub(r"\s+", " ", province_raw)
    )
    year = _denoise_year(rest)
    date = None
    if year:
        y = int(year)
        if 1900 <= y <= 2100:
            month = MONTHS[month_token.lower()[:3]]
            date = f"{y}-{month:02d}-{day:02d}"
    return {
        "city": city,
        "province": province,
        "date": date,
        "date_source": "dateline",
    }


def _parse_dateline(text: str) -> Optional[Dict]:
    for m in _DATELINE_RE.finditer(text):
        result = _dateline_from_match(m.group(1), m.group(2), int(m.group(4)), m.group(3), m.group(5))
        if result:
            return result
    for m in _DATELINE_RE_DMY.finditer(text):
        result = _dateline_from_match(m.group(1), m.group(2), int(m.group(3)), m.group(4), m.group(5))
        if result:
            return result
    return None


# A paper with no dateline still has to print a publisher's imprint address somewhere near the nameplate
_IMPRINT_PLACE_RE = re.compile(
    r"\b([A-Z][A-Za-z.'\-]*(?:\s+[A-Z][A-Za-z.'\-]*){0,3})"    # city (1-4 words)
    r"\s*,\s*"
    r"([A-Z]{2})"                                              # state/province code
    r"\s+"
    r"(?:\d{5}(?:-\d{4})?|[A-Z]\d[A-Z]\s?\d[A-Z]\d)"           # ZIP or postal code
    r"\b"
)


def _parse_imprint_place(text: str) -> Optional[Dict]:
    m = _IMPRINT_PLACE_RE.search(text)
    if not m:
        return None
    city = re.sub(r"\s+", " ", m.group(1)).strip().strip(",")
    region = m.group(2).strip()
    if not city or _is_weekday_token(city.split()[-1]):
        return None
    return {"city": city, "province": region}


_REGIONS_PATH = Path(__file__).resolve().parent / "data" / "regions.json"
_region_index: Optional[Dict[str, Optional[str]]] = None


# Normalize a region for lookup
def _region_key(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text or "").upper()


def _load_region_index() -> Dict[str, Optional[str]]:
    global _region_index
    if _region_index is not None:
        return _region_index
    index: Dict[str, Optional[str]] = {}
    try:
        table = json.loads(_REGIONS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        table = {}
    for country, regions in table.items():
        if country.startswith("_") or not isinstance(regions, list):
            continue  # skip the file's own documentation entry
        for region in regions:
            key = _region_key(region)
            if not key:
                continue
            if key in index and index[key] != country:
                # A key claimed by more than one country maps to None (ambiguous)
                index[key] = None
            else:
                index.setdefault(key, country)
    _region_index = index
    return index


def _country_from_region(region: Optional[str]) -> Optional[str]:
    key = _region_key(region or "")
    if not key:
        return None # papers from elsewhere are not assigned a country
    return _load_region_index().get(key)


def _strip_dateline_suffix(text: str) -> str:
    m = _DATELINE_RE_ANYWHERE.search(text)
    if not m:
        return text 
    if _is_weekday_token(m.group(2)):
        return text[:m.start(2)] 
    return text[:m.start()]


def _dateline_suffix(text: str) -> str:
    m = _DATELINE_RE_ANYWHERE.search(text)
    return text[m.start():] if m else ""


def _parse_dateline_anywhere(text: str) -> Optional[Dict]:
    return _parse_dateline(_dateline_suffix(text)) or _parse_dateline(text)


def _date_from_text(text: str) -> Optional[str]:
    top = "\n".join(text.splitlines()[:25]) 
    m = re.search(
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[.,]?\s+"
        r"(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b",
        top.lower(),
    ) 
    if m:
        month = MONTHS[m.group(1)]
        return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}"
    m = re.search(
        r"\b(january|february|march|april|may|june|july|august|september"
        r"|october|november|december)\.?,?\s+(\d{4})\b",
        top.lower(),
    )
    if m:
        return f"{m.group(2)}-{MONTHS[m.group(1)]:02d}"
    return None


def _date_precision(date: Optional[str]) -> Optional[str]:
    if not date or date == "unknown":
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return "day"
    if re.fullmatch(r"\d{4}-\d{2}", date):
        return "month"
    return None


def _parse_volume_issue(text: str) -> Dict:
    m = re.search(
        r"[VY]OL\.?\s*([0-9]+|[IVXLCDM]+)\s*[,.\s]*(?:N[EO]?O\.?|NUM\.?|NUMBER)\s*([0-9]+)",
        text, re.IGNORECASE,
    )
    if m:
        return {"volume": m.group(1), "issue": m.group(2)}
    m = re.search(
        r"N[EO]?O\.?\s*([0-9]+)\s*[—–-]+\s*([0-9]+)\s*(?:ST|ND|RD|TH)\s*YEAR",
        text, re.IGNORECASE,
    )
    if m:
        return {"volume": m.group(2), "issue": m.group(1)}
    m = re.search(r"[VY]OL\.?\s*([0-9]+)\.([0-9]+)\b", text, re.IGNORECASE)
    if m:
        return {"volume": m.group(1), "issue": m.group(2)}
    return {}


_MASTHEAD_DATE_RE = re.compile(
    r"\b(?:" + "|".join(MONTHS) + r")[a-z]*\.?,?\s+"
    r"(?:\d{1,2}(?:st|nd|rd|th)?,?\s+)?\d{4}\b",
    re.IGNORECASE,
)


def extract_masthead(text: str, allow_mixed_case: bool = False) -> Optional[str]:
    best = None
    for line in text.splitlines():
        line = _strip_urls(normalize_line(line))
        line = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9]+$", "", line)
        line = re.sub(r"^\d{1,3}\s+", "", line)
        if len(line) < 8 or len(line) > 110:
            continue
        alnum = sum(c.isalnum() for c in line) / max(len(line), 1)
        upper = sum(c.isupper() or c.isspace() for c in line) / max(len(line), 1)
        if _is_structural_artifact(line):
            continue
        if not (line[0].isupper() or line[0].isdigit()):
            continue
        if alnum > 0.8 and (allow_mixed_case or upper > 0.7) \
                and not _MASTHEAD_DATE_RE.search(line):
            if best is None or len(line) > len(best):
                best = line
    return best


def _header_tokens(text: str) -> List[str]:
    words = re.findall(r"[A-Za-z]{2,}", text.upper())
    return [w for w in words if w not in _HEADER_STOPWORDS_UPPER]


def _titlecase(s: str) -> str:
    return s.strip().title()


_MIN_HEADER_VOTES = 2
_MIN_HEADER_PHRASE_LEN = 4
_MIN_HEADLINE_LEN = 12
_HEADER_SIMILARITY = 0.85


def _phrases_fuzzy_match(a: str, b: str) -> bool:
    from difflib import SequenceMatcher

    wa, wb = a.split(), b.split()
    if len(wa) != len(wb):
        return False
    return all(
        SequenceMatcher(None, x.lower(), y.lower()).ratio() >= _HEADER_SIMILARITY
        for x, y in zip(wa, wb)
    )


def _cluster_header_phrases(counts: Dict[str, int]) -> Dict[str, int]:
    clusters: List[Dict] = []  # [{"rep": str, "count": int}, ...]
    for phrase, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        match = None
        for c in clusters:
            if _phrases_fuzzy_match(phrase, c["rep"]):
                match = c
                break
        if match is None:
            clusters.append({"rep": phrase, "count": n})
        else:
            match["count"] += n
            if len(phrase) > len(match["rep"]):
                match["rep"] = phrase
    return {c["rep"]: c["count"] for c in clusters}


INNER_HEADER_BAND = 0.05


def _unlabeled_header_fallback_text(page: Dict) -> str:
    height = page.get("height") or 0
    if not height:
        return ""
    band_h = height * INNER_HEADER_BAND
    lines = [
        b["text"]["normalized"]
        for b in page.get("blocks", [])
        if b.get("label") == "Text" and b.get("bbox", [0, 0, 0, 0])[1] <= band_h
    ]
    return "\n".join(l for l in lines if l)


def _header_phrase_votes(pages: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for page in pages:
        header_blocks = [b for b in page["blocks"] if b["label"] == "PageHeader"]
        texts = [b["text"]["normalized"] for b in header_blocks]
        if not texts:
            fallback = _unlabeled_header_fallback_text(page)
            if fallback:
                texts = [fallback]
        for raw_text in texts:
            text = _strip_dateline_suffix(_strip_urls(raw_text))
            words = _header_tokens(text)
            if not words:
                continue
            phrase = " ".join(words)
            if _is_structural_artifact(phrase):
                continue
            counts[phrase] = counts.get(phrase, 0) + 1
    return _cluster_header_phrases(counts)


def refine_newspaper(pages: List[Dict], fallback: str) -> tuple[str, str]:
    counts = _header_phrase_votes(pages)
    valid = {
        k: v for k, v in counts.items()
        if v >= _MIN_HEADER_VOTES and len(k) >= _MIN_HEADER_PHRASE_LEN
    }
    if valid:
        best = max(valid.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        return _titlecase(best), "page_headers"
    return fallback, "fallback"


_MIN_EXACT_NAME_LEN = 6


def _place_key(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"[.\s]+", "", value).casefold()


def _cross_issue_newspaper_name(
    pages: List[Dict],
    city: Optional[str],
    province: Optional[str],
    index_path: Path,
) -> Optional[str]:
    confirmed = [
        i for i in load_index(index_path)
        if i.get("newspaper_source") == "page_headers" and i.get("newspaper")
    ]
    if city and province:
        city_key, province_key = _place_key(city), _place_key(province)
        donor_names = {
            i["newspaper"] for i in confirmed
            if _place_key(i.get("city")) == city_key
            and _place_key(i.get("province")) == province_key
        }
        require_exact = False
    else:
        donor_names = {i["newspaper"] for i in confirmed}
        require_exact = True
    if not donor_names:
        return None

    candidates = _header_phrase_votes(pages)
    if not candidates:
        return None

    # Highest-voted candidates first
    for phrase, _n in sorted(candidates.items(), key=lambda kv: -kv[1]):
        for name in donor_names:
            if require_exact:
                if (len(phrase) >= _MIN_EXACT_NAME_LEN
                        and phrase.upper() == " ".join(_header_tokens(name)).upper()):
                    return name
            elif _phrases_fuzzy_match(phrase, name):
                return name
    return None


_FINGERPRINT_MIN_WORDS = 4
_FINGERPRINT_THRESHOLD = 0.5
_FINGERPRINT_STOPWORDS = {"the", "and", "for", "its", "with", "from", "that", "this"}

_CONFIDENT_NAME_SOURCES = {"page_headers", "override"}

_BACKFILLABLE_FIELDS = ("newspaper", "city", "province")


def _masthead_fingerprint(masthead: Optional[str]) -> set:
    if not masthead:
        return set()
    words = {
        w for w in re.sub(r"[^A-Za-z0-9]", " ", masthead.lower()).split()
        if len(w) > 3 and w not in _FINGERPRINT_STOPWORDS
    }
    return words if len(words) >= _FINGERPRINT_MIN_WORDS else set()


def _fingerprints_match(a: set, b: set) -> bool:
    if not a or not b:
        return False
    return len(a & b) / min(len(a), len(b)) >= _FINGERPRINT_THRESHOLD


def _same_publication(a: Dict, b: Dict) -> bool:
    na = (a.get("newspaper") or "").strip()
    nb = (b.get("newspaper") or "").strip()
    if (na and nb
            and a.get("newspaper_source") in _CONFIDENT_NAME_SOURCES
            and b.get("newspaper_source") in _CONFIDENT_NAME_SOURCES
            and (na.lower() == nb.lower() or _phrases_fuzzy_match(na, nb))):
        return True
    return _fingerprints_match(
        _masthead_fingerprint(a.get("masthead")),
        _masthead_fingerprint(b.get("masthead")),
    )


def backfill_from_siblings(meta: Dict, index_path: Path) -> Dict[str, str]:
    siblings = [
        e for e in load_index(index_path)
        if e.get("issue_id") != meta.get("issue_id") and _same_publication(meta, e)
    ]
    if not siblings:
        return {}

    filled: Dict[str, str] = {}
    for field in _BACKFILLABLE_FIELDS:
        if field == "newspaper":
            if meta.get("newspaper_source") != "fallback":
                continue
            values = {
                e["newspaper"] for e in siblings
                if e.get("newspaper")
                and e.get("newspaper_source") in _CONFIDENT_NAME_SOURCES
            }
        else:
            if meta.get(field):
                continue
            values = {e[field] for e in siblings if e.get(field)}
        if len(values) == 1:
            filled[field] = values.pop()
    return filled


def _name_variant_key(name: Optional[str]) -> str:
    n = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    n = re.sub(r"\s+", " ", n).strip()
    return re.sub(r"^(?:the|a|an)\s+", "", n)


def canonical_newspaper_names(entries: List[Dict]) -> Dict[str, str]:
    parent: Dict[str, str] = {e["issue_id"]: e["issue_id"] for e in entries}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            ka, kb = _name_variant_key(a.get("newspaper")), _name_variant_key(b.get("newspaper"))
            if (ka and ka == kb) or _same_publication(a, b):
                union(a["issue_id"], b["issue_id"])

    groups: Dict[str, List[Dict]] = {}
    for e in entries:
        groups.setdefault(find(e["issue_id"]), []).append(e)

    canonical: Dict[str, str] = {}
    for members in groups.values():
        named = [e for e in members if e.get("newspaper_source") != "fallback"]
        counts: Dict[str, int] = {}
        for e in (named or members):
            name = (e.get("newspaper") or "").strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
        if not counts:
            continue
        best = max(counts.items(), key=lambda kv: (kv[1], len(kv[0]), kv[0]))[0]
        for e in members:
            canonical[e["issue_id"]] = best
    return canonical


def _lead_headline(articles: List[Dict]) -> str:
    for art in articles:
        title = art.get("title")
        if not title or title == "Untitled":
            continue
        flat = title.replace("\n", " ").strip()
        if len(flat) < _MIN_HEADLINE_LEN or _is_structural_artifact(flat):
            continue
        return title
    return ""


def _page_header_text(page: Dict) -> str:
    header_blocks = [b for b in page.get("blocks", []) if b.get("label") == "PageHeader"]
    return "\n".join(b["text"]["normalized"] for b in header_blocks)


TOP_BAND_FRACTION = 0.26


def _embedded_text_top_band(page, frac: float = TOP_BAND_FRACTION) -> str:
    """Embedded text from just the top `frac` of the page height."""
    band_h = page.rect.height * frac
    lines = []
    for blk in page.get_text("dict").get("blocks", []):
        for ln in blk.get("lines", []):
            if ln["bbox"][1] <= band_h:
                text = "".join(s["text"] for s in ln["spans"])
                if text.strip():
                    lines.append(text)
    return "\n".join(lines)


_MASTHEAD_MAX_CHARS = 200

MASTHEAD_BAND = 0.18


def _page_top_band_text(
    page: Dict,
    frac: float = TOP_BAND_FRACTION,
    max_block_chars: Optional[int] = None,
) -> str:
    height = page.get("height") or 0
    if not height:
        return ""
    band_h = height * frac
    lines = []
    for b in page.get("blocks", []):
        if b.get("bbox", [0, 0, 0, 0])[1] > band_h:
            continue
        if b.get("label") == "Unassigned":
            continue
        text = b["text"]["normalized"]
        if max_block_chars is not None and len(text) > max_block_chars:
            continue
        if text:
            lines.append(text)
    return "\n".join(lines)


def _resembles_headline(candidate: str, headline: str) -> bool:
    if not headline:
        return False
    a = re.sub(r"\s+", " ", candidate).strip().lower()
    b = re.sub(r"\s+", " ", headline).strip().lower()
    if not a or not b:
        return False
    if a in b or b in a:
        return True
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio() > 0.6


def extract_issue_metadata(
    pdf_path: Path,
    issue_id: str,
    newspaper: Optional[str] = None,
    surya_first_page_text: str = "",
    pages: Optional[List[Dict]] = None,
) -> Dict:
    import fitz

    doc = fitz.open(str(pdf_path))
    try:
        page0 = doc[0] if len(doc) else None
        embedded_first = page0.get_text("text") if page0 else ""
        embedded_top_band = _embedded_text_top_band(page0) if page0 else ""
    finally:
        doc.close()

    first_page = pages[0] if pages else None
    inner_header_texts = [_page_header_text(p) for p in pages[1:]] if pages else []
    inner_header_texts = [t for t in inner_header_texts if t]

    page_header_text = ""
    lead_headline = ""
    if first_page:
        page_header_text = _page_header_text(first_page)
        lead_headline = _lead_headline(first_page.get("articles", []))

    surya_top_band = _page_top_band_text(first_page) if first_page else ""
    dateline = (
        _parse_dateline_anywhere(page_header_text)
        or _parse_dateline_anywhere(embedded_top_band)
        or _parse_dateline_anywhere(surya_top_band)
    )
    if not dateline:
        for text in inner_header_texts:
            dateline = _parse_dateline_anywhere(text)
            if dateline:
                break

    filename_date = _date_from_filename(issue_id)
    if filename_date:
        date, date_source = filename_date, "filename"
    elif dateline and dateline.get("date"):
        date, date_source = dateline["date"], "dateline"
    else:
        date = _date_from_text(embedded_first) or _date_from_text(surya_first_page_text)
        date = date or "unknown"
        date_source = "text"

    masthead = extract_masthead(page_header_text, allow_mixed_case=True) or ""
    for text in inner_header_texts:
        if masthead:
            break
        masthead = extract_masthead(_strip_dateline_suffix(text), allow_mixed_case=True) or ""
    if not masthead and first_page:
        headlines = [
            a.get("title", "") for a in first_page.get("articles", [])
            if a.get("title") and a.get("title") != "Untitled"
        ]
        top_band_text = "\n".join(
            l for l in _page_top_band_text(
                first_page,
                frac=MASTHEAD_BAND,
                max_block_chars=_MASTHEAD_MAX_CHARS,
            ).splitlines()
            if not any(_resembles_headline(l, h) for h in headlines)
        )
        masthead = extract_masthead(top_band_text, allow_mixed_case=True) or ""
    masthead = (
        masthead
        or extract_masthead(embedded_first)
        or extract_masthead(surya_first_page_text)
        or ""
    )
    vol_issue = (
        _parse_volume_issue(embedded_top_band)
        or _parse_volume_issue(embedded_first)
        or _parse_volume_issue(surya_first_page_text)
    )
    if not vol_issue:
        for text in inner_header_texts:
            vol_issue = _parse_volume_issue(text)
            if vol_issue:
                break

    if newspaper is None:
        stem = re.sub(r"_\d{8}$", "", issue_id)
        newspaper = re.sub(r"[_\-]+", " ", stem).strip().title() if stem else "Unknown"
        newspaper_source = "filename"
    else:
        newspaper_source = "override"

    title = lead_headline or (normalize_line(embedded_first.splitlines()[0]) if embedded_first else "")

    meta = {
        "issue_id": issue_id,
        "newspaper": newspaper,
        "newspaper_source": newspaper_source,
        "date": date or "unknown",
        "date_source": date_source,
        "date_precision": _date_precision(date),
        "day_of_week": _day_of_week(date),
        "masthead": masthead,
        "volume": vol_issue.get("volume", ""),
        "issue": vol_issue.get("issue", ""),
        "title": title,
        "source_pdf": pdf_path.name,
    }
    if dateline:
        meta["city"] = dateline["city"]
        meta["province"] = dateline["province"]
        meta["place_source"] = "dateline"
    else:
        imprint = (
            _parse_imprint_place(page_header_text)
            or _parse_imprint_place(surya_top_band)
            or _parse_imprint_place(embedded_top_band)
        )
        meta["city"] = imprint["city"] if imprint else None
        meta["province"] = imprint["province"] if imprint else None
        meta["place_source"] = "imprint" if imprint else None
    meta["country"] = _country_from_region(meta.get("province"))
    return meta


# Block assembly
def _box_area(box_bbox) -> float:
    return max(box_bbox[2] - box_bbox[0], 0) * max(box_bbox[3] - box_bbox[1], 0)


def _line_intersection_frac(line_bbox, box_bbox) -> float:
    x0 = max(line_bbox[0], box_bbox[0])
    y0 = max(line_bbox[1], box_bbox[1])
    x1 = min(line_bbox[2], box_bbox[2])
    y1 = min(line_bbox[3], box_bbox[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    line_area = max(1.0, (line_bbox[2] - line_bbox[0]) * (line_bbox[3] - line_bbox[1]))
    return inter / line_area


_MIN_BOX_OVERLAP = 0.15

_BLEED_OVERSHOOT = 0.3


def _line_bleeds_across_column(line: Dict, box_bbox) -> bool:
    words = line.get("words") or []
    if words:
        x0 = min(w["bbox"][0] for w in words)
        x1 = max(w["bbox"][2] for w in words)
    else:
        x0, x1 = line["bbox"][0], line["bbox"][2]
    box_w = max(box_bbox[2] - box_bbox[0], 1.0)
    left_overshoot = box_bbox[0] - x0
    right_overshoot = x1 - box_bbox[2]
    return max(left_overshoot, right_overshoot) > _BLEED_OVERSHOOT * box_w


def _assign_line_to_box(line_bbox, boxes) -> Optional[int]:
    best_text, best_text_frac = None, 0.0
    best_pic, best_pic_frac = None, 0.0
    for i, box in enumerate(boxes):
        frac = _line_intersection_frac(line_bbox, box["bbox"])
        if frac < _MIN_BOX_OVERLAP:
            continue
        if box["label"] in PICTURE_LABELS:
            if frac > best_pic_frac:
                best_pic, best_pic_frac = i, frac
        else:
            if frac > best_text_frac:
                best_text, best_text_frac = i, frac
    if best_text is not None:
        if best_pic is None or best_text_frac >= 0.5 or best_text_frac >= best_pic_frac - 0.1:
            return best_text
        return best_pic
    return best_pic if best_pic is not None else None


def _serialize_lines(ocr_lines: List[Dict]) -> List[Dict]:
    out = []
    for line in ocr_lines:
        out.append({
            "text": line.get("text", ""),
            "normalized": normalize_line(line.get("text", "")),
            "confidence": round(float(line.get("confidence") or 0), 4),
            "bbox": [float(v) for v in line.get("bbox", [0, 0, 0, 0])],
            "words": line.get("words") or [],
        })
    return out


def _union_bbox(bboxes) -> List[float]:
    if not bboxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def build_page(
    issue_id: str,
    page_number: int,
    image_width: int,
    image_height: int,
    ocr_lines: List[Dict],
    layout_boxes: List[Dict],
    artifacts: Dict[str, str],
) -> Dict:
    """Build the normalized page document from OCR lines + layout boxes (dicts)."""
    lines = sorted(ocr_lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    boxes = sorted(layout_boxes, key=lambda b: b["position"])

    box_lines: Dict[int, List[Dict]] = {i: [] for i in range(len(boxes))}
    unassigned: List[Dict] = []
    for line in lines:
        idx = _assign_line_to_box(line["bbox"], boxes)
        if idx is not None and _line_bleeds_across_column(line, boxes[idx]["bbox"]):
            idx = None
        if idx is not None:
            box_lines[idx].append(line)
        else:
            unassigned.append(line)

    blocks: List[Dict] = []
    relabeled_any = False
    for idx, box in enumerate(boxes):
        block_lines = box_lines[idx]
        label = box["label"]
        relabeled = False
        if label in PICTURE_LABELS and len(block_lines) >= RELABEL_MIN_LINES:
            label = "Text"
            relabeled = True
            relabeled_any = True

        if not block_lines and label not in NON_TEXT_LABELS:
            continue  # text-ish region with no OCR content: skip

        text_original = "\n".join(l["text"] for l in block_lines)
        ocr_conf = (sum(l["confidence"] for l in block_lines) / len(block_lines)
                    if block_lines else 0.0)

        sources = {l.get("source", "surya") for l in block_lines}
        if len(sources) > 1:
            src = "mixed"
        elif sources == {"embedded"}:
            src = "embedded"
        else:
            src = "surya"

        block = {
            "block_id": f"p{page_number}b{idx + 1:03d}",
            "label": label,
            "reading_order": box["position"],
            "bbox": [float(v) for v in box["bbox"]],
            "polygon": [[float(v) for v in pt] for pt in box["polygon"]],
            "layout_confidence": round(float(box["confidence"]), 4),
            "ocr_confidence": round(ocr_conf, 4),
            "source": src,
            "text": {"original": text_original, "normalized": join_lines([l["text"] for l in block_lines])},
            "lines": _serialize_lines(block_lines),
        }
        if relabeled:
            block["relabeled"] = True
        blocks.append(block)

    if unassigned:
        sources = {l.get("source", "surya") for l in unassigned}
        src = "mixed" if len(sources) > 1 else ("embedded" if sources == {"embedded"} else "surya")
        blocks.append({
            "block_id": f"p{page_number}b{len(blocks) + 1:03d}",
            "label": "Unassigned",
            "reading_order": len(boxes),
            "bbox": _union_bbox([l["bbox"] for l in unassigned]),
            "polygon": [],
            "layout_confidence": 0.0,
            "ocr_confidence": round(
                sum(l["confidence"] for l in unassigned) / len(unassigned), 4),
            "source": src,
            "text": {
                "original": "\n".join(l["text"] for l in unassigned),
                "normalized": join_lines([l["text"] for l in unassigned]),
            },
            "lines": _serialize_lines(unassigned),
        })

    blocks.sort(key=lambda b: (b["reading_order"], b["bbox"][1], b["bbox"][0]))

    return {
        "schema_version": SCHEMA_VERSION,
        "issue_id": issue_id,
        "page_number": page_number,
        "width": image_width,
        "height": image_height,
        "image": artifacts.get("image", ""),
        "blocks": blocks,
        "articles": [],
    }


def compute_coverage(surya_text: str, embedded_text: str) -> Dict:
    surya_chars = len(surya_text.strip())
    embedded_chars = len(embedded_text.strip())
    ratio = surya_chars / embedded_chars if embedded_chars else 1.0
    return {
        "surya_chars": surya_chars,
        "embedded_chars": embedded_chars,
        "ratio": round(ratio, 3),
        "low": ratio < 0.5,
    }


def link_continuations(pages: List[Dict]) -> None:
    articles_by_page: Dict[int, List[Dict]] = {p["page_number"]: p["articles"] for p in pages}
    for page in pages:
        for art in page["articles"]:
            target_page = art.get("continued_on_page")
            if not target_page or target_page not in articles_by_page:
                continue
            candidates = [
                other for other in articles_by_page[target_page]
                if other.get("continued_from_page") == page["page_number"]
            ]
            if len(candidates) == 1:
                art["continues_in_article_id"] = candidates[0]["article_id"]
                candidates[0]["continues_from_article_id"] = art["article_id"]


def flatten_blocks(page: Dict, issue_meta: Dict) -> List[Dict]:
    records = []
    for block in page["blocks"]:
        records.append({
            "block_id": block["block_id"],
            "issue_id": page["issue_id"],
            "newspaper": issue_meta.get("newspaper", ""),
            "issue_date": issue_meta.get("date", ""),
            "page_number": page["page_number"],
            "label": block["label"],
            "reading_order": block["reading_order"],
            "bbox": block["bbox"],
            "layout_confidence": block["layout_confidence"],
            "ocr_confidence": block["ocr_confidence"],
            "source": block.get("source", "surya"),
            "article_id": block.get("article_id", ""),
            "article_title": block.get("article_title", ""),
            "text_original": block["text"]["original"],
            "text_normalized": block["text"]["normalized"],
        })
    return records


def write_index(issues: List[Dict], index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": SCHEMA_VERSION,
        "issues": issues,
    }
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


def load_index(index_path: Path) -> List[Dict]:
    if not index_path.exists():
        return []
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return data.get("issues", [])
    except Exception:
        return []


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()