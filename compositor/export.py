"""Write a RAG-ready corpus.jsonl from processed issues.

Reads articles.ndjson (falling back to blocks.ndjson) and emits minimal rows —
text, metadata, page, coordinates, image path. Full per-issue artifacts are
left in place.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def _row_from_article(art: dict, issue_meta: dict) -> dict:
    return {
        "type": "article",
        "article_id": art["article_id"],
        "issue_id": issue_meta["issue_id"],
        "newspaper": issue_meta.get("newspaper", ""),
        "newspaper_canonical": issue_meta.get("newspaper_canonical", ""),
        "city": issue_meta.get("city", ""),
        "province": issue_meta.get("province", ""),
        "country": issue_meta.get("country") or "",
        "date": issue_meta.get("date", ""),
        "day_of_week": issue_meta.get("day_of_week"),
        "date_precision": issue_meta.get("date_precision"),
        "page_number": art.get("page_number", 0),
        "title": art.get("title", "Untitled"),
        "text_normalized": art.get("text_normalized", ""),
        "word_count": art.get("word_count", 0),
        "block_ids": art.get("block_ids", []),
        "bbox": art.get("bbox", []),
        "image": f"pages/page_{art.get('page_number', 0):03d}/page.png",
        "source": "surya",
        # Set only when structure.py::link_continuations confidently matched
        # the other half of a split story; None otherwise.
        "continues_in_article_id": art.get("continues_in_article_id"),
        "continues_from_article_id": art.get("continues_from_article_id"),
        "continued_on_page": art.get("continued_on_page"),
        "continued_from_page": art.get("continued_from_page"),
    }


def _row_from_block(block: dict, issue_meta: dict) -> dict:
    return {
        "type": "block",
        "block_id": block["block_id"],
        "issue_id": block["issue_id"],
        "newspaper": block.get("newspaper", ""),
        "newspaper_canonical": issue_meta.get("newspaper_canonical", ""),
        "city": issue_meta.get("city", ""),
        "province": issue_meta.get("province", ""),
        "country": issue_meta.get("country") or "",
        "date": block.get("issue_date", ""),
        "page_number": block.get("page_number", 0),
        "title": block.get("article_title", ""),
        "label": block.get("label", ""),
        "text_normalized": block.get("text_normalized", ""),
        "word_count": len(block.get("text_normalized", "").split()),
        "bbox": block.get("bbox", []),
        "image": f"pages/page_{block.get('page_number', 0):03d}/page.png",
        "source": block.get("source", "surya"),
    }


def export_corpus(
    index_path: Path,
    output_dir: Path,
    corpus_dir: Path,
    min_word_count: int = 15,
    use_corrected: bool = False,
) -> Path:
    from .clean import load_corrections
    from .structure import canonical_newspaper_names, load_index

    issues = load_index(index_path)
    # A publication can be indexed under two spellings if two issues read
    # their own nameplate differently; canonical is added, not substituted.
    canonical = canonical_newspaper_names(issues)
    for issue in issues:
        issue["newspaper_canonical"] = canonical.get(
            issue["issue_id"], issue.get("newspaper", "")
        )
    corpus_dir.mkdir(parents=True, exist_ok=True)
    out_path = corpus_dir / "corpus.jsonl"

    stats = {"issues": 0, "articles": 0, "blocks": 0, "words": 0, "skipped": 0}
    with open(out_path, "w", encoding="utf-8") as f:
        for issue in issues:
            issue_dir = output_dir / issue["issue_id"]
            articles_path = issue_dir / "articles.ndjson"
            blocks_path = issue_dir / "blocks.ndjson"
            corrections = load_corrections(issue_dir) if use_corrected else {}

            if articles_path.exists():
                for line in articles_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    art = json.loads(line)
                    art["page_number"] = int(art.get("page_number", 0))
                    if art.get("word_count", 0) < min_word_count:
                        stats["skipped"] += 1
                        continue
                    row = _row_from_article(art, issue)
                    if use_corrected and art["article_id"] in corrections:
                        row["text_normalized"] = corrections[art["article_id"]]["corrected_text"]
                        row["corrected"] = True
                        row["word_count"] = len(row["text_normalized"].split())
                    f.write(json.dumps(row) + "\n")
                    stats["articles"] += 1
                    stats["words"] += row["word_count"]
            elif blocks_path.exists():
                for line in blocks_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    block = json.loads(line)
                    if len(block.get("text_normalized", "").split()) < min_word_count:
                        stats["skipped"] += 1
                        continue
                    f.write(json.dumps(_row_from_block(block, issue)) + "\n")
                    stats["blocks"] += 1
                    stats["words"] += len(block.get("text_normalized", "").split())
            stats["issues"] += 1

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stats": stats,
        "row_count": stats["articles"] + stats["blocks"],
    }
    (corpus_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out_path