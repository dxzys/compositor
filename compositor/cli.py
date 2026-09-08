"""Command-line interface: process, batch, rebuild, group, stats, export, clean."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
INDEX_PATH = ROOT / "index.json"
CORPUS_DIR = ROOT / "corpus"

from .article import segment_page
from .export import export_corpus
from .layout import detect_layout
from .normalize import join_lines
from .ocr import detect_ocr
from .pdf import extract_embedded_text, pdf_page_sizes_pt, render_pages, save_page_images
from .structure import (
    canonical_newspaper_names,
    _country_from_region,
    _cross_issue_newspaper_name,
    backfill_from_siblings,
    build_page,
    compute_coverage,
    extract_issue_metadata,
    flatten_blocks,
    layout_boxes_from_json,
    layout_boxes_from_pydantic,
    link_continuations,
    load_index,
    now_iso,
    ocr_lines_from_json,
    ocr_lines_from_pydantic,
    refine_newspaper,
    write_index,
)
from .text_source import extract_embedded_lines, merge_line_sources, scale_lines


# Page assembly helpers
def _finalize_page(page: dict, embedded_text: str) -> None:
    """Add coverage metrics and article annotations."""
    surya_text = " ".join(b["text"]["normalized"] for b in page["blocks"])
    page["coverage"] = compute_coverage(surya_text, embedded_text)
    segment_page(page)


def _write_page_artifacts(page: dict, page_dir: Path, layout_raw: dict) -> None:
    (page_dir / "layout.json").write_text(
        json.dumps(layout_raw, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (page_dir / "page.json").write_text(
        json.dumps(page, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _assemble_issue_outputs(
    issue_id: str,
    pages: list,
    issue_meta: dict,
    out_dir: Path,
) -> None:
    link_continuations(pages)

    block_records = [
        rec for page in pages for rec in flatten_blocks(page, issue_meta)
    ]
    with open(out_dir / "blocks.ndjson", "w", encoding="utf-8") as f:
        for rec in block_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(out_dir / "articles.ndjson", "w", encoding="utf-8") as f:
        for page in pages:
            for art in page["articles"]:
                f.write(json.dumps(art, ensure_ascii=False) + "\n")

    issue_meta["page_count"] = len(pages)
    issue_meta["block_count"] = len(block_records)
    issue_meta["article_count"] = sum(len(p["articles"]) for p in pages)
    issue_meta["pages"] = [f"pages/page_{i + 1:03d}" for i in range(len(pages))]
    issue_meta["coverage_pages"] = [
        {"page_number": p["page_number"], "ratio": p.get("coverage", {}).get("ratio", 1.0),
         "low": p.get("coverage", {}).get("low", False)}
        for p in pages
    ]
    issue_meta["coverage_min_ratio"] = round(
        min((p.get("coverage", {}).get("ratio", 1.0) for p in pages), default=1.0), 3
    )
    issue_meta["output_dir"] = str(out_dir.relative_to(ROOT))
    issue_meta["processed_at"] = now_iso()

    (out_dir / "issue.json").write_text(
        json.dumps(issue_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _summary_record(issue_meta: dict) -> dict:
    date = issue_meta.get("date", "unknown")
    return {
        k: issue_meta.get(k)
        for k in (
            "issue_id", "newspaper", "newspaper_source", "date", "source_pdf",
            "page_count", "block_count", "article_count", "processed_at",
            "output_dir",
        )
    } | {
        "city": issue_meta.get("city", ""),
        "province": issue_meta.get("province", ""),
        "country": issue_meta.get("country"),
        "place_source": issue_meta.get("place_source"),
        # Kept here so siblings can be matched by masthead fingerprint
        # without re-reading every issue's pages.
        "masthead": issue_meta.get("masthead", ""),
        "year": date[:4] if date != "unknown" else "",
        "month": date[5:7] if date != "unknown" else "",
        "day": date[8:10] if date != "unknown" else "",
        "day_of_week": issue_meta.get("day_of_week"),
        "date_precision": issue_meta.get("date_precision"),
    }


def _update_index(issue_meta: dict) -> None:
    issues = load_index(INDEX_PATH)
    issues = [i for i in issues if i.get("issue_id") != issue_meta["issue_id"]]
    issues.append(_summary_record(issue_meta))
    write_index(issues, INDEX_PATH)


# Commands
def process_pdf(
    pdf_path: Path,
    force: bool = False,
    newspaper: str | None = None,
    dpi: int = 300,
) -> None:
    issue_id = pdf_path.stem
    out_dir = OUTPUT_DIR / issue_id
    pages_dir = out_dir / "pages"
    issue_json_path = out_dir / "issue.json"

    if issue_json_path.exists() and not force:
        print(f"[skip] {issue_id} already processed (use --force to redo)")
        return

    print(f"[1/4] Rendering {pdf_path.name} at {dpi} DPI ...")
    images = render_pages(pdf_path, dpi=dpi)
    print(f"      {len(images)} pages")

    print("[2/4] Running Surya layout + OCR ...")
    layout_results = detect_layout(images)
    ocr_results = detect_ocr(images)

    print("[3/4] Assembling structured documents ...")
    embedded_texts = extract_embedded_text(pdf_path)
    embedded_line_pages = extract_embedded_lines(pdf_path, scale=dpi / 72.0)
    pages = []
    for i, (img, ocr_res, lay_res) in enumerate(zip(images, ocr_results, layout_results)):
        pno = i + 1
        page_dir = pages_dir / f"page_{pno:03d}"
        artifacts = save_page_images(pno, img, page_dir)

        surya_lines = ocr_lines_from_pydantic(ocr_res)
        emb_lines = embedded_line_pages[i] if i < len(embedded_line_pages) else []
        lines = merge_line_sources(emb_lines, surya_lines)

        page = build_page(
            issue_id, pno, img.width, img.height,
            lines,
            layout_boxes_from_pydantic(lay_res),
            artifacts,
        )
        _finalize_page(page, embedded_texts[i] if i < len(embedded_texts) else "")
        _write_page_artifacts(page, page_dir, {
            "layout": json.loads(lay_res.model_dump_json()),
            "ocr": json.loads(ocr_res.model_dump_json()),
        })
        pages.append(page)
        flag = " [LOW COVERAGE]" if page.get("coverage", {}).get("low") else ""
        print(f"      page {pno:02d}: {len(page['blocks'])} blocks, "
              f"{len(page['articles'])} articles, coverage={page.get('coverage', {}).get('ratio', 1):.2f}{flag}")

    surya_first = join_lines(
        [b["text"]["normalized"] for b in pages[0]["blocks"]]
    ) if pages else ""
    issue_meta = extract_issue_metadata(
        pdf_path, issue_id, newspaper=newspaper, surya_first_page_text=surya_first,
        pages=pages,
    )
    refined, src = refine_newspaper(pages, issue_meta["newspaper"])
    if src == "fallback":
        # Not enough internal repetition to confirm a name on its own pages
        # (common on short issues) — try a sibling in the same city/province.
        cross = _cross_issue_newspaper_name(
            pages, issue_meta.get("city"), issue_meta.get("province"), INDEX_PATH
        )
        if cross:
            refined, src = cross, "cross_issue"
    issue_meta["newspaper"] = refined
    issue_meta["newspaper_source"] = src
    filled = _apply_sibling_backfill(issue_meta)
    if filled:
        print(f"      filled from sibling issues: {', '.join(filled)}")

    print("[4/4] Generating manifest ...")
    _assemble_issue_outputs(issue_id, pages, issue_meta, out_dir)
    _update_index(issue_meta)

    print(f"\nDone. {len(pages)} pages, {issue_meta['block_count']} blocks, "
          f"{issue_meta['article_count']} articles -> {out_dir}")


def rebuild_issue(issue_id: str) -> None:
    """Reassemble page.json/articles from cached layout.json — no re-OCR."""
    out_dir = OUTPUT_DIR / issue_id
    issue_json_path = out_dir / "issue.json"
    if not issue_json_path.exists():
        print(f"error: {issue_id} not found in {OUTPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    prev_meta = json.loads(issue_json_path.read_text(encoding="utf-8"))
    source_pdf = INPUT_DIR / prev_meta.get("source_pdf", f"{issue_id}.pdf")
    embedded_texts = extract_embedded_text(source_pdf) if source_pdf.exists() else []
    page_sizes = pdf_page_sizes_pt(source_pdf) if source_pdf.exists() else []
    embedded_line_pages = extract_embedded_lines(source_pdf, scale=1.0) if source_pdf.exists() else []

    pages = []
    for i in range(prev_meta.get("page_count", 0)):
        pno = i + 1
        page_dir = out_dir / "pages" / f"page_{pno:03d}"
        if not (page_dir / "layout.json").exists():
            print(f"error: missing {page_dir / 'layout.json'}", file=sys.stderr)
            sys.exit(1)
        layout = json.loads((page_dir / "layout.json").read_text(encoding="utf-8"))
        img = Image.open(page_dir / "page.png")

        scale = (img.width / page_sizes[i][0]) if i < len(page_sizes) and page_sizes[i][0] else 1.0
        emb_lines = scale_lines(list(embedded_line_pages[i]), scale) if i < len(embedded_line_pages) else []
        surya_lines = ocr_lines_from_json(layout.get("ocr", {}).get("text_lines", []))
        lines = merge_line_sources(emb_lines, surya_lines)

        page = build_page(
            issue_id, pno, img.width, img.height,
            lines,
            layout_boxes_from_json(layout.get("layout", {}).get("bboxes", [])),
            {"image": "page.png"},
        )
        _finalize_page(page, embedded_texts[i] if i < len(embedded_texts) else "")
        _write_page_artifacts(page, page_dir, layout)
        pages.append(page)
        flag = " [LOW COVERAGE]" if page.get("coverage", {}).get("low") else ""
        print(f"      page {pno:02d}: {len(page['blocks'])} blocks, "
              f"{len(page['articles'])} articles, coverage={page.get('coverage', {}).get('ratio', 1):.2f}{flag}")

    surya_first = join_lines(
        [b["text"]["normalized"] for b in pages[0]["blocks"]]
    ) if pages else ""
    newspaper = prev_meta.get("newspaper") if prev_meta.get("newspaper_source") == "override" else None
    issue_meta = extract_issue_metadata(
        source_pdf, issue_id, newspaper=newspaper, surya_first_page_text=surya_first,
        pages=pages,
    )
    refined, src = refine_newspaper(pages, issue_meta["newspaper"])
    if src == "fallback":
        # Not enough internal repetition to confirm a name on its own pages
        # (common on short issues) — try a sibling in the same city/province.
        cross = _cross_issue_newspaper_name(
            pages, issue_meta.get("city"), issue_meta.get("province"), INDEX_PATH
        )
        if cross:
            refined, src = cross, "cross_issue"
    issue_meta["newspaper"] = refined
    issue_meta["newspaper_source"] = src
    filled = _apply_sibling_backfill(issue_meta)
    if filled:
        print(f"      filled from sibling issues: {', '.join(filled)}")

    _assemble_issue_outputs(issue_id, pages, issue_meta, out_dir)
    _update_index(issue_meta)
    print(f"\nRebuilt {issue_id}: {issue_meta['block_count']} blocks, "
          f"{issue_meta['article_count']} articles -> {out_dir}")


def _apply_sibling_backfill(issue_meta: dict) -> list:
    """Fill this issue's metadata gaps from confirmed same-publication
    siblings already in the index. Returns the field names that were filled.
    """
    filled = backfill_from_siblings(issue_meta, INDEX_PATH)
    if not filled:
        return []
    for field, value in filled.items():
        issue_meta[field] = value
        if field == "newspaper":
            issue_meta["newspaper_source"] = "cross_issue"
        elif field in ("city", "province"):
            issue_meta["place_source"] = "cross_issue"
    issue_meta["country"] = _country_from_region(issue_meta.get("province"))
    return sorted(filled)


def _reconcile_cross_issue_names() -> int:
    """Re-check every fallback-named issue against the full index, in case a
    sibling processed later in this same batch now confirms its name.
    Returns how many issues were newly resolved.
    """
    reconciled = 0
    for entry in load_index(INDEX_PATH):
        if entry.get("newspaper_source") != "fallback" and entry.get("city"):
            continue
        issue_id = entry["issue_id"]
        out_dir = OUTPUT_DIR / issue_id
        issue_json_path = out_dir / "issue.json"
        page_files = sorted((out_dir / "pages").glob("page_*/page.json"))
        if not issue_json_path.exists() or not page_files:
            continue
        meta = json.loads(issue_json_path.read_text(encoding="utf-8"))
        pages = [json.loads(p.read_text(encoding="utf-8")) for p in page_files]
        changes = []
        # Only a still-unresolved name is looked up; one an issue already
        # confirmed from its own pages keeps that provenance.
        if meta.get("newspaper_source") == "fallback":
            cross = _cross_issue_newspaper_name(
                pages, meta.get("city"), meta.get("province"), INDEX_PATH
            )
            if cross:
                meta["newspaper"] = cross
                meta["newspaper_source"] = "cross_issue"
                changes.append(f"newspaper -> {cross!r}")
        changes += _apply_sibling_backfill(meta)
        if not changes:
            continue
        issue_json_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _update_index(meta)
        print(f"  [reconciled] {issue_id}: {', '.join(changes)}")
        reconciled += 1
    return reconciled


def batch(force: bool = False, newspaper: str | None = None, dpi: int = 300) -> None:
    pdfs = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {INPUT_DIR}")
        return
    print(f"Processing {len(pdfs)} PDF(s) from {INPUT_DIR}")
    for pdf in pdfs:
        process_pdf(pdf, force=force, newspaper=newspaper, dpi=dpi)

    reconciled = _reconcile_cross_issue_names()
    if reconciled:
        print(f"\nReconciled {reconciled} issue(s) against siblings processed later in this batch.")


def _ellipsis(text: str, width: int) -> str:
    """Trim an over-long field so the columns stay readable."""
    text = str(text or "")
    return text if len(text) <= width else text[: width - 1] + "\u2026"


def group() -> None:
    issues = load_index(INDEX_PATH)
    if not issues:
        print("No issues indexed yet.")
        return
    # Group by canonical name so one publication with two nameplate readings
    # doesn't appear twice.
    canonical = canonical_newspaper_names(issues)
    by_pub: dict = {}
    for i in issues:
        pub = canonical.get(i["issue_id"]) or i.get("newspaper", "Unknown")
        by_pub.setdefault(pub, []).append(i)
    for pub in sorted(by_pub, key=str.lower):
        items = sorted(by_pub[pub], key=lambda x: x.get("date", ""))
        print(f"\n{pub}  ({len(items)} issue(s))")
        for i in items:
            loc = f"{i.get('city', '')}{', ' + i.get('province', '') if i.get('province') else ''}"
            variant = ("" if i.get("newspaper") == pub
                       else f"  [read as: {i.get('newspaper')}]")
            print(f"  {i.get('date', '?'):<12} {_ellipsis(i.get('issue_id'), 44):<44} "
                  f"pages={i.get('page_count', 0):<4} articles={i.get('article_count', 0):<5} "
                  f"{loc}{variant}")


def stats() -> None:
    issues = load_index(INDEX_PATH)
    if not issues:
        print("No issues indexed yet.")
        return
    print(f"{'issue_id':<46}{'newspaper':<28}{'date':<12}{'pg':<4}{'blocks':<8}"
          f"{'arts':<6}{'cov':<6}place")
    print("-" * 128)
    total_blocks = total_articles = 0
    for i in issues:
        place = " ".join(x for x in (i.get("city"), i.get("province"),
                                     i.get("country")) if x)
        print(f"{_ellipsis(i['issue_id'], 44):<46}{_ellipsis(i['newspaper'], 26):<28}"
              f"{i['date']:<12}{i['page_count']:<4}{i['block_count']:<8}"
              f"{i.get('article_count', 0):<6}{i.get('coverage_min_ratio', 1):<6}"
              f"{place}".rstrip())
        total_blocks += i.get("block_count", 0)
        total_articles += i.get("article_count", 0)
    print("-" * 128)
    print(f"{len(issues)} issue(s), {total_blocks} blocks, {total_articles} articles")


def export(min_word_count: int = 15, use_corrected: bool = False) -> None:
    out = export_corpus(INDEX_PATH, OUTPUT_DIR, CORPUS_DIR, min_word_count,
                        use_corrected=use_corrected)
    print(f"Wrote {out}")


def clean(issue_id: str, model: str, ollama_url: str, force: bool = False,
          min_words: int = 15) -> None:
    from .clean import clean_issue

    out_dir = OUTPUT_DIR / issue_id
    issue_json = out_dir / "issue.json"
    if not issue_json.exists():
        print(f"error: {issue_id} not found in {OUTPUT_DIR}", file=sys.stderr)
        sys.exit(1)
    meta = json.loads(issue_json.read_text(encoding="utf-8"))
    source_pdf = INPUT_DIR / meta.get("source_pdf", f"{issue_id}.pdf")
    if not source_pdf.exists():
        source_pdf = None
        print("warn: source PDF not found; corrections will not compare embedded text")
    clean_issue(out_dir, source_pdf=source_pdf, model=model, ollama_url=ollama_url,
                force=force, min_words=min_words)


# CLI
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="compositor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="Process a single PDF into structured documents")
    p.add_argument("pdf", type=Path, help="path to the PDF (or name under input/)")
    p.add_argument("--force", action="store_true", help="reprocess even if cached")
    p.add_argument("--newspaper", default=None, help="override newspaper name")
    p.add_argument("--dpi", type=int, default=300)

    p = sub.add_parser("batch", help="Process all PDFs in input/")
    p.add_argument("--force", action="store_true")
    p.add_argument("--newspaper", default=None)
    p.add_argument("--dpi", type=int, default=300)

    p = sub.add_parser("rebuild", help="Reassemble docs from cached layout.json (no re-OCR)")
    p.add_argument("issue_id")

    sub.add_parser("group", help="List the corpus grouped by publication, then date")
    sub.add_parser("stats", help="Summarize the index.json manifest")

    p = sub.add_parser("export", help="Write a RAG-ready corpus.jsonl")
    p.add_argument("--min-word-count", type=int, default=15,
                    help="drop articles shorter than this (classified-ad "
                         "fragments, stray captions); 0 keeps everything")
    p.add_argument("--use-corrected", action="store_true",
                   help="use LLM-corrected article text (run `clean` first)")

    p = sub.add_parser("clean", help="LLM-correct OCR noise per article (local Ollama)")
    p.add_argument("issue_id")
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--force", action="store_true")
    p.add_argument("--min-words", type=int, default=15,
                   help="skip articles shorter than this many words")

    args = parser.parse_args(argv)

    if args.command == "process":
        pdf = Path(args.pdf)
        if not pdf.exists():
            pdf = INPUT_DIR / pdf.name
        if not pdf.exists():
            print(f"error: {pdf} not found", file=sys.stderr)
            sys.exit(1)
        process_pdf(pdf, force=args.force, newspaper=args.newspaper, dpi=args.dpi)
    elif args.command == "batch":
        batch(force=args.force, newspaper=args.newspaper, dpi=args.dpi)
    elif args.command == "rebuild":
        rebuild_issue(args.issue_id)
    elif args.command == "group":
        group()
    elif args.command == "stats":
        stats()
    elif args.command == "export":
        export(min_word_count=args.min_word_count, use_corrected=args.use_corrected)
    elif args.command == "clean":
        clean(args.issue_id, model=args.model, ollama_url=args.ollama_url,
              force=args.force, min_words=args.min_words)


if __name__ == "__main__":
    main()