from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import fitz
from PIL import Image


def render_pages(pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
    """Render PDF pages to PIL images at the given DPI."""
    doc = fitz.open(str(pdf_path))
    images: List[Image.Image] = []
    try:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            images.append(img)
    finally:
        doc.close()
    return images


def save_page_images(page_num: int, img: Image.Image, page_dir: Path) -> Dict[str, str]:
    page_dir.mkdir(parents=True, exist_ok=True)

    png_path = page_dir / "page.png"
    img.save(png_path, "PNG")

    return {"image": f"{page_dir.name}/page.png"}


def pdf_page_count(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def extract_embedded_text(pdf_path: Path) -> List[str]:
    doc = fitz.open(str(pdf_path))
    try:
        return [page.get_text("text") for page in doc]
    finally:
        doc.close()


def pdf_page_sizes_pt(pdf_path: Path) -> List[tuple]:
    doc = fitz.open(str(pdf_path))
    try:
        return [(page.rect.width, page.rect.height) for page in doc]
    finally:
        doc.close()