"""Surya layout analysis: region labels and reading order for a page."""
from __future__ import annotations

from typing import List

from PIL import Image


def detect_layout(images: List[Image.Image]) -> List:
    """Run Surya layout analysis over a batch of page images."""
    from .ocr import get_predictors

    preds = get_predictors()
    layout_predictor = preds["layout"]
    return layout_predictor(images)
