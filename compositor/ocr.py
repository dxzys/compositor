"""Surya text detection and OCR for batches of page images."""
from __future__ import annotations

from typing import Dict, List, Optional

from PIL import Image


_predictors: Optional[Dict] = None


def get_predictors() -> Dict:
    """Lazily load and cache the Surya predictors actually used.

    Built directly instead of via Surya's 'load_predictors()', which also
    loads the unused 'ocr_error' and 'table_rec' models.
    """
    global _predictors
    if _predictors is None:
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.layout import LayoutPredictor
        from surya.recognition import RecognitionPredictor
        from surya.settings import settings

        _predictors = {
            "layout": LayoutPredictor(FoundationPredictor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)),
            "recognition": RecognitionPredictor(FoundationPredictor(checkpoint=settings.RECOGNITION_MODEL_CHECKPOINT)),
            "detection": DetectionPredictor(),
        }
    return _predictors


def detect_ocr(images: List[Image.Image]) -> List:
    """Run text detection + OCR over a batch of page images."""
    preds = get_predictors()
    det_predictor = preds["detection"]
    rec_predictor = preds["recognition"]

    results = rec_predictor(
        images,
        det_predictor=det_predictor,
        return_words=True,
    )
    return results