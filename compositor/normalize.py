"""Light OCR text normalization; aggressive cleaning is left to downstream indexing."""
from __future__ import annotations

import re

# OCR artifacts and whitespace characters that should not affect search.
_ARTIFACTS = re.compile(r"[\u00ad\u200b\ufeff\u00a0\u25a0]")

# Some PDFs ship a font whose ToUnicode CMap is broken for specific glyphs, so
# a digit renders fine on the page but decodes to a raw control byte in the
# text layer (e.g. "Page \x18\x08"). \t/\n/\r are excluded because
# whitespace is handled separately below.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# Surya may wrap emphasized spans in HTML-style formatting tags.
_FORMAT_TAGS = re.compile(r"</?(?:b|i|u|em|strong|s)>", re.IGNORECASE)

# <br> is a break. Replace it with a space instead of dropping it
# so words on either side don't get glued together.
_BREAK_TAGS = re.compile(r"<br\s*/?>", re.IGNORECASE)

# A lone "|" is often a misread column-divider rule. Replace it with
# a space since it may be glued to a word ("Co-Operate|To").
_STRAY_PIPE = re.compile(r"\|")

# Surya sometimes misreads numeric content (weather tables, fractions,
# time ranges, etc.) as LaTeX math and wraps it in <math>...</math>.
# Only observed commands are unwrapped; not general LaTeX parsing.
_MATH_TAGS = re.compile(r"</?math>", re.IGNORECASE)
_LATEX_TEXT_CMD = re.compile(r"\\text\{([^{}]*)\}")
_LATEX_FRAC_CMD = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")
_LATEX_BULLET_CMD = re.compile(r"\\bullet")
_LATEX_SPACING_CMD = re.compile(r"\\quad|\\qquad|\\,|\\;")
_LATEX_LINEBREAK_CMD = re.compile(r"\\\\")


def normalize_line(text: str) -> str:
    """Strip artifacts and collapse whitespace in a single OCR line."""
    text = _ARTIFACTS.sub("", text)
    text = _CONTROL_CHARS.sub("", text)
    text = _BREAK_TAGS.sub(" ", text)
    text = _FORMAT_TAGS.sub("", text)
    text = _LATEX_TEXT_CMD.sub(r"\1", text)
    text = _LATEX_FRAC_CMD.sub(r" \1/\2", text)
    text = _LATEX_BULLET_CMD.sub("\u2022", text)
    text = _LATEX_SPACING_CMD.sub(" ", text)
    text = _LATEX_LINEBREAK_CMD.sub(" ", text)
    text = _MATH_TAGS.sub("", text)
    text = _STRAY_PIPE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def join_lines(lines: list[str]) -> str:
    """Join OCR lines into a paragraph.

    Handles hyphenated line breaks (``word-`` + next line starting lowercase
    joins without a hyphen; otherwise lines join with a space).
    """
    parts: list[str] = []
    for raw in lines:
        line = normalize_line(raw)
        if not line:
            continue
        if parts:
            prev = parts[-1]
            if prev.endswith("-") and line[0].islower():
                parts[-1] = prev[:-1] + line
            elif line[0].islower() and prev and prev[-1].isalnum():
                parts[-1] = prev + " " + line
            else:
                parts.append(line)
        else:
            parts.append(line)
    return "\n".join(parts)