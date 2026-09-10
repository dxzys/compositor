# compositor

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Code License: Apache 2.0](https://img.shields.io/badge/code%20license-Apache%202.0-green)](LICENSE)

Turn scanned newspaper PDFs into structured text.

## Quick start

Clone the repository and install:
```bash
pip install -e .
```

Make an `input` directory:
```bash
mkdir input
```

Copy your newspaper `.pdf` file(s) into it, either through using a file manager or through your CLI:
```bash
cp ~/path/to/newspapers/*.pdf input/
```

Batch process all PDFs in `input/`:
```bash
compositor batch

# alternatively, if the above doesn't work:
python3 -m compositor batch
```

Results land in `output/<pdf-stem>/`, where `<pdf-stem>` is the PDF filename.

## Commands

| command | what it does |
|---------|--------------|
| `process <pdf>` | Process a single PDF |
| `batch` | Process every PDF in `input/` (`--force` to redo) |
| `rebuild <issue_id>` | Reassemble documents from cached `layout.json`, no re-OCR |
| `group` | List the corpus grouped by publication & date |
| `stats` | Summarize the `index.json` manifest |
| `export` | Write `corpus/corpus.jsonl` (`--use-corrected`, `--min-word-count N`) |
| `clean <issue_id>` | Optional LLM OCR correction via a local Ollama server |

## Acknowledgements

Built on [Surya](https://github.com/datalab-to/surya) which does the layout analysis and text recognition this pipeline builds its document structure from. Also uses [PyMuPDF](https://github.com/pymupdf/pymupdf) for rendering and embedded-text extraction.
