"""Document ingestion and text extraction for PDF, TXT, and CSV inputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path) -> str:
    """Extract text from a PDF file using PyMuPDF (fitz).

    Falls back to pypdf if PyMuPDF is unavailable.
    """
    path = Path(path)
    text_parts: List[str] = []

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        logger.info("Loaded PDF via PyMuPDF: %s (%d pages)", path.name, len(text_parts))
    except ImportError:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
            logger.info("Loaded PDF via pypdf: %s (%d pages)", path.name, len(text_parts))
        except ImportError as exc:
            raise ImportError(
                "No PDF library found. Install PyMuPDF (`pip install pymupdf`) "
                "or pypdf (`pip install pypdf`)."
            ) from exc

    return "\n".join(text_parts)


def load_txt(path: str | Path) -> str:
    """Load plain text from a TXT file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    logger.info("Loaded TXT: %s (%d chars)", path.name, len(text))
    return text


def load_csv(path: str | Path, text_columns: List[str] | None = None) -> str:
    """Load and concatenate text from CSV columns.

    Args:
        path: Path to the CSV file.
        text_columns: Column names to extract.  If *None*, all string-type
            columns are concatenated.
    """
    import pandas as pd

    path = Path(path)
    df = pd.read_csv(str(path))

    if text_columns:
        cols = [c for c in text_columns if c in df.columns]
        if not cols:
            raise ValueError(
                f"None of the specified columns {text_columns} found in {path.name}. "
                f"Available columns: {list(df.columns)}"
            )
    else:
        cols = [c for c in df.columns if df[c].dtype == object]
        if not cols:
            cols = list(df.columns)

    combined = df[cols].fillna("").astype(str).agg(" ".join, axis=1).str.cat(sep="\n")
    logger.info("Loaded CSV: %s (%d rows, cols=%s)", path.name, len(df), cols)
    return combined


def ingest(path: str | Path, text_columns: List[str] | None = None) -> str:
    """Auto-detect file type and return extracted text.

    Args:
        path: Path to a PDF, TXT, or CSV file.
        text_columns: Only used when *path* is a CSV file.

    Returns:
        Extracted text as a single string.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".txt":
        return load_txt(path)
    if suffix == ".csv":
        return load_csv(path, text_columns=text_columns)

    raise ValueError(
        f"Unsupported file format '{suffix}'. Supported formats: .pdf, .txt, .csv"
    )
