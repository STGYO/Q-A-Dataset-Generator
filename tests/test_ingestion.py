"""Tests for document ingestion module."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# TXT ingestion
# ---------------------------------------------------------------------------


def test_load_txt_basic(tmp_path):
    from qa_generator.ingestion import load_txt

    f = tmp_path / "sample.txt"
    f.write_text("Hello, world!\nSecond line.", encoding="utf-8")
    result = load_txt(f)
    assert "Hello, world!" in result
    assert "Second line." in result


def test_load_txt_empty(tmp_path):
    from qa_generator.ingestion import load_txt

    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    assert load_txt(f) == ""


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------


def test_load_csv_all_columns(tmp_path):
    from qa_generator.ingestion import load_csv

    f = tmp_path / "data.csv"
    f.write_text("title,body\nFoo,Bar\nBaz,Qux\n", encoding="utf-8")
    result = load_csv(f)
    assert "Foo" in result
    assert "Bar" in result


def test_load_csv_specific_columns(tmp_path):
    from qa_generator.ingestion import load_csv

    f = tmp_path / "data.csv"
    f.write_text("title,body,score\nFoo,Bar,1\n", encoding="utf-8")
    result = load_csv(f, text_columns=["title"])
    assert "Foo" in result
    # body should not appear (only 'title' selected)
    assert "Bar" not in result


def test_load_csv_missing_columns(tmp_path):
    from qa_generator.ingestion import load_csv

    f = tmp_path / "data.csv"
    f.write_text("title,body\nFoo,Bar\n", encoding="utf-8")
    with pytest.raises(ValueError, match="None of the specified columns"):
        load_csv(f, text_columns=["nonexistent"])


# ---------------------------------------------------------------------------
# Auto-detect ingest
# ---------------------------------------------------------------------------


def test_ingest_txt(tmp_path):
    from qa_generator.ingestion import ingest

    f = tmp_path / "doc.txt"
    f.write_text("Auto detect test.", encoding="utf-8")
    result = ingest(f)
    assert "Auto detect test." in result


def test_ingest_csv(tmp_path):
    from qa_generator.ingestion import ingest

    f = tmp_path / "data.csv"
    f.write_text("col1\nvalue1\nvalue2\n", encoding="utf-8")
    result = ingest(f)
    assert "value1" in result


def test_ingest_unsupported_format(tmp_path):
    from qa_generator.ingestion import ingest

    f = tmp_path / "doc.docx"
    f.write_bytes(b"fake content")
    with pytest.raises(ValueError, match="Unsupported file format"):
        ingest(f)
