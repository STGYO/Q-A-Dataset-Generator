"""Tests for export module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qa_generator.exporter import apply_schema, export, export_json, export_jsonl


SAMPLE_PAIRS = [
    {"question": "What is Python?", "answer": "A programming language.", "_score": 0.9},
    {"question": "What is AI?", "answer": "Artificial Intelligence.", "_score": 0.8},
]


def test_apply_schema_default_keys():
    result = apply_schema(SAMPLE_PAIRS)
    assert result[0]["question"] == "What is Python?"
    assert result[0]["answer"] == "A programming language."


def test_apply_schema_custom_keys():
    result = apply_schema(SAMPLE_PAIRS, question_key="Q", answer_key="A")
    assert "Q" in result[0]
    assert "A" in result[0]
    assert "question" not in result[0]
    assert "answer" not in result[0]


def test_apply_schema_exclude_score():
    result = apply_schema(SAMPLE_PAIRS, include_score=False)
    assert "_score" not in result[0]


def test_apply_schema_include_score():
    result = apply_schema(SAMPLE_PAIRS, include_score=True)
    assert "_score" in result[0]


def test_export_json(tmp_path):
    out = tmp_path / "out.json"
    written = export_json(SAMPLE_PAIRS, out)
    assert written == out
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["question"] == "What is Python?"


def test_export_jsonl(tmp_path):
    out = tmp_path / "out.jsonl"
    written = export_jsonl(SAMPLE_PAIRS, out)
    assert written == out
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert "question" in record


def test_export_dispatch_json(tmp_path):
    out = tmp_path / "out.json"
    export(SAMPLE_PAIRS, out, fmt="json")
    assert out.exists()


def test_export_dispatch_jsonl(tmp_path):
    out = tmp_path / "out.jsonl"
    export(SAMPLE_PAIRS, out, fmt="jsonl")
    assert out.exists()


def test_export_unsupported_format(tmp_path):
    with pytest.raises(ValueError, match="Unsupported export format"):
        export(SAMPLE_PAIRS, tmp_path / "out.csv", fmt="csv")


def test_export_creates_parent_dirs(tmp_path):
    out = tmp_path / "nested" / "dir" / "out.jsonl"
    export(SAMPLE_PAIRS, out, fmt="jsonl")
    assert out.exists()
