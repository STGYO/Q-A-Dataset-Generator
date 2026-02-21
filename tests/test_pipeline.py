"""Tests for pipeline orchestration with a mock LLM."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from qa_generator.llm import LLMResponse
from qa_generator.pipeline import run_pipeline


def _make_mock_llm(response_text: str = None) -> MagicMock:
    """Create a mock LLM client that returns a fixed response."""
    if response_text is None:
        response_text = (
            "Q1: What is Python?\nA1: A high-level programming language.\n"
            "Q2: Who created Python?\nA2: Guido van Rossum.\n"
        )
    mock = MagicMock()
    mock.complete.return_value = LLMResponse(
        content=response_text,
        model="mock-model",
        prompt_tokens=50,
        completion_tokens=30,
        latency_seconds=0.1,
    )
    return mock


def test_pipeline_txt_to_jsonl(tmp_path):
    """Full pipeline with a TXT file and mock LLM → JSONL output."""
    input_file = tmp_path / "doc.txt"
    input_file.write_text(
        "Python is a versatile programming language created by Guido van Rossum. "
        "It emphasizes readability and simplicity. " * 10,
        encoding="utf-8",
    )
    output_file = tmp_path / "out.jsonl"
    mock_llm = _make_mock_llm()

    result = run_pipeline(
        input_path=input_file,
        llm_client=mock_llm,
        output_path=output_file,
        fmt="jsonl",
        chunk_size=300,
        chunk_overlap=50,
        num_pairs_per_chunk=2,
        max_workers=1,
        min_quality_score=0.0,
    )

    assert output_file.exists()
    lines = output_file.read_text().strip().splitlines()
    assert len(lines) > 0

    record = json.loads(lines[0])
    assert "question" in record
    assert "answer" in record

    assert result["metrics"]["total_chunks"] > 0
    assert result["metrics"]["successful_chunks"] > 0


def test_pipeline_csv_to_json(tmp_path):
    """Full pipeline with a CSV file → JSON output."""
    input_file = tmp_path / "data.csv"
    input_file.write_text(
        "text\n"
        "Machine learning is a subset of artificial intelligence.\n"
        "Deep learning uses neural networks with many layers.\n",
        encoding="utf-8",
    )
    output_file = tmp_path / "out.json"
    mock_llm = _make_mock_llm()

    result = run_pipeline(
        input_path=input_file,
        llm_client=mock_llm,
        output_path=output_file,
        fmt="json",
        chunk_size=500,
        chunk_overlap=50,
        num_pairs_per_chunk=2,
        max_workers=1,
        min_quality_score=0.0,
    )

    assert output_file.exists()
    data = json.loads(output_file.read_text())
    assert isinstance(data, list)


def test_pipeline_custom_schema(tmp_path):
    """Custom question/answer keys are applied in the output."""
    input_file = tmp_path / "doc.txt"
    input_file.write_text("The sky is blue due to Rayleigh scattering. " * 5, encoding="utf-8")
    output_file = tmp_path / "out.jsonl"
    mock_llm = _make_mock_llm()

    run_pipeline(
        input_path=input_file,
        llm_client=mock_llm,
        output_path=output_file,
        fmt="jsonl",
        question_key="Q",
        answer_key="A",
        min_quality_score=0.0,
    )

    lines = output_file.read_text().strip().splitlines()
    if lines:
        record = json.loads(lines[0])
        assert "Q" in record
        assert "A" in record
        assert "question" not in record


def test_pipeline_metrics_csv(tmp_path):
    """save_metrics_csv=True should produce a .metrics.csv file."""
    input_file = tmp_path / "doc.txt"
    input_file.write_text("Deep learning uses large neural networks. " * 5, encoding="utf-8")
    output_file = tmp_path / "out.jsonl"
    mock_llm = _make_mock_llm()

    run_pipeline(
        input_path=input_file,
        llm_client=mock_llm,
        output_path=output_file,
        fmt="jsonl",
        min_quality_score=0.0,
        save_metrics_csv=True,
    )

    metrics_file = output_file.with_suffix(".metrics.csv")
    assert metrics_file.exists()


def test_pipeline_llm_error_handled(tmp_path):
    """LLM errors per chunk should be captured without crashing the pipeline."""
    input_file = tmp_path / "doc.txt"
    input_file.write_text("Some text. " * 20, encoding="utf-8")
    output_file = tmp_path / "out.jsonl"

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = RuntimeError("LLM unavailable")

    result = run_pipeline(
        input_path=input_file,
        llm_client=mock_llm,
        output_path=output_file,
        fmt="jsonl",
        chunk_size=200,
        chunk_overlap=0,
        min_quality_score=0.0,
    )

    # Pipeline should complete but with failures
    assert result["metrics"]["failed_chunks"] > 0
    assert result["metrics"]["total_pairs_generated"] == 0
