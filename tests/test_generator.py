"""Tests for generator module (Q&A parsing and generation)."""

from __future__ import annotations

import pytest

from qa_generator.generator import parse_qa_response


def test_parse_qa_numbered():
    text = (
        "Q1: What is Python?\n"
        "A1: Python is a programming language.\n"
        "Q2: What is AI?\n"
        "A2: Artificial Intelligence.\n"
    )
    pairs = parse_qa_response(text)
    assert len(pairs) == 2
    assert pairs[0]["question"] == "What is Python?"
    assert pairs[0]["answer"] == "Python is a programming language."


def test_parse_qa_fallback():
    text = (
        "Q: What is the sky?\n"
        "A: It is blue.\n"
        "Q: Why?\n"
        "A: Because of scattering.\n"
    )
    pairs = parse_qa_response(text)
    assert len(pairs) == 2


def test_parse_qa_empty_response():
    pairs = parse_qa_response("")
    assert pairs == []


def test_parse_qa_no_valid_pairs():
    text = "This is just a random paragraph with no Q&A structure."
    pairs = parse_qa_response(text)
    assert pairs == []


def test_parse_qa_partial():
    """Unpaired Q without A should be ignored."""
    text = "Q1: What is Python?\nA1: A language.\nQ2: Another question without answer"
    pairs = parse_qa_response(text)
    # At least the first pair should be returned
    assert any(p["question"] == "What is Python?" for p in pairs)
