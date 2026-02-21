"""Tests for quality scoring and filtering."""

from __future__ import annotations

import pytest
from qa_generator.quality import (
    deduplicate,
    filter_pairs,
    score_pair,
)


def test_score_pair_perfect():
    pair = {"question": "What is the capital of France?", "answer": "The capital of France is Paris."}
    score = score_pair(pair)
    assert score == 1.0


def test_score_pair_empty_answer():
    pair = {"question": "What is the capital?", "answer": ""}
    score = score_pair(pair)
    assert score == 0.0


def test_score_pair_empty_question():
    pair = {"question": "", "answer": "Paris"}
    score = score_pair(pair)
    assert score == 0.0


def test_score_pair_short_answer():
    pair = {"question": "What is the capital of France?", "answer": "P"}
    score = score_pair(pair)
    assert score < 1.0


def test_score_pair_no_question_mark():
    pair = {"question": "Tell me about Paris", "answer": "Paris is a city."}
    score = score_pair(pair)
    # Contains no '?' but may have interrogative word pattern fallback
    assert 0.0 <= score <= 1.0


def test_deduplicate_removes_exact_duplicates():
    pairs = [
        {"question": "What?", "answer": "A"},
        {"question": "What?", "answer": "A"},
        {"question": "How?", "answer": "B"},
    ]
    result = deduplicate(pairs)
    assert len(result) == 2


def test_deduplicate_case_insensitive():
    pairs = [
        {"question": "WHAT IS THIS?", "answer": "Something"},
        {"question": "what is this?", "answer": "something"},
    ]
    result = deduplicate(pairs)
    assert len(result) == 1


def test_filter_pairs_removes_low_quality():
    pairs = [
        {"question": "What is the speed of light?", "answer": "The speed of light is approximately 3×10⁸ m/s."},
        {"question": "?", "answer": ""},  # bad
    ]
    result = filter_pairs(pairs, min_score=0.5)
    assert len(result) == 1
    assert result[0]["question"] == "What is the speed of light?"


def test_filter_pairs_adds_score():
    pairs = [
        {"question": "What is gravity?", "answer": "Gravity is a fundamental force."}
    ]
    result = filter_pairs(pairs, min_score=0.0)
    assert "_score" in result[0]
    assert 0.0 <= result[0]["_score"] <= 1.0


def test_filter_pairs_deduplication():
    pairs = [
        {"question": "What is AI?", "answer": "Artificial Intelligence."},
        {"question": "What is AI?", "answer": "Artificial Intelligence."},
    ]
    result = filter_pairs(pairs, min_score=0.0, deduplicate_pairs=True)
    assert len(result) == 1
