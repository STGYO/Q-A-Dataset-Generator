"""Parallel Q&A pair generation from text chunks."""

from __future__ import annotations

import logging
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from qa_generator.benchmarks import BenchmarkTracker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default prompt template
# ---------------------------------------------------------------------------

DEFAULT_PROMPT_TEMPLATE = """\
You are an expert at creating question-and-answer pairs for training datasets.

Given the following text excerpt, generate {num_pairs} high-quality question-answer pairs.
Each pair must be grounded in the provided text.

Text:
\"\"\"
{chunk}
\"\"\"

Respond ONLY with a numbered list in this exact format:
Q1: <question>
A1: <answer>
Q2: <question>
A2: <answer>
...

Do not include any other text, commentary, or explanations.
"""

# ---------------------------------------------------------------------------
# Parsing helper
# ---------------------------------------------------------------------------


def parse_qa_response(response_text: str) -> List[Dict[str, str]]:
    """Parse LLM-generated Q&A text into a list of ``{question, answer}`` dicts."""
    pairs: List[Dict[str, str]] = []

    # Match patterns like "Q1: ... A1: ..."
    q_pattern = re.compile(r"Q\d+[.:]\s*(.+?)(?=A\d+[.:])", re.DOTALL | re.IGNORECASE)
    a_pattern = re.compile(r"A\d+[.:]\s*(.+?)(?=Q\d+[.:]|$)", re.DOTALL | re.IGNORECASE)

    questions = [m.group(1).strip() for m in q_pattern.finditer(response_text)]
    answers = [m.group(1).strip() for m in a_pattern.finditer(response_text)]

    for q, a in zip(questions, answers):
        if q and a:
            pairs.append({"question": q, "answer": a})

    if not pairs:
        # Fallback: try line-by-line "Q: / A:" pattern
        lines = response_text.splitlines()
        current_q: Optional[str] = None
        for line in lines:
            line = line.strip()
            if re.match(r"^Q\d*[.:]", line, re.IGNORECASE):
                current_q = re.sub(r"^Q\d*[.:]\s*", "", line, flags=re.IGNORECASE).strip()
            elif re.match(r"^A\d*[.:]", line, re.IGNORECASE) and current_q:
                ans = re.sub(r"^A\d*[.:]\s*", "", line, flags=re.IGNORECASE).strip()
                if ans:
                    pairs.append({"question": current_q, "answer": ans})
                current_q = None

    return pairs


# ---------------------------------------------------------------------------
# Per-chunk generation
# ---------------------------------------------------------------------------


def generate_qa_for_chunk(
    chunk: str,
    llm_client,
    chunk_index: int = 0,
    num_pairs: int = 5,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    tracker: Optional[BenchmarkTracker] = None,
) -> Dict:
    """Generate Q&A pairs for a single text chunk.

    Returns a result dict with keys: ``chunk_index``, ``pairs``, ``error``,
    ``latency_seconds``, ``tokens_per_second``.
    """
    result = {
        "chunk_index": chunk_index,
        "pairs": [],
        "error": None,
        "latency_seconds": 0.0,
        "tokens_per_second": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    try:
        prompt = prompt_template.format(chunk=chunk, num_pairs=num_pairs)
        response = llm_client.complete(prompt)

        result["latency_seconds"] = response.latency_seconds
        result["tokens_per_second"] = response.tokens_per_second
        result["prompt_tokens"] = response.prompt_tokens
        result["completion_tokens"] = response.completion_tokens

        pairs = parse_qa_response(response.content)
        result["pairs"] = pairs

        logger.info(
            "Chunk %d: generated %d pairs (%.2fs, %.1f TPS)",
            chunk_index,
            len(pairs),
            response.latency_seconds,
            response.tokens_per_second,
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
        logger.error("Chunk %d: error — %s", chunk_index, exc)

    if tracker is not None:
        tracker.record(result)

    return result


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------


def generate_qa_parallel(
    chunks: List[str],
    llm_client,
    num_pairs: int = 5,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_workers: int = 4,
    use_threads: bool = True,
    tracker: Optional[BenchmarkTracker] = None,
) -> List[Dict]:
    """Generate Q&A pairs for all chunks in parallel.

    Args:
        chunks: List of text chunks.
        llm_client: An LLM client instance (must be picklable when
            *use_threads* is ``False``).
        num_pairs: Target number of Q&A pairs per chunk.
        prompt_template: Prompt template string.
        max_workers: Number of parallel workers.
        use_threads: Use ``ThreadPoolExecutor`` (True, recommended for I/O-bound
            API calls) or ``ProcessPoolExecutor`` (False, for CPU-bound local
            inference).
        tracker: Optional :class:`~qa_generator.benchmarks.BenchmarkTracker`.

    Returns:
        List of result dicts ordered by chunk index.
    """
    results: List[Optional[Dict]] = [None] * len(chunks)

    Executor = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with Executor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                generate_qa_for_chunk,
                chunk,
                llm_client,
                idx,
                num_pairs,
                prompt_template,
                tracker,
            ): idx
            for idx, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Worker for chunk %d raised: %s", idx, exc)
                results[idx] = {
                    "chunk_index": idx,
                    "pairs": [],
                    "error": str(exc),
                    "latency_seconds": 0.0,
                    "tokens_per_second": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }

    # Replace any remaining None (shouldn't happen)
    return [r for r in results if r is not None]
