"""End-to-end Q&A dataset generation pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from qa_generator.benchmarks import BenchmarkTracker
from qa_generator.chunking import chunk_text
from qa_generator.exporter import export
from qa_generator.generator import DEFAULT_PROMPT_TEMPLATE, generate_qa_parallel
from qa_generator.ingestion import ingest
from qa_generator.quality import filter_pairs

logger = logging.getLogger(__name__)


def run_pipeline(
    input_path: str | Path,
    llm_client,
    output_path: str | Path = "output/qa_dataset.jsonl",
    fmt: str = "jsonl",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    num_pairs_per_chunk: int = 5,
    max_workers: int = 4,
    use_threads: bool = True,
    min_quality_score: float = 0.5,
    question_key: str = "question",
    answer_key: str = "answer",
    include_score: bool = False,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    text_columns: Optional[List[str]] = None,
    save_metrics_csv: bool = False,
    tracker: Optional[BenchmarkTracker] = None,
) -> Dict:
    """Run the full pipeline: ingest → chunk → generate → filter → export.

    Args:
        input_path: Path to the source document (PDF, TXT, or CSV).
        llm_client: A configured LLM client instance.
        output_path: Destination file for the exported dataset.
        fmt: Export format — ``"json"`` or ``"jsonl"``.
        chunk_size: Maximum characters per text chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        num_pairs_per_chunk: Target Q&A pairs per chunk.
        max_workers: Parallel worker count.
        use_threads: Use threads (True) or processes (False).
        min_quality_score: Minimum score [0–1] to retain a pair.
        question_key: Output field name for questions.
        answer_key: Output field name for answers.
        include_score: Include ``_score`` in exported records.
        prompt_template: Custom prompt template (uses ``{chunk}`` and ``{num_pairs}``).
        text_columns: CSV column names to extract (None = all string columns).
        save_metrics_csv: Save per-chunk benchmark metrics as a CSV alongside output.
        tracker: Optional :class:`~qa_generator.benchmarks.BenchmarkTracker`.

    Returns:
        A summary dict with keys: ``pairs``, ``output_path``, ``metrics``.
    """
    if tracker is None:
        tracker = BenchmarkTracker()

    tracker.start_run()

    # 1. Ingest
    logger.info("Ingesting: %s", input_path)
    text = ingest(input_path, text_columns=text_columns)

    # 2. Chunk
    logger.info("Chunking text (size=%d, overlap=%d) …", chunk_size, chunk_overlap)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("Total chunks: %d", len(chunks))

    # 3. Generate Q&A
    logger.info("Generating Q&A pairs (%d workers, %d pairs/chunk) …", max_workers, num_pairs_per_chunk)
    raw_results = generate_qa_parallel(
        chunks=chunks,
        llm_client=llm_client,
        num_pairs=num_pairs_per_chunk,
        prompt_template=prompt_template,
        max_workers=max_workers,
        use_threads=use_threads,
        tracker=tracker,
    )

    # Flatten all pairs
    all_pairs: List[Dict] = []
    for result in raw_results:
        for pair in result.get("pairs", []):
            pair["chunk_index"] = result["chunk_index"]
            all_pairs.append(pair)

    # 4. Quality filter
    logger.info("Filtering %d raw pairs (min_score=%.2f) …", len(all_pairs), min_quality_score)
    filtered = filter_pairs(
        all_pairs,
        q_key="question",
        a_key="answer",
        min_score=min_quality_score,
    )

    # 5. Export
    logger.info("Exporting %d pairs → %s", len(filtered), output_path)
    written = export(
        filtered,
        output_path=output_path,
        fmt=fmt,
        question_key=question_key,
        answer_key=answer_key,
        include_score=include_score,
    )

    tracker.stop_run()
    tracker.log_summary()

    # Optional metrics CSV
    if save_metrics_csv:
        metrics_path = Path(output_path).with_suffix(".metrics.csv")
        tracker.save_csv(metrics_path)

    return {
        "pairs": filtered,
        "output_path": str(written),
        "metrics": tracker.summary(),
    }
