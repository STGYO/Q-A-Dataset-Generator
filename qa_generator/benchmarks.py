"""Benchmarking metrics: latency, throughput, and structured logging."""

from __future__ import annotations

import csv
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------


class Timer:
    """Simple wall-clock timer usable as a context manager."""

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self.start


# ---------------------------------------------------------------------------
# Per-chunk result tracker
# ---------------------------------------------------------------------------


@dataclass
class ChunkResult:
    chunk_index: int
    pairs_generated: int
    error: Optional[str]
    latency_seconds: float
    tokens_per_second: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class BenchmarkTracker:
    """Collects per-chunk metrics during a pipeline run."""

    results: List[ChunkResult] = field(default_factory=list)
    _total_timer: Optional[Timer] = field(default=None, repr=False, compare=False)

    def start_run(self) -> None:
        self._total_timer = Timer()
        self._total_timer.start = time.perf_counter()

    def stop_run(self) -> None:
        if self._total_timer is not None:
            self._total_timer.elapsed = time.perf_counter() - self._total_timer.start

    @property
    def total_elapsed(self) -> float:
        if self._total_timer:
            return self._total_timer.elapsed
        return 0.0

    def record(self, chunk_result: Dict) -> None:
        """Record a chunk result dict (as returned by :func:`~qa_generator.generator.generate_qa_for_chunk`)."""
        self.results.append(
            ChunkResult(
                chunk_index=chunk_result.get("chunk_index", -1),
                pairs_generated=len(chunk_result.get("pairs", [])),
                error=chunk_result.get("error"),
                latency_seconds=chunk_result.get("latency_seconds", 0.0),
                tokens_per_second=chunk_result.get("tokens_per_second", 0.0),
                prompt_tokens=chunk_result.get("prompt_tokens", 0),
                completion_tokens=chunk_result.get("completion_tokens", 0),
            )
        )

    # ------------------------------------------------------------------
    # Aggregated stats
    # ------------------------------------------------------------------

    @property
    def total_chunks(self) -> int:
        return len(self.results)

    @property
    def successful_chunks(self) -> int:
        return sum(1 for r in self.results if r.error is None)

    @property
    def failed_chunks(self) -> int:
        return sum(1 for r in self.results if r.error is not None)

    @property
    def total_pairs(self) -> int:
        return sum(r.pairs_generated for r in self.results)

    @property
    def avg_latency(self) -> float:
        lats = [r.latency_seconds for r in self.results if r.error is None]
        return sum(lats) / len(lats) if lats else 0.0

    @property
    def avg_tps(self) -> float:
        tps_vals = [r.tokens_per_second for r in self.results if r.tokens_per_second > 0]
        return sum(tps_vals) / len(tps_vals) if tps_vals else 0.0

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.results)

    def summary(self) -> Dict:
        return {
            "total_chunks": self.total_chunks,
            "successful_chunks": self.successful_chunks,
            "failed_chunks": self.failed_chunks,
            "total_pairs_generated": self.total_pairs,
            "avg_latency_seconds": round(self.avg_latency, 4),
            "avg_tokens_per_second": round(self.avg_tps, 2),
            "total_completion_tokens": self.total_completion_tokens,
            "total_elapsed_seconds": round(self.total_elapsed, 4),
        }

    def log_summary(self) -> None:
        s = self.summary()
        logger.info(
            "Pipeline summary: chunks=%d (ok=%d, fail=%d), pairs=%d, "
            "avg_latency=%.3fs, avg_TPS=%.1f, total_elapsed=%.2fs",
            s["total_chunks"],
            s["successful_chunks"],
            s["failed_chunks"],
            s["total_pairs_generated"],
            s["avg_latency_seconds"],
            s["avg_tokens_per_second"],
            s["total_elapsed_seconds"],
        )

    def save_csv(self, path: str | Path) -> Path:
        """Save per-chunk metrics to a CSV file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "chunk_index",
            "pairs_generated",
            "error",
            "latency_seconds",
            "tokens_per_second",
            "prompt_tokens",
            "completion_tokens",
        ]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(
                    {
                        "chunk_index": r.chunk_index,
                        "pairs_generated": r.pairs_generated,
                        "error": r.error or "",
                        "latency_seconds": r.latency_seconds,
                        "tokens_per_second": r.tokens_per_second,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                    }
                )
        logger.info("Saved benchmark CSV → %s", path)
        return path
