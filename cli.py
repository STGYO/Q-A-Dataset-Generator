"""Command-line interface for the Q&A Dataset Generator."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="qa-generator",
    help="Generate Q&A datasets from PDF, TXT, or CSV files using LLMs.",
    add_completion=False,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def generate(
    input_file: Path = typer.Argument(..., help="Path to the source document (PDF, TXT, or CSV)."),
    output: Path = typer.Option(
        Path("output/qa_dataset.jsonl"),
        "--output",
        "-o",
        help="Output file path.",
    ),
    fmt: str = typer.Option(
        "jsonl",
        "--format",
        "-f",
        help="Export format: 'json' or 'jsonl'.",
    ),
    # LLM options
    mode: str = typer.Option(
        "online",
        "--mode",
        "-m",
        help="LLM mode: 'online' (API) or 'offline' (Ollama/LM Studio).",
    ),
    provider: str = typer.Option(
        "openai",
        "--provider",
        "-p",
        help="LLM provider: 'openai', 'gemini', or 'ollama'.",
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model",
        help="Model name or identifier.",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="OPENAI_API_KEY",
        help="API key (or set via OPENAI_API_KEY / GEMINI_API_KEY env vars).",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="Override LLM API base URL (e.g. http://localhost:11434/v1 for Ollama).",
    ),
    temperature: float = typer.Option(0.3, "--temperature", "-t", help="Sampling temperature."),
    max_tokens: int = typer.Option(1024, "--max-tokens", help="Maximum tokens to generate."),
    # Pipeline options
    chunk_size: int = typer.Option(1000, "--chunk-size", help="Characters per text chunk."),
    chunk_overlap: int = typer.Option(200, "--chunk-overlap", help="Overlap between chunks."),
    num_pairs: int = typer.Option(5, "--num-pairs", "-n", help="Q&A pairs per chunk."),
    max_workers: int = typer.Option(4, "--workers", "-w", help="Parallel worker count."),
    min_score: float = typer.Option(0.5, "--min-score", help="Minimum quality score [0–1]."),
    # Schema options
    question_key: str = typer.Option("question", "--question-key", help="Output key for questions."),
    answer_key: str = typer.Option("answer", "--answer-key", help="Output key for answers."),
    include_score: bool = typer.Option(False, "--include-score", help="Include _score in output."),
    # CSV options
    text_columns: Optional[List[str]] = typer.Option(
        None,
        "--text-column",
        "-c",
        help="CSV column(s) to use as text (repeatable).",
    ),
    # Metrics
    save_metrics: bool = typer.Option(False, "--save-metrics", help="Save per-chunk metrics CSV."),
    # Misc
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Generate a Q&A dataset from a document."""

    _setup_logging(verbose)

    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    # Resolve API key from environment when not provided
    if api_key is None:
        if provider == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
        else:
            api_key = os.getenv("OPENAI_API_KEY")

    console.print(f"[bold cyan]Q&A Dataset Generator[/bold cyan]")
    console.print(f"  Input    : {input_file}")
    console.print(f"  Output   : {output}")
    console.print(f"  LLM      : {provider}/{model} ({mode})")
    console.print(f"  Format   : {fmt.upper()}")
    console.print()

    from qa_generator.llm import create_llm_client
    from qa_generator.pipeline import run_pipeline

    try:
        llm_client = create_llm_client(
            mode=mode,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        console.print(f"[red]Failed to create LLM client:[/red] {e}")
        raise typer.Exit(1)

    try:
        with console.status("[bold green]Running pipeline…"):
            result = run_pipeline(
                input_path=input_file,
                llm_client=llm_client,
                output_path=output,
                fmt=fmt,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                num_pairs_per_chunk=num_pairs,
                max_workers=max_workers,
                use_threads=True,
                min_quality_score=min_score,
                question_key=question_key,
                answer_key=answer_key,
                include_score=include_score,
                text_columns=text_columns or None,
                save_metrics_csv=save_metrics,
            )
    except Exception as e:
        console.print(f"[red]Pipeline error:[/red] {e}")
        raise typer.Exit(1)

    pairs = result["pairs"]
    metrics = result["metrics"]

    # Summary table
    table = Table(title="Pipeline Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="dim")
    table.add_column("Value")
    for k, v in metrics.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)

    console.print(
        f"\n[bold green]✓[/bold green] {len(pairs)} pairs saved → [cyan]{result['output_path']}[/cyan]"
    )


@app.command()
def version() -> None:
    """Show the package version."""
    from qa_generator import __version__

    console.print(f"qa-generator {__version__}")


if __name__ == "__main__":
    app()
