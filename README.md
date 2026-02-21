# Q-A-Dataset-Generator

A **production-grade automated Q&A dataset generator** that turns raw documents (PDF, TXT, CSV) into high-quality question-answer pairs using online or offline LLMs.

## Features

| Feature | Details |
|---|---|
| **Input formats** | PDF (PyMuPDF), TXT, CSV (pandas) |
| **Web UI** | Streamlit – upload files, configure LLM, download results |
| **CLI** | Typer-based CLI with full option support |
| **Online LLMs** | OpenAI (GPT-4o, etc.), Google Gemini |
| **Offline LLMs** | Ollama, LM Studio (OpenAI-compatible local endpoints) |
| **Text chunking** | LangChain `RecursiveCharacterTextSplitter` with configurable size/overlap |
| **Parallel generation** | `ThreadPoolExecutor` / `ProcessPoolExecutor` with configurable worker count |
| **Quality filtering** | Scoring, deduplication, min-score threshold |
| **Custom schema** | User-defined question/answer key names |
| **Export** | JSON and JSONL formats |
| **Benchmarking** | Per-chunk latency, tokens-per-second, metrics CSV |

---

## Project Structure

```
qa_generator/          # Core package
  __init__.py
  ingestion.py         # PDF / TXT / CSV ingestion
  chunking.py          # Overlapping text chunking
  llm.py               # LLM client factory (OpenAI, Gemini, Ollama)
  generator.py         # Parallel Q&A pair generation
  quality.py           # Quality scoring & deduplication
  exporter.py          # JSON / JSONL export with schema customisation
  pipeline.py          # End-to-end pipeline orchestration
  benchmarks.py        # Latency & TPS tracking

app.py                 # Streamlit web UI
cli.py                 # Typer CLI

tests/                 # pytest test suite
requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For PDF support install **PyMuPDF** (preferred) or pypdf:
```bash
pip install pymupdf          # or: pip install pypdf
```

For Google Gemini:
```bash
pip install google-generativeai
```

### 2. Web UI (Streamlit)

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually http://localhost:8501).  
Upload a PDF/TXT/CSV file, configure your LLM in the sidebar, and click **Generate Q&A Dataset**.

### 3. CLI

```bash
# Online – OpenAI GPT-4o
python cli.py generate my_document.pdf \
  --model gpt-4o \
  --api-key sk-... \
  --output dataset.jsonl \
  --num-pairs 5

# Offline – Ollama (llama3 running locally)
python cli.py generate my_document.pdf \
  --mode offline \
  --provider ollama \
  --model llama3 \
  --output dataset.jsonl

# Google Gemini
python cli.py generate my_document.txt \
  --provider gemini \
  --model gemini-1.5-flash \
  --api-key AIza...

# CSV with specific text columns + custom schema
python cli.py generate data.csv \
  --text-column title --text-column body \
  --question-key Q --answer-key A \
  --format json \
  --save-metrics
```

Run `python cli.py generate --help` for all options.

---

## Pipeline Stages

```
Input file
   │
   ▼
┌─────────────────────────────────┐
│  1. Ingestion                   │  PDF → PyMuPDF / pypdf
│     TXT → built-in I/O          │  CSV → pandas
└───────────────┬─────────────────┘
                │ raw text
                ▼
┌─────────────────────────────────┐
│  2. Chunking                    │  RecursiveCharacterTextSplitter
│     chunk_size / chunk_overlap  │  (LangChain or built-in)
└───────────────┬─────────────────┘
                │ text chunks[]
                ▼
┌─────────────────────────────────┐
│  3. Parallel Q&A Generation     │  ThreadPoolExecutor / ProcessPoolExecutor
│     LLM prompt → parse pairs    │  OpenAI / Gemini / Ollama
└───────────────┬─────────────────┘
                │ raw pairs[]
                ▼
┌─────────────────────────────────┐
│  4. Quality Filter              │  score_pair() → deduplicate()
│     min_score threshold         │
└───────────────┬─────────────────┘
                │ filtered pairs[]
                ▼
┌─────────────────────────────────┐
│  5. Export                      │  JSON array or JSONL (one obj/line)
│     custom schema keys          │  + optional metrics CSV
└─────────────────────────────────┘
```

---

## Configuration Reference

### LLM Modes

| Mode | Provider | Notes |
|---|---|---|
| `online` | `openai` | Requires `OPENAI_API_KEY` env var or `--api-key` |
| `online` | `gemini` | Requires `GEMINI_API_KEY` env var or `--api-key` |
| `offline` | `ollama` | Default base URL: `http://localhost:11434/v1` |
| `offline` | any | Custom `--base-url` for LM Studio or other OpenAI-compatible servers |

### Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key (used as default by CLI) |
| `GEMINI_API_KEY` | Google Gemini API key |

---

## Quality Scoring

Each generated pair is scored on three criteria (score ∈ [0, 1]):

1. **Minimum length** – question ≥ 10 chars, answer ≥ 5 chars  
2. **Answer length cap** – answer ≤ 2000 chars  
3. **Question heuristic** – ends with `?` or contains an interrogative word

Pairs with an empty question or answer score **0.0** and are always filtered out.  
Pairs below `--min-score` (default 0.5) are discarded before export.

---

## Benchmarking

With `--save-metrics`, a `.metrics.csv` file is written alongside the output containing per-chunk:
- `latency_seconds` – total LLM call latency
- `tokens_per_second` – completion tokens / latency
- `prompt_tokens` / `completion_tokens`
- `pairs_generated` / `error`

---

## Running Tests

```bash
pytest tests/ -v
```

All 43 tests should pass. No LLM API keys required (tests use mocks).

---

## Extending

- **Add a new LLM provider**: Subclass `BaseLLMClient` in `qa_generator/llm.py` and register it in `create_llm_client()`.
- **Custom prompt**: Pass a `--prompt-template` file or use the `prompt_template` argument in `run_pipeline()`.
- **Ray parallelism**: Replace `ThreadPoolExecutor` in `generator.py` with a Ray remote task for distributed workloads.
- **Haystack integration**: Swap the generator step with a Haystack `PromptNode` for retrieval-augmented Q&A generation.
