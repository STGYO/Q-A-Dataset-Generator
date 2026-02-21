"""Streamlit web interface for the Q&A Dataset Generator."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Q&A Dataset Generator",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Q&A Dataset Generator")
st.caption("Generate high-quality Q&A datasets from your documents using LLMs.")

# ---------------------------------------------------------------------------
# Sidebar – LLM configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ LLM Configuration")

    llm_mode = st.radio("Mode", ["Online (API)", "Offline (Ollama / LM Studio)"])

    if llm_mode == "Online (API)":
        provider = st.selectbox("Provider", ["OpenAI", "Google Gemini"])
        if provider == "OpenAI":
            model = st.text_input("Model", value="gpt-4o")
            api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
            base_url = st.text_input("Base URL (optional)", value="")
        else:
            model = st.text_input("Model", value="gemini-1.5-flash")
            api_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
            base_url = ""
    else:
        provider = "ollama"
        model = st.text_input("Local Model Name", value="llama3")
        api_key = "ollama"
        base_url = st.text_input("Ollama / LM Studio URL", value="http://localhost:11434/v1")

    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.05)
    max_tokens = st.number_input("Max Tokens", min_value=128, max_value=4096, value=1024, step=128)

    st.divider()
    st.header("🔧 Pipeline Settings")

    chunk_size = st.number_input("Chunk Size (chars)", min_value=200, max_value=8000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap (chars)", min_value=0, max_value=1000, value=200, step=50)
    num_pairs = st.number_input("Q&A Pairs per Chunk", min_value=1, max_value=20, value=5)
    max_workers = st.number_input("Parallel Workers", min_value=1, max_value=16, value=4)
    min_score = st.slider("Min Quality Score", 0.0, 1.0, 0.5, 0.05)

    st.divider()
    st.header("📄 Output Schema")
    question_key = st.text_input("Question Key", value="question")
    answer_key = st.text_input("Answer Key", value="answer")
    export_fmt = st.radio("Export Format", ["jsonl", "json"])
    include_score = st.checkbox("Include Quality Score", value=False)

# ---------------------------------------------------------------------------
# Main area – file upload
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload your document (PDF, TXT, or CSV)",
    type=["pdf", "txt", "csv"],
)

text_columns_input = st.text_input(
    "CSV Text Columns (comma-separated, leave empty for auto-detect)",
    value="",
    help="Only used for CSV files.",
)

run_button = st.button("🚀 Generate Q&A Dataset", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if run_button:
    if not uploaded:
        st.error("Please upload a file first.")
        st.stop()

    with st.spinner("Running pipeline …"):
        # Lazy imports so app starts without heavy deps installed
        from qa_generator.llm import create_llm_client
        from qa_generator.pipeline import run_pipeline

        # Save upload to a temp file
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name

        # Build LLM client
        try:
            llm_client = create_llm_client(
                mode="offline" if provider == "ollama" else "online",
                provider=provider.lower().replace(" ", "").replace("google", ""),
                model=model,
                api_key=api_key or None,
                base_url=base_url or None,
                temperature=float(temperature),
                max_tokens=int(max_tokens),
            )
        except Exception as e:
            st.error(f"Failed to initialise LLM client: {e}")
            st.stop()

        text_cols = [c.strip() for c in text_columns_input.split(",") if c.strip()] or None

        with tempfile.TemporaryDirectory() as out_dir:
            output_path = Path(out_dir) / f"qa_dataset.{export_fmt}"

            try:
                result = run_pipeline(
                    input_path=tmp_path,
                    llm_client=llm_client,
                    output_path=output_path,
                    fmt=export_fmt,
                    chunk_size=int(chunk_size),
                    chunk_overlap=int(chunk_overlap),
                    num_pairs_per_chunk=int(num_pairs),
                    max_workers=int(max_workers),
                    min_quality_score=float(min_score),
                    question_key=question_key,
                    answer_key=answer_key,
                    include_score=include_score,
                    text_columns=text_cols,
                    save_metrics_csv=True,
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

            pairs = result["pairs"]
            metrics = result["metrics"]

            # ----------------------------------------------------------------
            # Results
            # ----------------------------------------------------------------

            st.success(f"✅ Generated **{len(pairs)}** Q&A pairs!")

            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Pairs Generated", metrics["total_pairs_generated"])
            col2.metric("Chunks Processed", metrics["total_chunks"])
            col3.metric("Avg Latency (s)", f"{metrics['avg_latency_seconds']:.3f}")
            col4.metric("Avg TPS", f"{metrics['avg_tokens_per_second']:.1f}")

            # Preview table
            if pairs:
                import pandas as pd

                preview_data = [
                    {question_key: p.get("question", ""), answer_key: p.get("answer", "")}
                    for p in pairs[:20]
                ]
                st.subheader("Preview (first 20 pairs)")
                st.dataframe(pd.DataFrame(preview_data), use_container_width=True)

            # Download buttons
            output_bytes = output_path.read_bytes()
            st.download_button(
                label=f"⬇️ Download {export_fmt.upper()}",
                data=output_bytes,
                file_name=f"qa_dataset.{export_fmt}",
                mime="application/json",
            )

            # Metrics CSV
            metrics_path = output_path.with_suffix(".metrics.csv")
            if metrics_path.exists():
                st.download_button(
                    label="⬇️ Download Metrics CSV",
                    data=metrics_path.read_bytes(),
                    file_name="qa_metrics.csv",
                    mime="text/csv",
                )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.divider()
st.markdown(
    """
    **Q&A Dataset Generator** | 
    Supports PDF · TXT · CSV | 
    Online (OpenAI, Gemini) & Offline (Ollama, LM Studio) LLMs
    """
)
