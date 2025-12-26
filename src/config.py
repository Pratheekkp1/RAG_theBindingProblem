# src/config.py
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = PROJECT_ROOT / "indexes"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Embedding model (local)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Granularity configurations (tokens ~ proportional to chars)
GRANULARITIES = {
    "small": {"chunk_chars": 800, "overlap_chars": 150},
    "medium": {"chunk_chars": 1500, "overlap_chars": 250},
    "large": {"chunk_chars": 2500, "overlap_chars": 400},
}

# OpenAI config (or other LLM)
OPENAI_MODEL = "gpt-4o-mini"  # adjust if desired
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
