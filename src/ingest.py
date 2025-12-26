# src/ingest.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

from pypdf import PdfReader
from tqdm import tqdm

from .config import PDF_DIR, GRANULARITIES


@dataclass
class Chunk:
    id: str
    paper_id: str
    granularity: str
    text: str


def extract_text_from_pdf(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(txt)
    return "\n".join(pages)


def simple_clean(text: str) -> str:
    # Minimal cleaning; you can extend this later
    text = text.replace("\r", " ")
    # collapse multiple spaces/newlines
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def chunk_text(
    text: str,
    chunk_chars: int,
    overlap_chars: int,
    paper_id: str,
    granularity: str,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    i = 0
    n = len(text)

    counter = 0
    while i < n:
        start = max(0, i - overlap_chars if chunks else i)
        end = min(n, start + chunk_chars)
        chunk_str = text[start:end].strip()
        if chunk_str:
            chunk_id = f"{paper_id}::{granularity}::{counter}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    paper_id=paper_id,
                    granularity=granularity,
                    text=chunk_str,
                )
            )
            counter += 1
        i = end
    return chunks


def build_chunks_for_all_pdfs() -> Dict[str, List[Chunk]]:
    """
    Returns: dict[granularity] -> list[Chunk]
    """
    pdf_paths = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {PDF_DIR}. Put binding-problem papers there."
        )

    all_chunks: Dict[str, List[Chunk]] = {g: [] for g in GRANULARITIES.keys()}

    for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
        paper_id = pdf_path.stem  # filename without extension
        text = extract_text_from_pdf(pdf_path)
        text = simple_clean(text)

        for gran_name, cfg in GRANULARITIES.items():
            chunks = chunk_text(
                text=text,
                chunk_chars=cfg["chunk_chars"],
                overlap_chars=cfg["overlap_chars"],
                paper_id=paper_id,
                granularity=gran_name,
            )
            all_chunks[gran_name].extend(chunks)

    return all_chunks
