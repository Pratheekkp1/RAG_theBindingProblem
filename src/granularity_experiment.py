# src/granularity_experiment.py
from __future__ import annotations

import json
from typing import List, Dict
import numpy as np

from sentence_transformers import SentenceTransformer

from .rag import BindingRAG
from .config import EMBEDDING_MODEL_NAME, PROJECT_ROOT


QUESTIONS = [
    "What is the neuronal binding problem in visual perception?",
    "Compare feature integration theory and temporal synchrony accounts of binding.",
    "What evidence supports the role of attention in feature binding?",
    "How does gamma-band synchrony relate to perceptual binding?",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_experiment():
    print("Loading embedding model for answer similarity...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    rag_small = BindingRAG(granularity="small")
    rag_medium = BindingRAG(granularity="medium")
    rag_large = BindingRAG(granularity="large")

    results: Dict[str, Dict] = {}

    for q in QUESTIONS:
        print(f"\nQuestion: {q}")
        ans_small = rag_small.answer(q)
        ans_medium = rag_medium.answer(q)
        ans_large = rag_large.answer(q)

        results[q] = {
            "small": ans_small,
            "medium": ans_medium,
            "large": ans_large,
        }

        emb_small = embedder.encode([ans_small], convert_to_numpy=True)[0]
        emb_medium = embedder.encode([ans_medium], convert_to_numpy=True)[0]
        emb_large = embedder.encode([ans_large], convert_to_numpy=True)[0]

        sim_s_m = cosine_similarity(emb_small, emb_medium)
        sim_m_l = cosine_similarity(emb_medium, emb_large)
        sim_s_l = cosine_similarity(emb_small, emb_large)

        results[q]["similarity"] = {
            "small_vs_medium": round(sim_s_m, 4),
            "medium_vs_large": round(sim_m_l, 4),
            "small_vs_large": round(sim_s_l, 4),
        }

        print(f"Similarity (small vs medium): {sim_s_m:.3f}")
        print(f"Similarity (medium vs large): {sim_m_l:.3f}")
        print(f"Similarity (small vs large): {sim_s_l:.3f}")

    out_path = PROJECT_ROOT / "results" / "granularity_experiment.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nExperiment complete. Results saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
