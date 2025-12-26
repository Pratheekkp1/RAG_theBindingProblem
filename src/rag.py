# src/rag.py
from __future__ import annotations

from typing import List, Literal

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from .config import INDEX_DIR, EMBEDDING_MODEL_NAME, OPENAI_MODEL, OPENAI_API_KEY, GRANULARITIES


Granularity = Literal["small", "medium", "large"]


class BindingRAG:
    def __init__(self, granularity: Granularity = "medium"):
        if granularity not in GRANULARITIES:
            raise ValueError(f"Unknown granularity: {granularity}")

        self.granularity = granularity

        self.client = chromadb.PersistentClient(
            path=str(INDEX_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection_name = f"binding_{granularity}"
        self.collection = self.client.get_collection(self.collection_name)

        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        if OPENAI_API_KEY is None:
            raise RuntimeError("OPENAI_API_KEY not set in environment.")
        self.llm = OpenAI(api_key=OPENAI_API_KEY)

    def retrieve(self, query: str, k: int = 6):
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True)[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )
        # Flatten
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        ids = results["ids"][0]
        return ids, docs, metadatas

    def build_context(self, docs: List[str], metadatas: List[dict]) -> str:
        parts = []
        for i, (d, m) in enumerate(zip(docs, metadatas)):
            label = f"CHUNK {i} | paper_id={m.get('paper_id')} | granularity={m.get('granularity')}"
            parts.append(label + "\n" + d.strip())
        return "\n\n" + ("\n\n".join(parts))

    def answer(self, question: str, k: int = 6) -> str:
        ids, docs, metadatas = self.retrieve(question, k=k)
        context = self.build_context(docs, metadatas)

        prompt = f"""
You are an expert assistant on the neuronal binding problem and related neuroscience.
Answer the user's question using ONLY the information in the context below.
If the context does not contain the answer, say you cannot confirm it from the provided documents.

Cite specific chunks by their CHUNK label when relevant (e.g., "see CHUNK 2").
Be precise and avoid speculation.

CONTEXT:
{context}

QUESTION: {question}
"""

        completion = self.llm.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a careful, citation-focused neuroscience assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        return completion.choices[0].message.content.strip()


def interactive_cli():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--granularity",
        type=str,
        choices=list(GRANULARITIES.keys()),
        default="medium",
    )
    args = parser.parse_args()

    rag = BindingRAG(granularity=args.granularity)

    print(f"Neuronal Binding Problem RAG | granularity={args.granularity}")
    print("Type your question (or 'exit'):")

    while True:
        try:
            q = input(">> ").strip()
        except EOFError:
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        answer = rag.answer(q)
        print("\n--- ANSWER ---")
        print(answer)
        print("--------------\n")


if __name__ == "__main__":
    interactive_cli()
