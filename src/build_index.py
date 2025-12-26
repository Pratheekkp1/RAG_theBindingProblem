# src/build_index.py
from typing import Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import INDEX_DIR, EMBEDDING_MODEL_NAME, GRANULARITIES
from .ingest import build_chunks_for_all_pdfs, Chunk


def get_client():
    client = chromadb.PersistentClient(
        path=str(INDEX_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    return client


def main():
    print("Loading embedding model...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Building chunks for all PDFs...")
    chunks_by_granularity: Dict[str, List[Chunk]] = build_chunks_for_all_pdfs()

    client = get_client()

    for gran, chunks in chunks_by_granularity.items():
        if not chunks:
            continue

        collection_name = f"binding_{gran}"
        print(
            f"\nPreparing collection '{collection_name}' with {len(chunks)} chunks")

        # (re)create collection
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        col = client.create_collection(name=collection_name)

        texts = [c.text for c in chunks]
        ids = [c.id for c in chunks]
        metadatas = [
            {"paper_id": c.paper_id, "granularity": c.granularity} for c in chunks
        ]

        print("Computing embeddings...")
        embeddings = embedder.encode(
            texts, show_progress_bar=True, convert_to_numpy=True)

        # Add in batches
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            col.add(
                ids=ids[i: i + batch_size],
                documents=texts[i: i + batch_size],
                metadatas=metadatas[i: i + batch_size],
                embeddings=embeddings[i: i + batch_size].tolist(),
            )

        print(f"Finished collection '{collection_name}'.")

    print("\nAll indexes built.")


if __name__ == "__main__":
    main()
