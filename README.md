# Neuroscience RAG: The Binding Problem

A retrieval-augmented generation (RAG) system that answers questions about the **neuronal binding problem** (how the brain combines color, shape and motion into one coherent percept), grounded strictly in a corpus of research papers with chunk-level citations.

It also runs a small **experiment on chunk granularity**: the same questions are answered with small, medium and large chunks to see how chunk size changes what the model says.

## Pipeline

```
research PDFs
   │  pypdf text extraction → cleanup
   ▼
chunking at 3 granularities (sliding window with overlap)
   small 800 chars / 150 overlap · medium 1500 / 250 · large 2500 / 400
   │  all-mpnet-base-v2 sentence embeddings (local)
   ▼
ChromaDB — one persistent collection per granularity
   │  top-k semantic retrieval
   ▼
GPT-4o-mini, temperature 0.1, instructed to answer ONLY from context
and cite chunks ("see CHUNK 2"), or to say it can't confirm the answer
```

## Design choices

- **Grounded, citation-first prompting.** The model must answer only from the retrieved context and admit when the papers don't support an answer, which reduces hallucination on a specialist topic.
- **Local embeddings** (`sentence-transformers`), so indexing costs nothing and the corpus never leaves the machine; only the final top-k context goes to the LLM.
- **Granularity as a parameter.** Each chunk size gets its own Chroma collection, so you can compare retrieval strategies without re-indexing.

## Granularity experiment

`src/granularity_experiment.py` asks four questions (e.g. *"Compare feature integration theory and temporal synchrony accounts of binding"*) at each granularity. It embeds the three answers and reports pairwise cosine similarity to show how much chunk size changes the content of the answer. Results are saved to `results/granularity_experiment.json`.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...

# 1. Put papers in data/pdfs/, then build the three indexes
python -m src.build_index

# 2. Ask questions interactively (answers print with their source chunks)
python -m src.rag --granularity medium

# 3. Run the granularity comparison
python -m src.granularity_experiment
```

## Project structure

```
src/
  config.py                  paths, embedding model, chunk sizes, LLM config
  ingest.py                  PDF → text → overlapping chunks
  build_index.py             embed chunks → ChromaDB collections
  rag.py                     BindingRAG: retrieve → build context → generate (+ CLI)
  granularity_experiment.py  small vs. medium vs. large comparison
```

## Tech

Python · ChromaDB · sentence-transformers (all-mpnet-base-v2) · OpenAI API · pypdf · NumPy
