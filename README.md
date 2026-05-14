# Novel JEPA Lab

JEPA-inspired latent planner + local LLM Korean novel generation lab.

This project provides a Streamlit GUI where a user can generate synthetic Korean narrative transition data, validate/filter it, embed scene summaries, train a small embedding-space predictor, and compare three generation modes:

1. LLM-only
2. RAG + LLM
3. JEPA-inspired Planner + RAG + LLM

The local LLM is not fine-tuned. Only a small PyTorch MLP learns to map the current scene embedding to the next scene embedding.

## Workflow

```text
Streamlit GUI button
-> Ollama generates synthetic scene transition JSONL
-> pydantic validates and filters samples
-> Ollama embeds current and next scene summaries
-> FAISS indexes next-scene embeddings
-> PyTorch predictor trains in latent space
-> predictor retrieves likely next-scene directions
-> local LLM writes Korean prose
-> metrics compare LLM-only, RAG, and JEPA modes
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Make sure Ollama is running locally and that your selected chat/embedding models are installed. The GUI also includes dry-run mode, which exercises the pipeline without Ollama.

For RTX 4060/CUDA acceleration, install the CUDA build after creating the virtual environment:

```bash
pip install -r requirements-gpu.txt
```

## Run

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

On Windows, you can also double-click:

```text
run_server.bat
```

## Streamlit Tabs

- Project: one-click dry-run or Ollama-backed full pipeline
- Dataset: generate and filter JSONL transition samples
- Embedding: embed summaries and build the FAISS index
- Train: train the JEPA-inspired MLP predictor
- Generate: run LLM-only, RAG, or JEPA planner generation
- Evaluate: write a Markdown comparison report
- Reports: view saved reports

## Current Pipeline Behavior

- Synthetic samples are diversified with rotating scene plans: subgenre, plot function, emotion arc, conflict, motif, and relationship tension.
- `data/synthetic/sample_cache.jsonl` stores generated samples by model, genre, sample id, and diversity plan. Re-running the same request reuses matching samples instead of calling Ollama again.
- `data/embeddings/embedding_cache.jsonl` stores text embeddings by embedding model and text hash. The embedding stage only calls Ollama for missing vectors.
- `data/embeddings/scenes.npz` is reused when the filtered dataset and embedding model are unchanged.
- The FAISS index is reused when it is newer than the embedding file.
- The predictor defaults to a practical residual MLP: hidden dim 1024, 4 layers, dropout, weight decay, early stopping, gradient clipping, and CUDA FP32 when available.
- AMP is optional and disabled by default because the small predictor usually does not benefit enough to justify CUDA compatibility risk.
- Evaluation reports include mode ranking, embedding continuity, repetition profile, keyword consistency, novelty from previous scene, lexical diversity, length fit, progression score, dialogue ratio, sentence stats, contradiction checks, and pairwise output diversity.

## Artifacts

- `data/synthetic/generated.jsonl`
- `data/synthetic/sample_cache.jsonl`
- `data/filtered/filtered.jsonl`
- `data/embeddings/scenes.npz`
- `data/embeddings/embedding_cache.jsonl`
- `data/indexes/next_scene.faiss`
- `checkpoints/predictor/best.pt`
- `reports/runs/latest_train_history.json`
- `reports/runs/comparison_*.md`

## Notes

This is JEPA-inspired rather than a reproduction of Meta JEPA. It applies the core idea of predicting target representations in a joint embedding space to narrative scene transitions.
