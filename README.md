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

## Beginner Quickstart

### 1. Start Ollama and install models

Make sure Ollama is running. Install the default chat and embedding models:

```powershell
ollama pull gemma4:e4b
ollama pull embeddinggemma
```

### 2. Launch the app

Double-click:

```text
run_server.bat
```

Or run from PowerShell:

```powershell
cd C:\프로그래밍_프로젝트\novel_jepa_lab
.\run_server.bat
```

Open:

```text
http://127.0.0.1:8501
```

### 3. Test with dry-run mode first

In the left sidebar:

```text
Dry-run mode: on
Reuse cached data: on
```

Dry-run mode checks the UI and pipeline flow without calling Ollama.

### 4. Use real local models

When dry-run works, turn it off:

```text
Dry-run mode: off
Ollama base URL: http://localhost:11434
Chat model: gemma4:e4b
Embedding model: embeddinggemma:latest
```

If Ollama is reachable, the sidebar shows installed models as dropdowns. You can still choose `직접 입력` when you want to type a model name manually.
The default chat model is `gemma4:e4b`. On the target RTX 4060 8GB setup this model should be treated as a partial-offload model, not a full-VRAM model. The app sends controlled Ollama options by default: `num_gpu=40`, `num_ctx=4096`, `num_batch=128`, and `keep_alive=30s`.
Use the sidebar `Ollama runtime` expander to check loaded models, approximate GPU residency, VRAM size, and context length. If `model runner has unexpectedly stopped` appears, lower `Ollama GPU layers`, `Ollama context length`, or `Ollama batch size`.

### 5. Run the full experiment

Open the `Project` tab.

1. Pick a genre preset, or choose `직접 입력` and enter your own genre.
2. Leave `Scene preset` on `자동 순환` for a first run. The preset list resets automatically when the genre changes.
3. Set `Samples` to `8` to `20` for a first real test.
4. Turn on `Create fresh dataset for this run` when you want to ignore the synthetic sample cache and make a new dataset.
5. Click `Run Full Pipeline`. The app fills genre-matched world, characters, and previous scene defaults automatically.

The app runs:

```text
generate synthetic data
-> filter JSONL
-> embed summaries
-> build FAISS index
-> train JEPA-inspired predictor
-> generate LLM/RAG/JEPA outputs
-> write evaluation report
```

### 6. Write a long-form story

Open the `Chat` tab.

1. Create a new session.
2. Pick the session genre. World setting and characters are filled with genre-matched defaults.
3. Choose `JEPA Planner + RAG + LLM` for the default long-form mode.
4. Pick a scene preset when you want the next scene to follow a specific situation.
5. Write a next-scene instruction.
6. Click `Generate next scene`.

Example instruction:

```text
이전 장면의 감정선을 이어서 다음 장면을 써 주세요.
새 단서와 선택 압박을 포함해 주세요.
```

The right-side Memory panel shows:

- message count
- scene summaries
- compressed memory summary
- knowledge graph nodes and edges
- Mermaid graph text

Useful buttons:

- `Compress now`: update the long-term memory summary.
- `Rebuild graph`: rebuild the knowledge graph from the session.
- `Export MD`: export the session to Markdown.
- `Delete`: delete the current session.

### 7. Recommended first workflow

```text
1. Start with Dry-run mode on.
2. Confirm the app works.
3. Turn Dry-run mode off.
4. Generate 20 samples in Dataset or Project.
5. Run Embedding.
6. Train the predictor.
7. Use Chat with JEPA Planner + RAG + LLM.
```

Keep `Reuse cached data` on. The first run is slower; later runs reuse samples and embeddings.

## Streamlit Tabs

- Project: one-click dry-run or Ollama-backed full pipeline
- Chat: long-form writing session with compressed memory and knowledge graph
- Dataset: generate and filter JSONL transition samples
- Embedding: embed summaries and build the FAISS index
- Train: train the JEPA-inspired MLP predictor
- Generate: run LLM-only, RAG, or JEPA planner generation
- Evaluate: write a Markdown comparison report
- Reports: view saved reports

## Current Pipeline Behavior

- Synthetic samples are diversified with genre-specific scene presets. Each built-in genre has multiple preset situations with plot function, emotion arc, conflict, motif, relationship tension, scene goal, and next hook.
- Project, Dataset, Generate, and Chat tabs can apply the same scene preset idea, so the training data and generated prose can follow matching genre situations.
- Changing genre resets stale scene preset selections to `자동 순환` and refreshes one-click demo inputs with matching world, characters, and previous scene defaults.
- Project and Dataset tabs provide genre presets plus a custom genre option.
- The sidebar can load installed Ollama models from `/api/tags`, and Ollama API errors include the endpoint, model, HTTP status, and response detail.
- Synthetic data generation requests Ollama JSON mode and extracts the first valid balanced JSON object if the model still adds surrounding text.
- RAG/JEPA generation now feeds the LLM a compact beat card instead of dumping all retrieved context into prose.
- `Prompt examples` limits how many retrieved examples enter the prompt, while `Retrieval top K` still controls the search pool.
- Name consistency checks compare generated outputs against the character list and report unknown or likely misspelled names.
- Optional auto-repair rewrites only detected name inconsistencies before the output is saved.
- Ollama calls can unload the embedding model before chat calls, and unload the chat model before embedding calls, to keep VRAM pressure predictable.
- The trained JEPA predictor checkpoint stores model weights plus training metadata/history; `checkpoints/predictor/model_card.json` mirrors the latest training summary for inspection.
- `data/synthetic/sample_cache.jsonl` stores generated samples by model, genre, sample id, and diversity plan. Re-running the same request reuses matching samples instead of calling Ollama again.
- `data/embeddings/embedding_cache.jsonl` stores text embeddings by embedding model and text hash. The embedding stage only calls Ollama for missing vectors.
- `data/embeddings/scenes.npz` is reused when the filtered dataset and embedding model are unchanged.
- The FAISS index is reused when it is newer than the embedding file.
- The predictor defaults to a practical residual MLP: hidden dim 1024, 4 layers, dropout, weight decay, early stopping, gradient clipping, and CUDA FP32 when available.
- AMP is optional and disabled by default because the small predictor usually does not benefit enough to justify CUDA compatibility risk.
- Evaluation reports include mode ranking, embedding continuity, repetition profile, keyword consistency, novelty from previous scene, lexical diversity, length fit, progression score, dialogue ratio, sentence stats, contradiction checks, and pairwise output diversity.
- The full pipeline view shows a live stage table, current step message, artifact snapshot, cache reuse counts, and live training loss/cosine charts while training runs.
- The full pipeline and Generate tab stream prose output while Ollama is generating, then replace the live text with the final consistency-checked text.
- Chat sessions are stored as JSON files under `data/sessions/` and keep messages, scene summaries, a long-term memory summary, story state, and a knowledge graph.
- Long-form generation uses hybrid context compression: recent messages + accumulated summary + graph nodes/edges + optional RAG/JEPA retrieval.
- The Chat tab can compress memory on demand, rebuild the graph, show graph nodes/edges plus Mermaid text, and export a session Markdown file.

## Artifacts

- `data/synthetic/generated.jsonl`
- `data/synthetic/sample_cache.jsonl`
- `data/filtered/filtered.jsonl`
- `data/embeddings/scenes.npz`
- `data/embeddings/embedding_cache.jsonl`
- `data/indexes/next_scene.faiss`
- `data/sessions/*.json`
- `checkpoints/predictor/best.pt`
- `checkpoints/predictor/model_card.json`
- `reports/runs/latest_train_history.json`
- `reports/runs/comparison_*.md`
- `reports/runs/session_*.md`

## Notes

This is JEPA-inspired rather than a reproduction of Meta JEPA. It applies the core idea of predicting target representations in a joint embedding space to narrative scene transitions.
