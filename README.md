# Novel JEPA Lab

JEPA-inspired latent planner + local LLM Korean novel generation lab.

This project provides a Streamlit GUI where a user can generate synthetic Korean narrative transition data, validate/filter it, embed scene summaries, train a small embedding-space predictor, and compare three generation modes:

1. LLM-only
2. RAG + LLM
3. JEPA-inspired Planner + RAG + LLM

The local LLM is not fine-tuned. Only a small PyTorch MLP learns to predict the next narrative-state representation from a structured current narrative context.

## Workflow

```text
Streamlit GUI button
-> Ollama generates synthetic scene transition JSONL
-> pydantic validates and filters samples
-> Ollama embeds structured current context and next-state target text
-> FAISS indexes next-scene embeddings
-> PyTorch predictor trains to predict target representations in latent space
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
The `Ollama 500 recovery` expander controls the automatic recovery path for intermittent runner crashes. When `/api/chat` returns a recoverable 500-class error before any streamed text is shown, the app unloads the chat and embedding models, waits briefly, then retries once with conservative fallback options: `num_gpu=35`, `num_ctx=3072`, `num_batch=64`, `num_predict=1200`, and `keep_alive=10s`.

### 5. Run the full experiment

Open the `Project` tab.

1. Pick a genre preset, or choose `직접 입력` and enter your own genre.
2. Leave `Scene preset` on `자동 순환` for a first run. The preset list resets automatically when the genre changes.
3. Use the sample recommendation row in the UI. `quick` is for smoke checks, `balanced` is the default JEPA diagnostic range, `research` is better for a graduation-project comparison, and `robust` is for slower but stabler retrieval experiments. You can also choose a custom scene preset; when a preset changes, the advanced world, characters, and previous-scene fields refresh for that preset.
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

## JEPA-Inspired Planner

This project is not a faithful reproduction of Meta JEPA. It is a JEPA-inspired narrative latent planner for Korean long-form novel generation:

- Frozen text embedding model: provides the representation space.
- Trainable PyTorch MLP predictor: predicts the next narrative-state representation from structured context.
- FAISS: retrieves likely target scene directions from the predicted representation.
- Local LLM: writes the final Korean prose from the beat card and retrieved direction.
- Inference scene analyzer: structures the raw current scene into summary, emotion, conflict, state, plot function, active characters, unresolved clues, and next pressure before embedding.

The comparison modes mean:

- LLM-only: tests pure local LLM generation.
- RAG + LLM: retrieves examples from the current-context index and uses the matched samples' next scenes.
- RAG next-index baseline: embeds the current context directly against the next-scene target index.
- JEPA-inspired Planner + RAG + LLM: predicts a likely next-state representation first, then retrieves target-scene directions from the next-scene index.

Planner diagnostics report validation-first metrics. The report keeps all-sample metrics as `all_*`, but the headline `pred_target_cosine`, `retrieval_hit_at_k`, `retrieval_mean_score`, and transition diversity are taken from the saved validation split when available. This avoids presenting train-set retrieval as generalization.

When `normalize_prediction=True`, cosine alignment is the main objective and norm regularization is disabled internally (`effective_loss_norm_weight=0`). Norm regularization is meaningful only for `normalize_prediction=False` experiments.

## Current Pipeline Behavior

- Synthetic samples are diversified with genre-specific scene presets. Each built-in genre has multiple preset situations with plot function, emotion arc, conflict, motif, relationship tension, scene goal, and next hook.
- The embedding stage builds structured context encoder input from world, character, current scene, and preset metadata, and target encoder input from the next narrative state.
- Training supports context/field dropout, delta prediction, output normalization, and a JEPA-style representation prediction loss.
- Training checkpoints store the base-sample train/validation indices used by planner diagnostics.
- Evaluation reports include a `Planner Diagnostics` section with validation predicted-target cosine, retrieval hit@k, RAG-current/RAG-next/JEPA-next baselines, retrieval overlap, diversity, and predicted vector norm.
- The project builds both `current_context.faiss` and `next_scene.faiss` indexes.
- Project, Dataset, Generate, and Chat tabs can apply the same scene preset idea, so the training data and generated prose can follow matching genre situations.
- Changing genre resets stale scene preset selections to `자동 순환` and refreshes one-click demo inputs with matching world, characters, and previous scene defaults.
- Changing the scene preset refreshes Project and Generate advanced inputs with preset-specific world notes, relationship tension, scene goal, and next hook. If a browser session keeps stale text, click `Apply selected scene preset` to force-reset the visible inputs.
- Scene preset selectors support custom direct input; custom labels are passed through as custom scene-preset metadata instead of being silently replaced by automatic rotation.
- Project and Dataset tabs provide genre presets plus a custom genre option.
- The sidebar can load installed Ollama models from `/api/tags`, and Ollama API errors include the endpoint, model, HTTP status, and response detail.
- Synthetic data generation requests Ollama JSON mode and extracts the first valid balanced JSON object if the model still adds surrounding text.
- Synthetic samples now include explicit diversity axes: pacing, point-of-view distance, stakes scale, clue type, relationship shift, transition shape, and pressure source. These axes are stored in metadata and included in planner context text.
- The Project and Dataset tabs show recommended sample counts plus diversity coverage. As a rule of thumb: quick runs use roughly 8-24 samples, balanced JEPA diagnostics use about 32-96, research demos use about 96-216, and robust experiments use 192+ depending on genre preset count.
- Dataset generation uses a candidate multiplier. If some samples fail JSON/schema validation, the generator can try extra candidate ids without requiring a manual rerun.
- Synthetic JSON generation defaults to a lower max token budget than prose generation to speed up sample creation while keeping the schema compact.
- RAG/JEPA generation now feeds the LLM a compact beat card instead of dumping all retrieved context into prose.
- `Prompt examples` limits how many retrieved examples enter the prompt, while `Retrieval top K` still controls the search pool.
- Name consistency checks compare generated outputs against the character list and report unknown or likely misspelled names.
- Optional auto-repair rewrites only detected name inconsistencies before the output is saved.
- Ollama calls can unload the embedding model before chat calls, and unload the chat model before embedding calls, to keep VRAM pressure predictable.
- Chat generation has a recovery retry for Ollama 500-class runner failures. The retry is only attempted before streamed output begins, so the UI does not duplicate partially generated prose.
- The trained JEPA predictor checkpoint stores model weights plus training metadata/history; `checkpoints/predictor/model_card.json` mirrors the latest training summary for inspection.
- `data/synthetic/sample_cache.jsonl` stores generated samples by model, genre, sample id, and diversity plan. Re-running the same request reuses matching samples instead of calling Ollama again.
- The GUI includes a sample cache browser under `Reports / Storage -> Cache / artifacts`, so you can inspect cached sample ids, genre, preset, plot function, and key prefix. Exact cache reuse requires the same schema/model/genre/sample slot/diversity plan; legacy-compatible reuse can reuse older cache rows when the genre/sample slot/preset/plot function still line up.
- `data/embeddings/embedding_cache.jsonl` stores text embeddings by embedding model and text hash. The embedding stage only calls Ollama for missing vectors.
- Dry-run embeddings and real Ollama embeddings are cached under separate backend keys, so a dry-run FAISS index is not reused for real RAG retrieval.
- `data/embeddings/scenes.npz` is reused when the filtered dataset and embedding model are unchanged.
- The FAISS index is reused when it is newer than the embedding file.
- The predictor defaults to a practical residual MLP: hidden dim 1024, 4 layers, dropout, weight decay, early stopping, gradient clipping, and CUDA FP32 when available.
- AMP is optional and disabled by default because the small predictor usually does not benefit enough to justify CUDA compatibility risk.
- Evaluation reports include mode ranking, embedding continuity, repetition profile, keyword consistency, novelty from previous scene, lexical diversity, length fit, progression score, dialogue ratio, sentence stats, contradiction checks, and pairwise output diversity.
- The full pipeline view shows a live stage table, current step message, artifact snapshot, cache reuse counts, and live training loss/cosine charts while training runs.
- The full pipeline and Generate tab stream prose output while Ollama is generating, then replace the live text with the final consistency-checked text.
- In the Generate tab and the full-pipeline `RAG live` / `JEPA live` tabs, RAG and JEPA modes show a live pipeline trace before and during the answer: scene analysis, retrieval, JEPA target prediction, prompt assembly, generation, and consistency repair. This is a system trace, not hidden model chain-of-thought.
- The full pipeline has its own training controls in the Project tab. The default predictor training budget is 80 epochs with early stopping patience 12, so the small JEPA-inspired MLP can train longer without always running every epoch.
- If the training graph ends before the requested epoch count, early stopping fired because validation cosine did not improve for the patience window. Set patience to `0` or enable `Run all requested epochs` to force the full epoch count.
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
