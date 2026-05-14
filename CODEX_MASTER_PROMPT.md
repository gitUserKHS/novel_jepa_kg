# Codex Master Prompt Pack

Use these prompts one by one. Do not ask Codex to build everything in a single pass.

## 00 - Project Scaffold

```text
Create a Python project named jepa_novel_lab.

Goal:
Build a Streamlit web GUI for a JEPA-inspired latent planner + local LLM novel generation system.

Requirements:
- Use Python 3.11+.
- Use Streamlit for GUI.
- Use PyTorch for the predictor model.
- Use FAISS for vector search.
- Use Ollama HTTP API for local LLM and embeddings.
- Store synthetic data as JSONL.
- Store configs in YAML.
- Add clear folder structure under src/.
- Add README.md, AGENTS.md, TASKS.md, .env.example, and requirements.txt.
- Add placeholder modules with docstrings and TODOs.
- Do not implement everything yet. Scaffold the project cleanly.

Create:
app.py
configs/default.yaml
src/llm/ollama_client.py
src/llm/prompts.py
src/data/generate_synthetic.py
src/data/scene_schema.py
src/data/validate_jsonl.py
src/data/filter_dataset.py
src/embedding/embed_scenes.py
src/embedding/vector_store.py
src/planner/dataset.py
src/planner/model.py
src/planner/train.py
src/planner/predict.py
src/generation/generate_baseline.py
src/generation/generate_with_rag.py
src/generation/generate_with_jepa.py
src/evaluation/metrics.py
src/evaluation/judge.py
src/evaluation/report.py
src/utils/config.py
src/utils/logging.py
src/utils/paths.py
```

## 01 - Data Generation

```text
Implement the synthetic narrative dataset generation pipeline.

Requirements:
- Use Ollama local chat API.
- Generate JSONL samples with fields: world, characters, scene_t, scene_t_plus_1.
- scene_t and scene_t_plus_1 must include: summary, emotion, conflict, state, plot_function.
- Add robust JSON parsing and retry logic.
- Add validation using pydantic.
- Add a quality filter that removes:
  - invalid JSON
  - empty summaries
  - identical scene_t and scene_t_plus_1
  - samples with missing character goals
  - samples that mention copyrighted franchise names
- Add a Streamlit Dataset tab:
  - genre input
  - number of samples
  - model name
  - generate button
  - progress bar
  - preview table
- Save output to data/synthetic/generated.jsonl.
- Save filtered output to data/filtered/filtered.jsonl.
- Use Korean prompts and Korean outputs by default.
```

## 02 - Embedding and Training

```text
Implement embeddings and JEPA-inspired predictor training.

Requirements:
- Use Ollama /api/embed for embeddings.
- Embed scene_t.summary and scene_t_plus_1.summary.
- Save embeddings as data/embeddings/scenes.npz with arrays:
  - current_embeddings
  - next_embeddings
  - sample_ids
- Implement a PyTorch MLP predictor:
  input: current scene embedding
  output: next scene embedding
- Loss: 1 - cosine_similarity(pred, target).
- Add train/validation split.
- Save checkpoint to checkpoints/predictor/best.pt.
- Save training history to reports/runs/latest_train_history.json.
- Add Streamlit tabs:
  Embedding tab with embed button.
  Train tab with epoch, batch size, learning rate controls.
  Show training loss and validation cosine similarity charts.
```

## 03 - Generation and Evaluation

```text
Implement story generation and evaluation.

Requirements:
Given a user story context and previous scene summary:
1. Embed the previous scene.
2. Predict next scene embedding using the trained predictor.
3. Retrieve top-k similar scene_t_plus_1 examples from FAISS.
4. Build a prompt containing:
   - user world setting
   - character settings
   - previous scene
   - predicted next scene direction
   - retrieved examples
   - style constraints
5. Call Ollama local LLM to generate Korean web novel prose.

Add three generation modes:
- LLM only
- RAG + LLM
- JEPA Planner + RAG + LLM

Add evaluation metrics:
- repetition rate
- embedding continuity score
- keyword consistency score
- simple contradiction checklist
- optional LLM judge prompt

Add Streamlit Generate and Evaluate tabs.
Generate a Markdown report under reports/runs/.
```

## 04 - Streamlit One-click UI

```text
Make the Streamlit GUI feel like a one-click experiment dashboard.

Requirements:
- Sidebar project settings:
  - Ollama base URL
  - chat model name
  - embedding model name
  - output directory
- Main tabs:
  1. Project
  2. Dataset
  3. Embedding
  4. Train
  5. Generate
  6. Evaluate
  7. Reports
- Add a 'Run Full Pipeline' button that runs:
  generate dataset -> validate/filter -> embed -> train -> build FAISS -> generate sample -> evaluate.
- Every stage must save artifacts and be restartable.
- Show progress bars, status messages, and error boxes.
- Never silently fail. Catch exceptions and display readable messages.
- Add a dry-run mode for quick UI testing.
```

## 05 - Hardening and Reports

```text
Harden the project for a graduation-project demo.

Requirements:
- Add logging to reports/runs/<timestamp>/run.log.
- Add config snapshot saving.
- Add sample input data so the GUI can demo without generating thousands of samples.
- Add README screenshots placeholders and usage instructions.
- Add tests for JSON validation and metric functions.
- Add a report generator that compares:
  - LLM-only
  - RAG + LLM
  - JEPA Planner + RAG + LLM
- Report should include:
  - configuration
  - dataset count
  - training loss chart path
  - evaluation table
  - generated examples
  - limitations
- Keep implementation simple and robust for Windows.
```
