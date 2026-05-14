# TASKS.md

## Phase 0 - Project Scaffold

- [x] Create folder structure.
- [x] Add Streamlit app skeleton.
- [x] Add YAML config loader.
- [x] Add Ollama client wrapper.
- [x] Add sample data directory.

## Phase 1 - Synthetic Data

- [x] Generate JSONL scene transition samples.
- [x] Validate JSON with pydantic.
- [x] Filter invalid or low-quality samples.
- [x] Add preview table in Streamlit.
- [x] Diversify samples with rotating narrative plans.
- [x] Reuse cached samples by dataset key.

## Phase 2 - Embeddings

- [x] Embed scene_t.summary.
- [x] Embed scene_t_plus_1.summary.
- [x] Save embeddings as .npz.
- [x] Build FAISS index for next-scene examples.
- [x] Reuse cached embeddings and unchanged FAISS index.

## Phase 3 - Latent Predictor

- [x] Implement MLP predictor.
- [x] Add cosine loss.
- [x] Add train/validation split.
- [x] Save best checkpoint.
- [x] Show loss chart in GUI.
- [x] Add practical residual MLP sizing and CUDA FP32 training.
- [x] Add dropout, weight decay, gradient clipping, and early stopping.

## Phase 4 - Generation

- [x] Implement LLM-only mode.
- [x] Implement RAG + LLM mode.
- [x] Implement JEPA Planner + RAG + LLM mode.
- [x] Display predicted next-scene direction via retrieved examples.
- [x] Display retrieved examples through generated prompts and reports.
- [x] Add long-form Chat tab with persistent sessions.
- [x] Add hybrid recent-message, summary, and knowledge-graph memory for long context.
- [x] Add session Markdown export.
- [x] Add beat-card grounded prompting for RAG/JEPA generation.
- [x] Limit retrieved examples injected into prompts.
- [x] Add optional name consistency repair after generation.

## Phase 4.5 - Chat Memory

- [x] Save chat sessions as JSON files.
- [x] Compress long sessions into memory summaries.
- [x] Extract and merge knowledge graph nodes/edges.
- [x] Show graph tables and Mermaid text.
- [x] Support LLM-only, RAG, and JEPA modes in chat generation.

## Phase 5 - Evaluation

- [x] Repetition rate.
- [x] Embedding continuity score.
- [x] Keyword consistency score.
- [x] Name consistency score and issue list.
- [x] Simple contradiction checklist.
- [x] Add lexical diversity, novelty, progression, length fit, dialogue ratio, sentence stats, ranking, and pairwise output diversity.
- [ ] Optional LLM judge.
- [x] Markdown report export.

## Phase 6 - Demo Hardening

- [x] Add dry-run mode.
- [x] Add sample dataset via dry-run pipeline.
- [x] Add clear error handling.
- [x] Add intuitive pipeline progress, artifact status, cache counters, and live training progress.
- [x] Add final demo report.
- [x] Commit stable version.
