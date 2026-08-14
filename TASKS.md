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
- [x] Add genre-specific scene presets for dataset generation.

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
- [x] Refactor predictor into JEPA-inspired context/target representation prediction modules.
- [x] Add context/field dropout, delta prediction, output normalization, and configurable representation loss.
- [x] Add inference current-scene analyzer to reduce train/inference context mismatch.
- [x] Save train/validation indices for validation-first planner diagnostics.
- [x] Add target-relative VISReg-inspired anti-collapse regularization with automatic small-data disable.
- [x] Record normalized effective rank and require it in consumer model promotion.

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
- [x] Stream generated prose into the UI during generation.
- [x] Add longer sectioned novel output with titled sections and per-section body targets.
- [x] Apply selected genre scene presets to LLM-only, RAG, JEPA, and Chat generation.
- [x] Build compact JEPA planner beat cards from predicted target retrieval.
- [x] Split RAG baselines into current-context index and next-scene index retrieval.
- [x] Add synthetic sample diversity axes for JEPA planner training.
- [x] Add controlled hallucination generation mode for creative but planner-grounded expansion.
- [x] Limit the full pipeline comparison to JEPA and controlled hallucination outputs for lower VRAM pressure.
- [x] Make Creative Hallucination + JEPA the only active prose generation mode.
- [x] Generate approximately 30,000 characters through sequential resource-bounded sections.
- [x] Save the long-form draft checkpoint after every completed section.
- [x] Extract a compact continuity record from every generated section without an extra LLM call.
- [x] Add lightweight story-memory RAG for characters, facts, clues, locations, and state changes.
- [x] Save the story-memory ledger after every completed section.
- [x] Add 5,000/10,000/custom-character generation turns.
- [x] Resume the next turn from persistent prose, memory, ledger, and run-state checkpoints.
- [x] Import UTF-8 Markdown/text drafts and rebuild continuity memory for continuation.
- [x] Export and restore portable ZIP bundles containing prose, memory, KG/state, turn progress, and story settings.
- [x] Add latest entity-state tracking and knowledge-graph triples to section memory.
- [x] Add hierarchical multi-section compression without extra LLM calls.
- [x] Fit recent prose, compressed memory, and KG retrieval into an 8K context budget.
- [x] Track consumed narrative beats and include them in every following section prompt.
- [x] Assign exactly one primary narrative function through a per-section direction update.
- [x] Detect repeated plot beats after generation and retry the section once at a lower temperature.
- [x] Report repeated subtitles, repeated beats, adjacent-section similarity, and retry success statistics.
- [x] Persist a hierarchical global story outline with per-beat state-change and setup/payoff constraints.
- [x] Add a resource-bounded stability gate for truncation, names, state transitions, and resolved clues.
- [x] Combine repetition and stability findings into at most one section revision before commit.

## Phase 4.5 - Chat Memory

- [x] Save chat sessions as JSON files.
- [x] Compress long sessions into memory summaries.
- [x] Extract and merge knowledge graph nodes/edges.
- [x] Show graph tables and Mermaid text.
- [x] Support LLM-only, RAG, and JEPA modes in chat generation.
- [x] Support controlled hallucination mode in chat generation.

## Phase 5 - Evaluation

- [x] Repetition rate.
- [x] Embedding continuity score.
- [x] Keyword consistency score.
- [x] Name consistency score and issue list.
- [x] Simple contradiction checklist.
- [x] Add lexical diversity, novelty, progression, length fit, dialogue ratio, sentence stats, ranking, and pairwise output diversity.
- [x] Add planner diagnostics to comparison reports.
- [x] Report validation-first planner metrics and RAG-current/RAG-next/JEPA-next baseline overlap.
- [x] Measure creative expansion, hallucination presence, useful hallucination score, and hallucination risk.
- [x] Measure section count fit and per-section body coverage.
- [x] Optional LLM judge: local-model 1-10 rubric scoring (개연성/창의성/할루시네이션
      통제/일관성/몰입도) with useful-vs-harmful hallucination findings, off by default
      and diagnostic only.
- [x] Markdown report export.

## Phase 6 - Demo Hardening

- [x] Add dry-run mode.
- [x] Add sample dataset via dry-run pipeline.
- [x] Add clear error handling.
- [x] Add intuitive pipeline progress, artifact status, cache counters, and live training progress.
- [x] Add a full-pipeline fresh dataset option.
- [x] Add scene preset controls to Project, Dataset, Generate, and Chat tabs.
- [x] Reset scene presets and one-click defaults when the selected genre changes.
- [x] Add Ollama partial-offload controls and runtime VRAM visibility.
- [x] Add sample-count recommendations, diversity coverage reporting, and candidate-pool retry generation.
- [x] Store JEPA predictor training metadata with the checkpoint.
- [x] Add JEPA-inspired smoke checks for builders, loss, retrieval, and dry-run pipeline.
- [x] Add final demo report.
- [x] Commit stable version.

## Phase 7 - Consumer Long-form Service

- [x] Separate localhost admin UI (`8502`) from LAN/VPN consumer UI (`8501`).
- [x] Add consumer registration/login, scrypt password hashes, server-side sessions, and account-owned stories.
- [x] Isolate every consumer story under `data/consumer_stories/<story_id>/`.
- [x] Add SQLite WAL metadata, FIFO queue, single-job constraints, and worker heartbeat.
- [x] Continue generation after browser disconnect and preserve partial sections after worker failure.
- [x] Add maintenance draining and block candidate training outside active maintenance.
- [x] Add immutable candidate versions, quality gates, SHA-256 verification, and atomic promotion.
- [x] Record per-section creativity, consistency, progression, repetition, memory, and JEPA metrics.
- [x] Restrict admin draft access to research-consented stories while retaining anonymous metrics.
- [x] Add consumer draft/bundle downloads, immediate deletion, and 30-day retention cleanup.
- [x] Add separate Windows web/worker tasks and CI-gated self-hosted deployment.
- [x] Add consumer store, worker recovery, quality-gate, 30K fake generation, and AppTest coverage.
- [x] Redesign the consumer login/library/chat surface around ordinary account-based use.
- [x] Store per-section generation stability and revision metrics.
- [x] Let consumers choose the overall character target and per-turn generation length.
- [x] Give the final target section an explicit central-conflict resolution contract.
- [x] Guarantee the ending: finish the finale in the turn that crosses the target, persist
      `novel_completed` in run state and `stories.completed_at`, and gate completion on the
      written ending instead of the character count.
- [x] Make the creativity dial real: scale sampling temperature from the selected
      hallucination target so 안정/균형/대담 differ mechanically, not only in prompt text.
- [x] Send `top_p` and `repeat_penalty` on prose calls to curb long-form phrase looping.
- [x] Let the per-story outline own section plot so every novel no longer follows the same
      hardcoded 15-phase spine.
- [x] Add genre-specific beat trigger tables so the repetition guard works outside SF/thriller,
      and thread the story genre from the consumer through generation.
- [x] Score narrative plausibility with the trained JEPA predictor: compare each section's
      realized next-state embedding against the predicted one and route low-cosine sections
      into the bounded revision pass.
- [x] Calibrate the plausibility threshold on real generated prose (9 genuine sections vs
      7 deliberate causal breaks) and ship `scripts/calibrate_coherence.py` so the number
      can be re-measured after changing the embedding model or retraining the predictor.
- [x] Keep the chat model resident during the gate's embedding call, cutting its overhead
      from 7.4% to 2.9% of section generation time.
- [x] Record plausibility per section in `section_metrics`, not only as a per-turn average.
- [x] Report the plausibility gate in the Markdown evaluation report.
- [x] Rescore an accepted rewrite so the stored score never describes discarded prose.
- [x] Offset the primary-function cycle by a story seed so different premises do not share
      the same function order.
- [x] Retrain and promote the service model on 112 samples across four genres, lifting the
      validation split from 6 to 16 rows and hit@5 from 0.667 to 0.938.
- [x] Recalibrate the plausibility threshold against the promoted model over two genres
      (0.69 -> 0.64) after confirming that genre shifts the cosine scale.
- [x] Let consumers read and edit the settings a story was created with — title, genre,
      premise, world, protagonist, characters, target length, research consent — from the
      writing screen. Edits apply from the next turn; written sections are left alone.
- [x] Let consumers delete a finished chat turn, clear the whole conversation log, reset a
      story's manuscript while keeping its settings, or delete the story outright, and give
      the writing screen its own back link to the library. Every path deletes rows rather
      than flagging them, and an active turn blocks reset so the worker cannot write into a
      story that was just cleared.
- [x] Stop trusting the model to delimit its own private record. Match the whole marker
      family, and strip any unmarked or truncated record before prose reaches a reader,
      so `<STORY_MEMORY>` JSON can never appear in the novel.
- [x] Stream prose to the consumer UI. The worker writes the in-flight section to a live
      file that the polling fragment renders, so a reader sees text arriving instead of
      a two-minute silence per section.
- [x] Size the Ollama context window from a measurement instead of assumption: the prose
      prompt reaches ~14.6K characters late in a story, so `num_ctx` 8192 was silently
      truncating the front of the prompt, where the world and character canon live.
      Raised to 16384 and the recovery fallback to 8192.
