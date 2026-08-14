# Novel JEPA Lab

JEPA-inspired latent planner + local LLM Korean long-form novel service and research lab.

The project has two separate Streamlit surfaces:

1. `consumer_app.py`: account-based story library and chat-style long-form generation.
2. `app.py`: localhost-only dataset, embedding, training, evaluation, and service administration.

The research UI can generate synthetic Korean narrative transition data, validate/filter it, embed scene summaries, train a small embedding-space predictor, and generate one long-form planner-backed creative mode:

1. Creative Hallucination + JEPA

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
-> a persistent hierarchical outline assigns the active global story beat
-> local LLM writes one Korean novel section and a compact continuity record
-> story-memory RAG retrieves relevant past facts, clues, locations, and state changes
-> a local stability gate revises truncation, state drift, resolved-clue reopening, or repetition once
-> metrics evaluate the controlled hallucination novel
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

Start the consumer app and its persistent queue worker:

```powershell
.\run_service.bat
```

Start the private admin and training UI separately:

```powershell
.\run_admin.bat
```

- Consumer: `http://<host-PC-IP>:8501`, bound to `0.0.0.0` for trusted LAN/VPN users.
- Admin: `http://127.0.0.1:8502`, unavailable from other PCs.

Consumers create an account or sign in at the service URL. Passwords are stored
as per-user scrypt hashes, login sessions use hashed server-side tokens, and
every story is bound to its owner account in SQLite. Returning users continue
their work from `내 작품`; there is no shared invite code or per-story key.
New stories can choose an overall target from 10,000 to 50,000 Korean
characters in 1,000-character steps. Each chat turn separately chooses a
resource-safe 2,000, 3,000, or 5,000-character generation target.

A story is marked complete only after the generator actually writes the ending
section, never by character count alone. The turn that reaches the overall
target keeps writing until that closing section exists, so a draft cannot stall
at the target with the central conflict still open. Once the ending is written,
`stories.completed_at` is set, the library card shows `완결`, and further turns
are refused instead of continuing past the resolution.

The compatibility wrapper `run_server.bat` now opens the same private admin UI
as `run_admin.bat`. See [DEPLOYMENT.md](DEPLOYMENT.md) for model promotion,
Windows auto-start, health checks, and CI/CD.

### Consumer readiness

Consumer generation is enabled only when an immutable JEPA artifact version has
passed the service quality gate and is recorded in `artifacts/active.json`.

The active model is `20260806T111529Z-7262bfe2`: 112 samples over four genres
(한국형 판타지 미스터리, 한국형 SF 미스터리, 궁중 판타지, 법정 미스터리) with a
16-row validation split. It replaced a 40-sample model whose validation split
was only 6 rows, and it improved retrieval sharply:

| metric | 40 samples | 112 samples |
| --- | --- | --- |
| validation rows | 6 | 16 |
| validation_cosine | 0.883 | 0.880 |
| hit_at_5 | 0.667 | 0.938 |
| normalized_top1_diversity | 0.833 | 0.938 |
| jepa_gain_over_rag_next | — | 0.174 |

Retraining also separated the plausibility distributions that previously
overlapped. Superseded versions stay under `artifacts/versions/`.

To build and promote a new model in the admin `Service` tab:

1. Build at least 40 filtered samples; 96 is recommended.
2. Enter maintenance and wait for the queue to become idle.
3. Train and inspect a service candidate.
4. Promote it only when every gate check passes.

There is no silent RAG-only fallback when the active JEPA model is missing or invalid.

## Beginner Quickstart

### 1. Start Ollama and install models

Make sure Ollama is running. Install the default chat and embedding models:

```powershell
ollama pull gemma4:e4b
ollama pull embeddinggemma
```

### 2. Launch the app

Double-click the admin launcher:

```text
run_admin.bat
```

Or run from PowerShell:

```powershell
cd C:\프로그래밍_프로젝트\novel_jepa_lab
.\run_admin.bat
```

Open:

```text
http://127.0.0.1:8502
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
The default chat model is `gemma4:e4b`. On the target RTX 4060 8GB setup the app starts with `num_gpu=24`, `num_ctx=16384`, `num_batch=32`, and `keep_alive=60s`.

`num_ctx` was raised from 8192 after measuring the assembled prose prompt: it grows to
about 14.6K characters by the sixth continuation turn and converges near 15K, since the
recent-excerpt, story-memory, and consumed-beat budgets are each capped. At roughly
1.5-2 Korean characters per token that is 8-10K input tokens, and `generation.max_tokens`
adds 1800 more, so an 8192 window could not hold a late-story section. `prose_prompt()`
puts `[세계관]` and `[인물]` at the very top, so overflow costs the story canon first while
the trailing output instructions survive. Enable flash attention and a q8_0 KV cache
(below) and the doubled window costs roughly what the old f16 8K cache did.

For the same reason `fallback_num_ctx` is 8192 rather than 4096: the recovery path should
save VRAM through `num_gpu`, `num_batch`, and `num_predict`, not by truncating the canon.

### Ollama server flags for 8GB cards

Flash attention and KV cache quantization are Ollama **server** environment variables, not
per-request options, so they are set outside this repository and need an Ollama restart:

```powershell
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0
```

`OLLAMA_KV_CACHE_TYPE` is ignored unless flash attention is on. `gemma4:e4b` uses grouped
query attention with only 2 KV heads plus sliding-window layers, so its KV cache is a few
hundred MB rather than gigabytes; q8_0 roughly halves it. The model itself is 9.6GB against
8GB of VRAM, so it is always partially offloaded and `num_gpu` remains the dominant speed
knob — raise it a step at a time until the runner stops cleanly loading.
Use the sidebar `Ollama runtime` expander to check loaded models, approximate GPU residency, VRAM size, and context length. If `model runner has unexpectedly stopped` appears, lower `Ollama GPU layers`, `Ollama context length`, or `Ollama batch size`.
The `Ollama 500 recovery` expander controls the automatic recovery path for intermittent runner crashes. The conservative fallback uses `num_gpu=16`, `num_ctx=8192`, `num_batch=16`, `num_predict=1200`, and `keep_alive=10s`.

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
-> generate a sectioned Creative Hallucination + JEPA novel
-> write evaluation report
```

### 6. Generate and continue a long-form story

Open the `Generate` tab.

1. Choose `5,000`, `10,000`, or a custom character target for one turn.
2. Click `Start new novel` for the first turn.
3. Add an optional next-turn direction.
4. Click `Continue next turn` to append to the saved draft.
5. Use `Download continuation bundle` when moving the project to another PC.
6. Use `Load a saved novel for continuation` to restore a `.zip`, `.md`, or `.txt` file.

The continuation survives an app restart because prose, section memories,
compressed timeline, state ledger, knowledge graph, and run state are saved
after every section.

Each section is also replanned with one primary narrative function. A consumed
beat ledger records completed reveals, alliance shifts, warnings, movements,
clue resolutions, threats, and emotional turns. If a new section appears to
announce one of those events again, the generator lowers the temperature and
rewrites that section once before saving it.

A persistent 6-20 beat story outline now supplies the active act, required state
change, setup/payoff, and forbidden repeat to every section. The same bounded
revision pass also checks body completion, duplicate subtitles, character names,
state transitions, and resolved clues. See
[`docs/research/jepa_regularization_longform_stability.md`](docs/research/jepa_regularization_longform_stability.md)
for the research rationale and evaluation protocol.

- A `.zip` continuation bundle restores prose, memory, KG/state, turn progress,
  world setting, characters, and the original previous-scene seed.
- A plain `.md` or `.txt` draft is also accepted. Its compact memories, state
  ledger, KG, and hierarchical summaries are rebuilt locally from section prose.

### 7. Write through a chat session

Open the `Chat` tab.

1. Create a new session.
2. Pick the session genre. World setting and characters are filled with genre-matched defaults.
3. Use the fixed `Creative Hallucination + JEPA` mode.
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

### 8. Recommended first workflow

```text
1. Start with Dry-run mode on.
2. Confirm the app works.
3. Turn Dry-run mode off.
4. Generate 20 samples in Dataset or Project.
5. Run Embedding.
6. Train the predictor.
7. Generate with Creative Hallucination + JEPA.
```

Keep `Reuse cached data` on. The first run is slower; later runs reuse samples and embeddings.

## Streamlit Tabs

- Project: one-click dry-run or Ollama-backed full pipeline
- Chat: long-form writing session with compressed memory and knowledge graph
- Dataset: generate and filter JSONL transition samples
- Embedding: embed summaries and build the FAISS index
- Train: train the JEPA-inspired MLP predictor
- Generate: create 5,000/10,000/custom-character turns and continue the saved novel
- Evaluate: write a Markdown comparison report, optionally with a local LLM judge review
- Reports: view saved reports
- Service: consumer queue, maintenance, worker heartbeat, model gate/promotion, anonymous metrics, and consented drafts

## JEPA-Inspired Planner

This project is not a faithful reproduction of Meta JEPA. It is a JEPA-inspired narrative latent planner for Korean long-form novel generation:

- Frozen text embedding model: provides the representation space.
- Trainable PyTorch MLP predictor: predicts the next narrative-state representation from structured context.
- FAISS: retrieves likely target scene directions from the predicted representation.
- Local LLM: writes the final Korean prose from the beat card and retrieved direction.
- Inference scene analyzer: structures the raw current scene into summary, emotion, conflict, state, plot function, active characters, unresolved clues, and next pressure before embedding.

The codebase retains earlier baseline implementations for research reference, but the active GUI generation mode is Creative Hallucination + JEPA:

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
- Predictor training optionally adds a target-relative VISReg-inspired scale/shape/center loss. It is disabled below 40 samples and records normalized effective-rank diagnostics; the consumer quality gate requires an effective-rank ratio of at least 0.50.
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
- The overall narrative guide defaults to about 30,000 Korean characters, while each click generates a user-selected 5,000/10,000/custom-character turn.
- Creative Hallucination + JEPA is the only active prose mode. JEPA keeps the direction grounded while the LLM adds plausible clues, symbols, sensory details, and emotional inferences.
- The creativity level changes sampling, not just prompt text. `hallucination_target` scales the prose temperature around its default through `hallucination_temperature_span`, so the consumer 안정/균형/대담 profiles sample at roughly 0.83/0.95/1.07. The balanced profile keeps the historical temperature.
- Prose calls also send `ollama.top_p` and `ollama.repeat_penalty`. The repetition penalty attacks long-form phrase looping at sampling time instead of relying only on the after-the-fact repetition guard.
- When the hierarchical outline is active it owns each section's plot, and the section role asks only for scene craft. Disabling the outline restores the fixed 15-phase spine. This stops two competing plot instructions from entering the same prompt and stops every story from following one hardcoded structure.
- Long-form prose is generated one section per Ollama call with a bounded recent-context excerpt, avoiding a single oversized 30,000-character request.
- `reports/runs/creative_longform_latest.md` is updated after every completed section, so a partial draft survives an Ollama failure.
- Each section also produces a compact private continuity record in the same Ollama response. This avoids an extra summarization call, but it means the model decides where its own record begins, so the split never relies on it: `MEMORY_TAG_RE` matches the whole marker family (`<STORY_MEMORY>`, `<<STORY_MEMORY>>`, `</STORY_MEMORY>`, the canonical triple-angle form), and `strip_machine_block()` removes an unmarked or token-truncated record as a final guard. Prose shown to a reader is filtered on both the streaming path and the saved path.
- `data/consumer_stories/<story_id>/live.txt` holds the section being written right now. The worker runs in its own process, so a stream callback cannot reach the browser; the consumer UI polls this file once a second and renders it under the running job, and the worker clears it once the section commits to `draft.md`.
- Story-memory RAG combines relevant section memories with a latest-state ledger, unresolved clue ledger, query-relevant KG triples, and four-section compressed timeline.
- Per-section direction updates choose exactly one primary narrative function and include a bounded consumed-beat summary.
- The repetition guard checks completed sections against consumed reveals, alliance shifts, system warnings, movements, clue resolutions, threats, and emotional turns, then retries a repeated section once at a lower temperature.
- The guard's trigger words are genre-aware. The base table is SF/thriller vocabulary, so `GENRE_BEAT_TRIGGERS` adds each genre's own words for the same seven beats: a court drama matches 밀지/역모/자객, a legal mystery matches 영장/위증/증거. Genre tables extend the base instead of replacing it, and an unknown genre falls back to the base. The consumer passes the story's genre through `configure_story_run`.
- A JEPA plausibility gate scores 개연성 after each section. The trained predictor turns the preceding context into an expected next-state vector, the section's own continuity record is embedded in the training target's shape, and their cosine says whether the section actually follows from its setup. Below `jepa_coherence_min_cosine` the section enters the same bounded revision pass with an explicit causality instruction. Any missing checkpoint or backend failure downgrades the check to unavailable rather than losing the section.
- `mean_jepa_coherence`, `min_jepa_coherence`, and the coherence retry counts are stored per job, and each scored section also carries its own `jepa_coherence_score` in `section_metrics`, so plausibility can be correlated with repetition, stability, and creativity per section.
- Evaluation reports include a `Narrative Plausibility Gate` block under `Planner Diagnostics`.
- An optional LLM judge (`evaluation.use_llm_judge`, off by default) asks the local chat model
  to score each output 1-10 on 개연성, 창의성, 할루시네이션 통제, 설정 일관성, and 몰입도, and to
  list which invented details enriched the story versus contradicted the setup. The judge reads
  a bounded head/tail excerpt so a 30K-character draft fits the 8K context, its scores are
  diagnostic only (they never change `overall_score` or the ranking), and any Ollama failure
  downgrades the section to `unavailable` instead of losing the report. The Evaluate tab has a
  checkbox to enable it per run; dry-run mode returns a fixed stub review.
- An accepted rewrite is always rescored. Otherwise a section replaced for a stability or repetition reason would keep the discarded draft's plausibility score.
- The per-section narrative function still cycles through all six roles, but its phase is offset by a stable story seed, so two different premises do not put the same function in the same slot.

#### Calibrating the plausibility threshold

`jepa_coherence_min_cosine` is **not portable**. Cosine values depend on the embedding model and on how well the predictor was trained, so the default is only valid for the model it was measured on. Recalibrate with:

```bash
python scripts/calibrate_coherence.py --samples 3 --draft reports/runs/creative_longform_latest.md
```

The script writes genuine continuations and deliberate causal breaks from the same story state, scores both with the production gate, and reports the thresholds that separate them.

The shipped default of `0.64` was measured on 2026-08-06 against artifact `20260806T111529Z-7262bfe2` (112 samples) with `embeddinggemma:latest`, across two genres:

| sample | n | min | mean | max |
| --- | --- | --- | --- | --- |
| genuine sections | 6 | 0.651 | 0.708 | 0.766 |
| deliberate causal breaks | 6 | 0.502 | 0.596 | 0.637 |

At `0.64` no genuine section is rewritten and every break is caught. The distributions no longer overlap, which they did on the previous 40-sample model (genuine floor 0.708 against a break at 0.713).

Two cautions. **Genre shifts the scale**: the fantasy-mystery floor was 0.732 while the courtroom floor was 0.651, so a threshold tuned on one genre can wrongly rewrite another. And absolute values are compressed by the embedder — even deliberately mismatched dataset pairs never scored below 0.497 — which is why an intuitively low threshold never fires at all. The gate is a filter for gross non-sequiturs, not a proof of causality.

- The gate keeps the chat model resident while embedding (`jepa_coherence_keep_chat_loaded`). Evicting a multi-GB chat model for the small embedder makes the next section pay a full reload: measured 8.96s per section with the eviction versus 3.47s without, against 121.7s of generation, so the overhead drops from 7.4% to 2.9%. Set it to `false` if VRAM is too tight to hold both.
- Guarded sections still stream immediately in the Generate tab. If a retry is needed, the temporary draft is replaced in place instead of leaving the UI apparently idle or appending duplicate prose.
- Repetition retries require a high-confidence plot-beat match; adjacent-section vocabulary similarity remains an evaluation metric but does not trigger an expensive rewrite by itself.
- Run state and evaluation reports track repeated subtitles, repeated narrative beats, adjacent-section similarity, repetition retry count, and retry success rate.
- The context budget carries only the recent prose excerpt and the most relevant compressed memory/KG slice instead of the full draft. The per-part character caps are what bound the prompt; `num_ctx` never sizes it, so the window must be set wide enough to hold the assembled result.
- `reports/runs/creative_longform_memory.jsonl` is updated after every section and can be inspected separately from the prose checkpoint.
- `reports/runs/creative_longform_ledger.json` stores current states, KG relations, clue status, and hierarchical summaries.
- `reports/runs/creative_longform_state.json` tracks turn and checkpoint progress so continuation works after restarting Streamlit.
- The same run state stores `novel_completed`. A finished draft refuses another continuation turn instead of writing a second ending.
- The Generate tab can export a portable ZIP continuation bundle and import either that bundle or a UTF-8 Markdown/text draft.
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
- Evaluation reports include mode ranking, embedding continuity, repetition profile, narrative-beat repetition, adjacent-section similarity, retry statistics, keyword consistency, novelty from previous scene, lexical diversity, length fit, progression score, dialogue ratio, sentence stats, section structure metrics, contradiction checks, controlled hallucination metrics, and pairwise output diversity.
- The full pipeline view shows a live stage table, current step message, artifact snapshot, cache reuse counts, and live training loss/cosine charts while training runs.
- The full pipeline streams only the Creative Hallucination + JEPA novel to reduce local Ollama VRAM pressure.
- Streaming UI updates are throttled, avoiding a full Streamlit rerender for every generated character.
- The Generate tab and full pipeline show a live trace for JEPA planning and sequential section generation. This is a system trace, not hidden model chain-of-thought.
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
- `reports/runs/creative_longform_latest.md`
- `reports/runs/creative_longform_memory.jsonl`
- `reports/runs/creative_longform_ledger.json`
- `reports/runs/creative_longform_state.json`
- `data/sessions/*.json`
- `data/consumer_stories/<story_id>/draft.md`
- `data/consumer_stories/<story_id>/memory.jsonl`
- `.runtime/consumer.sqlite3`
- `artifacts/candidates/<version>/`
- `artifacts/versions/<version>/`
- `artifacts/active.json`
- `checkpoints/predictor/best.pt`
- `checkpoints/predictor/model_card.json`
- `reports/runs/latest_train_history.json`
- `reports/runs/comparison_*.md`
- `reports/runs/session_*.md`

## Notes

This is JEPA-inspired rather than a reproduction of Meta JEPA. It applies the core idea of predicting target representations in a joint embedding space to narrative scene transitions.
