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
- [x] Stop the honorific pattern from reading role address as an invented character:
      아가씨/도련님/선생님 matched `HONORIFIC_RE` and every such section was rewritten by
      the stability gate for obeying the prompt's own "use role names" instruction. A stem
      before 씨/님/군/양 now only counts when it resembles a known character's name.
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

## Phase 8 - 자체 모델 서버 + ChatGPT 식 두 모드 (2026-08-23)

- [x] `model_server/`: Qwen3.5-4B 4bit 모델 서버 (OpenAI 호환 SSE, GQA-split 어텐션, 110K 문맥,
      한국어 토큰 필터 normal/strict, LoRA 어댑터 + 배율, 단일 GPU 직렬화, 끊김 시 생성 중단).
- [x] `src/llm/local_client.py`: OllamaClient 호환 표면 + 스트리밍 제너레이터 + dry-run.
- [x] `llm:` 설정 섹션과 `runtime.make_llm_client`; 워커·헬스체크·런처를 로컬 백엔드로 전환.
- [x] 소비자 경로에서 JEPA 산출물 게이트 제거 (`configure_story_run` 은 paths 없는 manifest 허용).
- [x] `src/generation/longform.py`: 이야기 지도 → 장별 과제 → 생성 → 게이트(1회 재생성) →
      별도 메모리 호출 → 원자 저장. 잘린 아웃라인/메모리 JSON 복구.
- [x] `consumer_app.py` 재작성: 사이드바 모드(일반 채팅 / 장편 소설), 계정별 대화 저장(`ChatStore`),
      말투 선택, 자유 입력 기획 대화 → 작품 카드 → 집필 시작, 기존 집필 화면 유지.
- [x] 테스트: 클라이언트(SSE/dry-run), 생성기(가짜 LLM), 채팅 저장소, 새 UI AppTest 시나리오.
- [x] 디코드 속도: CUDA 그래프 디코드(정적 버퍼 캐시, 단일 GPU 스레드)로 10 → 30~48 tok/s.
      3,000자 턴 451초 → 113~153초. 버킷 기본 32K(NOVEL_QWEN_MAX_BUCKET, 20K ctx 실측
      37 tok/s; 65536 도 실측 OK) — 초과 문맥은 eager 폴백, CUDA 오류 시 자동 강등.
- [x] 연재 품질: DRY 샘플러(rep12 3.7→1.3, distinct3 0.77→0.87, 기본 0.8/1.75/2) +
      메모리 JSON 간결 스키마·잘린 JSON 복구·한국어 필터 적용.
