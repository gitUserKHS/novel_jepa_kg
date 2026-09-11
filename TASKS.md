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
- [x] 채팅 목록·CRUD (2026-08-23): 사이드바 날짜 묶음·검색·열린 대화 강조·전체 삭제, 빈 대화 재사용,
      이름 바꾸기·내보내기·비우기·삭제(팝오버 2단계 확인), 턴 삭제·고쳐 보내기·답변 다시 생성·끊긴 답변
      다시 받기, 제안 문장, 토스트. `ChatStore` 검색/`find_empty_chat`/`delete_messages_from`/`delete_all_chats`,
      `src/utils/timefmt.py`. AppTest `consumer_chat_crud` 시나리오.
- [x] 원고 편집 (2026-08-23): `src/service/story_editor.py` — 장 단위 직접 수정 / AI 퇴고 스트림(기본 모델,
      strict 한국어 필터) / 요약 메모리 재추출·직접 편집. 저장 시 draft·memory·ledger·state.json·DB 진행 지표를
      함께 갱신, 집필 중이면 거부(`get_editable_story`). UI: `✏️ 원고 수정` 패널(폴링 fragment 바깥) + 장별 `✏️`.
- [x] 자동 이어쓰기 실측 (2026-08-24, Qwen3.5-4B, 1만 자 목표 GL 일상물 2회): 두 번 다 자동 턴 3개·5장으로 완결까지
      (3.6~4.5분), 완결 뒤 대기 턴 없음. 지시는 구체적·작품에 맞음("유리가 그림에 숨겨둔 트라우마를 선아에게 털어놓게 해줘").
      1차에 직전 지시와 소재가 겹쳐 본문이 되풀이됨 → 제안 프롬프트에 [이미 시킨 전개] 블록(최근 6개) 추가, 2차는 겹침 없음.
      4B 는 설정에 없는 과거(이혼·병원 사고)를 지어내는 경향 — 지시문이 아니라 생성 단계의 환각. 1차 1장 "66회 반복 + 미완"
      조합은 걷어내기 조건 밖이었음 → TRIMMABLE_ISSUES 로 반복+미완도 걷어내고 마지막 문장까지 되돌림.
- [x] 자동 이어쓰기 (2026-08-24): `stories.auto_continue` + `jobs.origin('user'|'auto')`. 워커가 턴을 마치면
      `_maybe_continue` 가 같은 설정으로 자동 턴을 넣고, 자동 턴을 집을 때 `auto_continue.propose_next_direction`
      (이야기 지도의 다음 beat + 계층 요약 + 미해결 단서 → 지시 한두 문장) 으로 지시를 채워 말풍선에 남긴다.
      완결·`auto_continue_max_turns`(기본 20)·점검이면 멈춤. UI: 켜기/끄기 + `다음 한 턴만 AI 에게`.
- [x] 게이트 재생성 투명화 (2026-08-24): 실측 사례 "4장: 같은 구절이 6회 반복됨" — 재생성 때 실시간 본문이
      설명 없이 지워져 "장이 날아간" 것으로 보였다. 반복만 문제면 `trim_repetitions` 로 걷어내고 채택(재생성
      없음), 재생성 시 사유·버려진 초안·채택 결정을 `live_note.json` 으로 화면에 전달(`LiveProseWriter.note_section`),
      끝난 턴엔 `metrics_json.retry_notes`. 버려진 초안 격리는 `test_longform_retry_isolation.py` 가 고정.
      `enable_stability_retry` 가 이제 Qwen 생성기에도 적용된다.
- [x] AI 퇴고 품질 실측 (2026-08-23, Qwen3.5-4B, 7월 원고 2천 자 장): 라벨만 주면 복사(유사도 0.99), "짧게
      끊어라" 는 토막·반토막, "분량 유지" 는 다시 복사, 대화 늘리기는 반복 루프. 해법: 요청 유형 3모드(최소/문체/
      부분) + 문체 모드는 `plan_edits`(바꿀 곳 JSON, 잘린 JSON 복구, 1회 재시도) → 계획 반영 재작성 + 분량 범위
      밖이면 `retry_note_for` 로 1회 재시도. 결과: 감각 묘사 1.0~1.2배·유사도 0.80~0.90, 긴장감 0.9배·0.82,
      부분 수정 0.99(요청 줄만 변경), 오탈자 0.995. 편차는 남는다(긴장감 한 번은 0.44배) — UI 가 비율을 표시.
      메모리 재추출은 8~10초, 요약·사실·단서·상태까지 정확.
- [x] 코드 리뷰 반영 (2026-08-23): 편집 시 state.json 의 turns_completed 감소·완결 해제 버그, 장 번호/메모리
      내용 해시로 위젯 키(턴 삭제·재추출 뒤 옛 입력이 다른 장 위에 저장되던 문제), 사람 본문은 모델용 정리기를
      거치지 않음, 죽은 '다시 시도' 버튼, 첫 턴 enqueue 실패 시 거짓 성공 토스트, 집필 지침을 [집필 지침] 블록으로
      (세계관 시트·story_seed 오염 제거), 선택상자는 session_state 대입으로 동기화, 검색창 비우기, 런타임 캐시,
      내려받기 지연 생성, `client.toolbarMode=minimal`, stStatusWidget 숨김, 빈 대화 제목 초기화, 필드 한도 공유.
- [x] 템플릿 직접 작성 (2026-08-23): 기획 화면 `📝 직접 작성` 탭(양식 + 집필 지침), `src/service/template_store.py`
      (계정별 `story_templates`, 같은 이름 덮어쓰기), 작품 설정에서 템플릿 저장. `stories.style_guide` 열 추가 —
      워커·편집기가 공유하는 `story_sheets.world_sheet` 를 통해 매 장 프롬프트의 [작품 설정]에 붙는다.
- [x] 사이드바가 사라지던 버그 (2026-08-23): Streamlit 1.57 은 접힌 사이드바를 펴는 `»` 버튼을 툴바 안에
      그리는데 앱 CSS 가 `stToolbar` 전체를 숨겨 좁은 창(768px 이하)에서 본문을 누르거나 `«` 를 누르면
      다시 열 수 없었다. 이제 배포 버튼·⋮ 메뉴(`stToolbarActions`)만 숨긴다.
- [x] 디코드 속도: CUDA 그래프 디코드(정적 버퍼 캐시, 단일 GPU 스레드)로 10 → 30~48 tok/s.
      3,000자 턴 451초 → 113~153초. 버킷 기본 32K(NOVEL_QWEN_MAX_BUCKET, 20K ctx 실측
      37 tok/s; 65536 도 실측 OK) — 초과 문맥은 eager 폴백, CUDA 오류 시 자동 강등.
- [x] 연재 품질: DRY 샘플러(rep12 3.7→1.3, distinct3 0.77→0.87, 기본 0.8/1.75/2) +
      메모리 JSON 간결 스키마·잘린 JSON 복구·한국어 필터 적용.

## Phase 9 - Gemma4 26B-A4B (Ollama) · 채팅 제거 · 개연성 장치 (2026-09-03)

- [x] 모델 교체: 소설 백엔드를 Ollama 의 `hf.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M`
      (MoE 25.2B, 활성 4B, 17GB, 262K 문맥)로. `src/llm/ollama_chat.OllamaChatClient` 가 `LocalLLMClient` 와 같은 표면을
      제공하고 `runtime.make_llm_client` 가 `llm.backend`(`ollama` 기본 | `local` 레거시)로 고른다. 워커는 항상
      `generate_longform` 을 쓰고 JEPA 매니페스트를 요구하지 않는다. 런처는 `run_ollama.ps1` 로 Ollama 확인·모델 확인·
      사전 적재를 한다 (`run_service.bat` 이 백엔드에 따라 분기).
- [x] Ollama 실측 (09-03, RTX 4060 8GB, ollama 0.33.2): 모델 16.9 GiB 중 VRAM 4.6 GiB, 디코드 27~30 tok/s, 프리필
      533 tok/s(8.7K 토큰 16.4s), 같은 접두사 재요청 0.4s(KV 캐시), 한국어 1.5~1.6 자/토큰. **think=false 필수** —
      없으면 320 토큰이 전부 thinking 필드로 가고 content 는 빈다. `format=json` 정상. `num_ctx` 는 요청마다 명시(기본
      4096 은 장 프롬프트를 자른다), 바꾸면 모델을 다시 올리므로 16384 로 고정.
- [x] 일반 채팅 제거: `consumer_app.py` 의 모드 전환·채팅 화면·말투 선택·`ChatStore`(+테스트) 삭제, 로그인 직후 기획 화면.
      `llm.chat_*`/어댑터 설정과 `NOVEL_LLM_CHAT_ADAPTER` 제거. 기존 DB 의 `chats` 테이블은 그대로 두되 읽지 않는다.
- [x] 개연성 장치 (`src/generation/plausibility.py`, `longform.py`): (1) 장을 쓰기 전에 **장 설계**(목표·장애·전환·결과·
      가져다 쓰는 사실·되풀이 금지) JSON 을 받아 작가 프롬프트에 넣는다. (2) 쓴 뒤 **연속성 편집자** 역할의 모델이
      확정 사실·상태 원장·직전 장면과의 모순, 원인 없는 사건, 연속성, 설계 실현을 1~10 으로 검토한다. (3) 모순이 하나라도
      있거나 점수 < `plausibility_min_score`(7) 이면 문제 목록을 들고 **고쳐 쓰기**(초안을 넣고 문제만 바로잡음) → 재검토
      → 더 나으면 채택, 아니면 처음 판 유지 + 결정 기록. 검토·설계 호출 실패는 본문을 막지 않는다. 점수·모순·결정은
      `retry_reasons`(🧭)·`section_metrics.plausibility_score`·`jobs.metrics_json.mean/min_plausibility` 로 남고, 고쳐 쓰는
      동안 화면은 `live_note.kind=repair` 로 사유와 고치기 전 초안을 보인다.
- [x] 실제 턴 실측 (09-03, 1만 자 목표 미스터리, 첫 턴 3,755자·2장·이야기 지도 12 beat): 276초, 게이트·고쳐 쓰기 0회,
      두 장 모두 검토 10/10. 그런데 본문에서 개 콩떡이 "하린아, 또 그거 보고 있어?" / "하린아, 으르렁!" 이라고 말하는
      대목을 검토가 놓쳤다 — "동물이 말하면 모순" 이라는 일반 지시를 넣어도 여전히 놓침(9~10점). 검토 절차에 **모든 대사를
      화자에 배정**(dialogue: quote·speaker·can_speak)을 넣자 두 장 모두에서 잡아냈고(can_speak=false → 모순 자동 추가),
      같은 장을 고쳐 쓰기 프롬프트로 다시 쓰면 개의 대사가 서술로 바뀐다. 교훈: 자기 글의 오류는 열거시켜야 보인다.
- [x] 접두사 캐시 설계: 장 단위 호출(설계·생성·검토·기록)이 `NOVEL_SYSTEM_PROMPT` 와 `_canon_prefix`(작품 설정 → 인물 →
      이야기 지도 → 줄거리·상태 → 지나간 사건 → 직전 장면 끝 → 집필 지침)를 공유하고 역할 지시는 끝에 붙인다. JSON 모드는
      시스템 프롬프트를 건드리지 않는다. 문맥 예산(`prompt_char_budget`)을 넘으면 메모리 → 지나간 사건 → 꼬리 순으로 줄인다.
- [x] 이야기 지도: `outline_beat_count` 8 → 12, JSON 모드, 원인 → 결과 사슬 규칙 추가. 메모리 추출 프롬프트에 `relations`
      와 "이 장에서 새로 확정된 사실만" 규칙.
- [x] 테스트: `test_ollama_chat_client.py`(NDJSON·think·format·힌트 위치·사용량·상태), `test_plausibility_gate.py`(설계 →
      생성 → 검토 → 고쳐 쓰기 계약, 실패 내성, 문맥 예산, dry-run 전체 경로), 기존 생성기·프론트 테스트 갱신,
      채팅 테스트 삭제. `health_check.py` 는 설정된 백엔드를 검사한다.
