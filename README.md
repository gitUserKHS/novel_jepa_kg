# Novel Studio (novel_jepa_lab)

Ollama 의 **Gemma4 26B-A4B**(MoE 25.2B · 활성 4B · Q4_K_M · 17GB) 로 돌아가는 한국어 장편 소설 서비스.
RTX 4060 8GB 한 장에서 모델(VRAM ~4.6 GiB + RAM)·생성 워커·웹을 모두 띄운다.

2026-09-03 에 모델을 자체 Qwen3.5-4B 모델 서버에서 Ollama 의 Gemma4 26B-A4B 로 바꾸고, 소비자 앱의 일반
채팅을 없애 장편 소설 하나에 집중했다. 같은 날 **개연성 장치**를 넣었다: 장을 쓰기 전에 인과 설계를 정하고,
쓴 뒤에 연속성 편집자 역할의 모델이 확정 사실·앞선 사건과의 모순을 검토해 문제가 있으면 고쳐 쓴다.

```text
소비자 브라우저 -> consumer_app.py :8501 -> SQLite 큐 -> src/service/worker.py
                                                          |  src/generation/longform.py
                                                          |  (이야기 지도 -> 장 설계 -> 생성 -> 기계 게이트
                                                          |   -> 개연성 검토 -> 고쳐 쓰기 -> 메모리 기록)
                                                          v
                                  Ollama :11434 (Gemma4 26B-A4B Q4_K_M · num_ctx 16K · think off)
```

## 빠른 시작

1. [Ollama](https://ollama.com) 를 설치하고 모델을 받는다 (17GB). Ollama 는 로그인 시 자동으로 떠 있고,
   없으면 런처가 `ollama serve` 를 띄운다.

   ```powershell
   ollama pull hf.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M
   ```

   다른 태그를 쓰려면 `configs/default.yaml` 의 `llm.model` 또는 `.env` 의 `NOVEL_LLM_MODEL` 을 바꾼다.
   Ollama 가 다른 주소에 있으면 `NOVEL_LLM_OLLAMA_BASE_URL`.

2. 프로젝트 venv (웹·워커). Ollama 백엔드는 `requirements.txt` 외의 의존성이 없다.

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. 실행. `run_service.bat` 하나로 Ollama 확인(모델이 없으면 pull 명령을 알려 줌) → 모델 사전 적재(백그라운드)
   → 워커 → 웹이 뜬다.

   ```powershell
   .\run_service.bat          # http://127.0.0.1:8501 (LAN 은 http://<PC-IP>:8501)
   .\run_ollama.bat           # Ollama 와 모델만 확인하고 미리 올려 둘 때
   ```

   첫 적재는 1~2분(17GB 를 VRAM·RAM 에 나눠 올림), 이후 `llm.keep_alive`(30분) 동안 상주한다.

4. 상태 확인 (웹·설정된 모델 백엔드·워커 heartbeat):

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\health_check.py
   ```

### 모델 없이 둘러보기

17GB 모델은 RTX 4060 8GB + 32GB RAM 기준이다. 그만한 장비가 없거나 화면만 먼저 보고 싶으면 `.env` 에
`NOVEL_LLM_DRY_RUN=1` 을 넣고 1 단계를 건너뛴다. 모든 모델 호출이 고정 응답(`src/llm/dry_run.py`)으로
바뀌어 Ollama 없이 회원가입·기획·집필·원고 수정까지 실제 화면 흐름을 그대로 볼 수 있다. 생성되는 원고는
같은 예시 문장이므로 품질 판단에는 쓸 수 없다.

## 사용법

회원가입하면 바로 **새 작품 기획** 화면이다. 왼쪽 사이드바에 내 작품 목록이 있다.

- **기획** — 두 가지. `💬 대화로 기획` 은 **말로 한다**: "비 오는 항구 도시의 잔잔한 미스터리, 주인공은
  등대지기 딸, 3만 자" 처럼 말하면 기획 편집자가 작품 카드(제목·장르·소재·세계관·주인공·인물·분량)를
  제안하고, 대화로 다듬은 뒤 `이 설정으로 집필 시작` 을 누르면 첫 장이 큐에 들어간다. `📝 직접 작성` 은
  같은 양식을 손으로 채운다 — 여기에는 **집필 지침**(문체·시점·금기) 칸이 더 있어 매 장 프롬프트에 규칙으로
  들어간다. 양식은 `템플릿으로 저장` 해 두고 다음 작품에서 `불러오기` 할 수 있고(계정별, 같은 이름이면 덮어씀),
  대화로 만든 카드를 양식으로 가져와 다듬어도 된다.
- **할루시네이션 강도** — 집필 화면의 0~1 슬라이더. 0 은 정해진 설정에 붙어 쓰고 1 은 마음껏 지어낸다. 프롬프트 문구가
  아니라 샘플링 온도로 이어지며(`longform._creativity_temperature`, 기본 0.35 = 기본 온도), 지어낸 것이 앞 설정과
  어긋나면 개연성 장치가 잡는다. 값은 턴마다 저장돼(`jobs.creativity`) 자동 이어쓰기가 같은 강도를 물려받는다.
- **집필** — "박 노인이 20년 전 일을 털어놓게 해줘" 처럼 다음 전개를 말하며 장을 이어 쓴다. 브라우저를 닫아도
  워커가 계속 쓰고, 쓰는 장은 1초마다 화면에 흘러나온다. 완성되면 전체 원고(.md)와 이어쓰기 번들(.zip)을
  내려받는다.
- **자동 이어쓰기** — `🤖 자동 이어쓰기 켜기` 를 누르면 워커가 턴을 마칠 때마다 이야기 지도와 요약 메모리를
  보고 다음 전개를 한두 문장으로 정해(그 문장이 🤖 말풍선에 남는다) 같은 창의성·분량으로 다음 턴을 넣는다.
  완결되거나 연속 `consumer.auto_continue_max_turns`(기본 20)턴이 되면 멈춘다. `🤖 다음 한 턴만 AI 에게` 는
  한 번만 맡긴다.
- **원고 수정** — `✏️ 원고 수정` 패널(또는 각 장 아래 `✏️ 이 장 수정`)에서 장 하나를 고른다. `직접 수정`,
  `AI 보조`(빠른 요청 또는 직접 요청 → 모델이 고칠 곳을 먼저 고른 뒤 다시 쓴 제안을 보여 줌 → 저장),
  `요약 메모리`(요약·사실·미해결 단서를 손으로 고치거나 AI 로 다시 추출). 저장하면 그 장의 메모리·원장·
  진행 지표가 함께 바뀌고 이어쓰기는 고친 내용 위에서 계속된다. 집필 중인 작품은 패널이 잠긴다.
- **턴 아래의 기록** — 끝난 턴 아래에 게이트가 한 일이 남는다. `✂️` 반복 문장을 걷어냄, `🔁` 기계 게이트가
  장을 다시 씀(사유 포함), `🧭` 개연성 검토(장별 점수 1~10, 모순·원인 없는 사건, 고쳐 쓴 판을 채택했는지).
  쓰는 중에 본문이 사라지면 그 자리에 사유와 "버려진 초안 보기" / "고치기 전 초안 보기" 가 뜬다. 버려진
  초안은 저장되지 않는다.

## 개연성 장치 (`src/generation/plausibility.py`, 2026-09-03)

"개연성" 은 프롬프트에 "개연성 있게 써" 라고 적는 것으로는 지켜지지 않는다. 장마다 세 단계를 기계적으로 거친다.

1. **장 설계** — 본문을 쓰기 전에 모델에게 이 장의 인과를 JSON 으로 받는다: 목표(시점 인물이 이루려는 것),
   장애(앞선 사건에서 비롯), 전환, 결과(달라진 상태와 원인), 가져다 쓰는 앞선 사실, 되풀이 금지. 이 설계가
   작가 프롬프트의 `[이번 장의 설계 — 이 인과를 따른다]` 로 들어간다. 설계를 못 받으면 설계 없이 쓴다.
2. **연속성 검토** — 쓴 장을 연속성 편집자 역할의 모델이 검토한다. 본 것은 네 가지: (1) 작품 설정·인물·상태
   원장의 확정 사실(장소·소유·관계·목표·생사·시간)과의 **모순**, (2) 앞선 사건이나 장면 안의 원인이 없는
   **사건·행동·우연**, (3) 직전 장면 끝과의 **연속성**, (4) 설계·사용자 요청이 실제로 일어났는가. 결과는
   1~10 점수와 모순·원인 없는 사건 목록. 검토는 **본문의 모든 대사를 화자에 배정**하는 것부터 시작한다 —
   09-03 실측에서 모델은 자기 글의 개가 "하린아, 으르렁!" 이라고 말하는 대목을 일반 지시로는 두 번 놓쳤고,
   대사를 하나씩 배정하게 하자 두 장 모두에서 잡아냈다. 말할 수 없는 존재의 대사는 자동으로 모순이 된다.
3. **고쳐 쓰기** — 모순이 하나라도 있거나 점수가 `generation.plausibility_min_score`(7) 미만이면, 초안과 문제
   목록을 주고 "문제만 바로잡아 장 전체를 다시 쓰라" 고 한다(사건 순서·문장·분량 유지). 고친 판은 기계 게이트와
   재검토를 다시 거치고, 문제가 줄었거나 점수가 올랐을 때만 채택한다. 아니면 처음 판을 두고 그 결정을 기록한다.
   고쳐 쓰기는 장당 한 번이다.

검토·설계 호출이 실패(연결 오류·JSON 아님)하면 본문을 막지 않고 "검토 불가" 로 기록한다 — 검토는 문지기지
병목이 아니다. 점수는 `section_metrics.plausibility_score`, 턴 요약 `jobs.metrics_json.mean_plausibility` /
`min_plausibility` / `plausibility_repairs`, 그리고 화면의 🧭 줄로 남는다. 끄려면 `enable_plausibility_gate`,
`enable_scene_plan`.

두 종류의 다시 쓰기는 다르다. 기계 게이트(너무 짧음·미완 문장·직전 장 베끼기·구절 반복)의 재생성은 버린
초안을 프롬프트에 넣지 않는다 — 4B 시절 복사 루프의 교훈이고 `tests/test_longform_retry_isolation.py` 가
고정한다. 개연성 고쳐 쓰기는 초안을 넣고 문제만 바로잡게 한다 — 모순 한 줄을 고치는 데 장을 새로 쓰면 다른
곳이 또 어긋난다. `tests/test_plausibility_gate.py` 가 이 계약을 고정한다.

### 속도 (RTX 4060 8GB, 실측 2026-09-03, ollama 0.33.2)

| 항목 | 실측 |
|---|---|
| 모델 적재 | 16.9 GiB 중 VRAM 4.6 GiB, 나머지 RAM. 첫 적재 1~2분 |
| 디코드 | 27~30 tok/s, 한국어 1.5~1.6 자/토큰 → 1,800자 장 본문 ≈ 45초 |
| 프리필 | 533 tok/s (8.7K 토큰 16.4초) |
| 같은 접두사 재요청 | 0.4초 (KV 캐시) — 설계·검토·기록 호출은 거의 출력 시간만 |

체감: 첫 턴(이야기 지도 12 beat + 장 2개, 3,755자) 실측 **276초**. 장마다 설계 ~10초 + 생성 ~60초 + 검토 ~15초 +
기록 ~15초가 들고, 고쳐 쓰기가 나면 장당 +1~1.5분. 첫 호출이 모델 적재를 기다리지 않도록 `run_service.bat` 이 미리
올린다.

접두사 캐시가 이 속도의 핵심이다. 장 단위 호출(설계·생성·검토·기록)은 모두 같은 시스템 프롬프트와 같은 순서의
접두사(작품 설정 → 인물 → 이야기 지도 → 줄거리·상태 → 지나간 사건 → 직전 장면 끝 → 집필 지침)로 시작하고
역할 지시를 끝에 붙인다. JSON 모드도 시스템 프롬프트를 건드리지 않는다(`format=json` + 사용자 메시지 끝 힌트).

## 장편 생성기가 하는 일 (`src/generation/longform.py`)

| 약점 (실측) | 대응 |
|---|---|
| 직전 장 전문이 문맥에 있으면 베끼는 루프 (4B 실측, 26B 도 예방) | 앞 내용은 스토리 메모리·상태 원장·계층 요약으로 치환, 직전 장은 꼬리 ~2,600자만 |
| 원인 없는 사건·설정 모순 (개연성) | 장 설계 → 연속성 검토 → 고쳐 쓰기 (위 절) |
| 장 내부 반복·미완 문장·너무 짧은 장 | 기계 게이트: 반복은 걷어내기, 그 외 하드 문제만 1회 재생성 |
| 문맥 창 초과 시 앞부분(작품 설정) 소실 | `prompt_char_budget` 안에서 메모리 → 지나간 사건 → 꼬리 순으로 줄임 + 잘림 경고 |
| Gemma4 기본 thinking 으로 본문이 비는 문제 | 모든 요청에 `think=false` |

흐름: 이야기 지도(12 beat, 1회) -> 장별 과제(핵심 서사 기능 + 소비된 비트) -> 장 설계 -> 장 생성 -> 기계 게이트
-> 개연성 검토(-> 고쳐 쓰기 -> 재검토) -> 메모리 기록(JSON) -> draft/memory/ledger/outline/state 원자 저장.
목표 분량에 닿는 턴은 결말 장까지 쓰고 `completed_at` 을 찍는다.

## 모델 백엔드 (`src/llm/ollama_chat.py`)

Ollama `POST /api/chat` (NDJSON 스트리밍) 위의 얇은 클라이언트. 레거시 로컬 서버 클라이언트와 같은 표면
(`chat` / `chat_messages` / `stream_messages` / `status` / `count_tokens` / dry-run)을 유지해 생성기·워커·웹이
백엔드를 가리지 않는다.

- 모든 요청에 `think: false` (Gemma4 는 이게 없으면 답이 전부 thinking 필드로 간다), `options.num_ctx`(16384),
  `keep_alive`(30m), `format: "json"`(JSON 모드).
- 한국어 토큰 필터·DRY 샘플러·LoRA 어댑터 같은 레거시 서버 옵션은 받아서 무시한다.
- `status()` 는 `/api/tags` 로 모델 설치 여부, `/api/ps` 로 적재·VRAM 을 본다. 모델이 없으면 화면과 워커 로그에
  `ollama pull <태그>` 를 알려 준다.
- 프롬프트 토큰이 창을 채우면(`prompt_eval_count + max_tokens >= num_ctx`) 경고 로그 — Ollama 는 앞을 소리 없이 자른다.

`llm.backend: local` 로 두면 예전의 Qwen3.5-4B 모델 서버(`model_server/server.py`, 별도 랩 venv, 110K 문맥,
한국어 토큰 필터, CUDA 그래프 디코드)를 그대로 쓴다. `run_service.bat` 이 백엔드에 따라 Ollama 또는 모델 서버를
띄운다.

## 설정 (`configs/default.yaml`)

- `llm`: `backend`(`ollama` | `local`), `model`(Ollama 태그), `ollama_base_url`, `num_ctx`(16384, 고정 권장),
  `keep_alive`, 소설 샘플링(`novel_temperature` 0.75 · `top_p` 0.9 · `top_k` 40 · `repetition_penalty` 1.05),
  `memory_max_tokens` / `plan_max_tokens` / `review_max_tokens`. 환경변수 `NOVEL_LLM_BACKEND`, `NOVEL_LLM_MODEL`,
  `NOVEL_LLM_OLLAMA_BASE_URL`(또는 `OLLAMA_BASE_URL`), `NOVEL_LLM_BASE_URL`(local), `NOVEL_LLM_DRY_RUN`(테스트).
- `generation`: 장 길이(`section_min_chars` 1800), `outline_beat_count` 12, 문맥 예산(`longform_recent_context_chars`
  2600, `story_memory_context_chars` 3200, `consumed_beat_context_chars` 1200), 개연성(`enable_scene_plan`,
  `enable_plausibility_gate`, `plausibility_min_score` 7), 기계 게이트(`enable_stability_retry`).
- `consumer`: 포트·보관 기간·턴/목표 분량 선택지. `target_chars` 10,000~50,000.
- `ollama`/`data`/`training`/`chat`: 관리자 연구 UI(`app.py`)용 레거시 설정.

## 저장 위치

- `.runtime/consumer.sqlite3` — 계정·세션·작품·큐·지표 + 작품 템플릿(`story_templates`). 예전 일반 채팅 테이블
  (`chats`, `chat_messages`)은 남아 있지만 읽지 않는다.
- `data/consumer_stories/<story_id>/` — `draft.md`, `memory.jsonl`, `ledger.json`, `outline.json`, `state.json`,
  `live.txt`, `live_note.json`
- Ollama 모델: Ollama 의 저장소(`~/.ollama/models`). 레거시 모델 서버 로그: `.runtime/model_server.*.log`

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Ollama 클라이언트(NDJSON·think·JSON 모드·상태), 생성기(가짜 LLM), 개연성 장치(설계 → 검토 → 고쳐 쓰기 계약,
실패 내성, 문맥 예산), 저장소·워커·자동 이어쓰기, 프론트(AppTest, `NOVEL_LLM_DRY_RUN=1`)를 모델 없이 검증한다.
실제 모델로 한 턴을 돌려보려면 Ollama 에 모델을 받은 뒤 `python -m src.service.worker --once` 로 큐의 작업 하나를
처리한다.

## 레거시: JEPA 연구 도구

`app.py`(관리자, `run_admin.bat`, :8502)는 Ollama(`gemma4:e4b` + `embeddinggemma`) 기반 합성 데이터 -> 임베딩 ->
JEPA 예측기 학습 -> 평가 파이프라인을 그대로 품고 있다. 소비자 서비스는 이 산출물(`artifacts/active.json`)을
요구하지 않는다. 연구 배경과 평가 프로토콜은 `docs/research/jepa_regularization_longform_stability.md`,
`EVALUATION.md`.
