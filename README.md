# Novel Studio (novel_jepa_lab)

로컬 **Qwen3.5-4B(4bit)** 로 돌아가는 한국어 장편 소설 서비스 + 일반 채팅.
RTX 4060 8GB 한 장에서 모델 서버·생성 워커·웹을 모두 띄운다.

2026-08-23 에 백엔드를 Ollama(gemma4:e4b) + JEPA 플래너에서 **자체 모델 서버 + 구조적
장편 생성기**로 갈아엎었다. JEPA/임베딩/RAG 인덱스는 소비자 경로에서 제거했고(품질에
기여하지 못하고 의존성만 늘림), 이야기 지도·스토리 메모리·소비된 비트 원장·안정성
게이트처럼 실제로 긴 글의 일관성을 지키는 장치만 남겨 새 모델에 맞게 다시 썼다.

```text
소비자 브라우저 -> consumer_app.py :8501 -+- 일반 채팅: 모델 서버로 직접 스트리밍
                                         +- 장편 소설: SQLite 큐 -> src/service/worker.py
                                                                       |
모델 서버  model_server/server.py :8765 <-------------------------------+
(Qwen3.5-4B 4bit / GQA-split 어텐션 / 120K 문맥 / 한국어 토큰 필터 / LoRA 어댑터)
```

## 빠른 시작

1. 모델 서버 환경. 프로젝트 venv 와 **별도**다 (`requirements-model-server.txt`).
   기본값은 랩 환경 `C:\연구_프로젝트\ai_아키텍처\.venv` 를 쓰며, 다른 환경은
   `NOVEL_QWEN_PYTHON` 으로 지정한다. 모델 가중치는 `NOVEL_QWEN_MODEL_DIR`
   (기본: `C:\llm_files\qwen35-4b`), 말투 어댑터는 `NOVEL_QWEN_ADAPTERS`
   (`이름=경로;...`, 기본: `cute=C:\llm_files\adapters\cute`).

   **처음 받은 팀원은** 모델이 없으므로 먼저 내려받는다 (베이스 12GB + 어댑터 81MB,
   어댑터는 팀 비공개 저장소라 초대가 필요하다). 자세한 절차는 `DEPLOYMENT.md` §1.1.

   ```powershell
   hf auth login
   .\.venv\Scripts\python.exe .\scripts\setup_model.py --namespace <팀-네임스페이스>
   ```
2. 프로젝트 venv (웹·워커):

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. 실행. `run_service.bat` 하나로 모델 서버(없으면 기동) + 워커 + 웹이 뜬다.

   ```powershell
   .\run_service.bat          # http://127.0.0.1:8501 (LAN 은 http://<PC-IP>:8501)
   .\run_model_server.bat     # 모델 서버만 따로 띄우거나 상태를 볼 때
   ```

   첫 기동은 모델 로드에 ~10초, 이후 서버는 계속 떠 있다(VRAM ~3 GiB).

4. 상태 확인:

   ```powershell
   .\.venv\Scripts\python.exe .\scripts\health_check.py
   ```

## 사용법

회원가입 후 왼쪽 사이드바에서 모드를 고른다.

- **일반 채팅** — ChatGPT 처럼 쓴다. 대화는 계정별로 저장되고, `말투` 에서
  기본 / 밝고 귀여운 말투(cute LoRA x0.6)를 고를 수 있다. 사이드바 목록은
  오늘·어제·지난 7일 식으로 묶이고, 5개부터는 제목·본문 검색창이 생긴다.
  제목 옆 `⋯` 에서 이름 바꾸기·마크다운 내려받기·메시지 비우기, `🗑` 에서
  대화 삭제(2단계 확인). 최근 질문 5개 옆 `✏️` 는 고쳐서 다시 보내기(그 질문부터
  뒤는 버림), 모든 질문의 `✕` 는 질문·답변 한 쌍 삭제, 마지막 답변 아래 `🔄 다시
  생성`. `새 대화` 는 빈 대화가 이미 있으면 그걸 다시 연다.
- **장편 소설** — 기획은 두 가지. `💬 대화로 기획` 은 **말로 한다**:
  "비 오는 항구 도시의 잔잔한 미스터리, 주인공은 등대지기 딸, 3만 자" 처럼
  말하면 기획 편집자가 작품 카드(제목·장르·소재·세계관·주인공·인물·분량)를
  제안하고, 대화로 다듬은 뒤 `이 설정으로 집필 시작` 을 누르면 첫 장이
  큐에 들어간다. `📝 직접 작성` 은 같은 양식을 손으로 채운다 — 여기에는
  **집필 지침**(문체·시점·금기) 칸이 더 있어 매 장 프롬프트의 작품 설정에
  붙는다. 양식은 `템플릿으로 저장` 해 두고 다음 작품에서 `불러오기` 할 수 있고
  (계정별, 같은 이름이면 덮어씀), 대화로 만든 카드를 양식으로 가져와 다듬어도
  된다. 작품 화면의 `작품 설정 › 템플릿으로 저장` 도 같은 템플릿을 만든다.
  이후에는 "박 노인이 20년 전 일을 털어놓게 해줘" 처럼 다음 전개를 말하며
  장을 이어 쓴다. 브라우저를 닫아도 워커가 계속 쓰고, 완성되면
  전체 원고(.md)와 이어쓰기 번들(.zip)을 내려받는다.
- **원고 수정** — 작품 화면의 `✏️ 원고 수정` 패널(또는 각 장 아래 `✏️ 이 장 수정`)
  에서 장 하나를 고른다. `직접 수정` 은 본문을 그대로 편집해 저장, `AI 보조` 는
  빠른 요청(더 긴장감 있게 · 짧고 담백하게 · …)이나 직접 적은 요청으로 모델이
  고쳐 쓴 제안을 먼저 보여 주고 마음에 들면 저장. 문체 전체를 바꾸는 요청은
  모델이 먼저 "고칠 곳 6~10개" 를 고른 뒤 그 계획대로 다시 쓴다(4B 모델은 이
  단계가 없으면 원문을 그대로 베낀다, 2026-08-23 실측). 제안 분량이 원문의
  70% 미만·140% 초과면 이유를 붙여 자동으로 한 번 더 쓰고, 제안 아래에 분량
  비율을 보여 준다 — 같은 요청도 장에 따라 편차가 크니 저장 전에 훑어보자.
  저장하면 그 장의 **요약
  메모리**(summary·사실·단서)를 모델이 다시 뽑아 `memory.jsonl`·`ledger.json`·
  `state.json`·진행 지표를 함께 맞춘다 — 생성기는 앞 내용을 원문이 아니라 이
  메모리로 보므로, 이 갱신이 없으면 다음 장이 고치기 전 이야기를 이어 쓴다.
  `요약 메모리` 탭에서는 요약·사실·미해결 단서를 손으로 고치거나 AI 로 다시
  추출할 수 있다. 집필 중(대기·실행)인 작품은 패널이 잠긴다.

### 속도 (RTX 4060, 실측 2026-08-23)

디코드는 **CUDA 그래프**(`model_server/qwen_stack/graph_decode.py`)로 돈다. eager 디코드는
토큰당 CPU 111 ms > GPU 92 ms, 커널 런치 ~1만 개(선형어텐션 24층의 순수 PyTorch 재귀)로
CPU-바운드였는데, 1토큰 forward 를 정적 버퍼 캐시로 캡처해 재생하면 **10 → 30~48 tok/s**
(그리디 토큰은 eager 와 완전 일치). 문맥+생성이 버킷 상한(기본 32K, `NOVEL_QWEN_MAX_BUCKET`,
KV 버퍼 1 GiB 상주)을 넘으면 eager 로 자동 폴백한다. 실측: 20K 문맥 37 tok/s(버킷 32K),
65536 버킷은 52K 문맥 바늘 회수 정답·peak 5.4 GiB.
그래프는 (버킷, 어댑터, 배율)별로 첫 사용 때 캡처되고, 캡처/재생은 단일 GPU 스레드에서만
수행한다(다른 스레드에서 재캡처하면 CUDA 가 프로세스째 죽는 것을 실측).

체감: 3,000자 턴이 **약 2~2.5분**(첫 턴은 이야기 지도 설계 +30~60초), 채팅 응답은 수 초.
생성 중인 장은 1초마다 화면에 흘러나온다.

## 장편 생성기가 하는 일 (`src/generation/longform.py`)

랩에서 같은 모델로 실측한 약점에 맞춰 설계했다(`C:\연구_프로젝트\ai_아키텍처\applied\README.md`):

| 실측된 약점 | 대응 |
|---|---|
| 직전 장 전문이 문맥에 있으면 그대로 베끼는 루프 | 앞 내용은 스토리 메모리·상태 원장·계층 요약으로 치환, 직전 장은 꼬리 ~2,200자만 |
| 반복 페널티를 올리면 한자·영어로 샘 | 페널티 1.05 + 서버 측 한국어 토큰 필터(본문은 strict: 영어 단어 토큰까지 차단) |
| 장 내부 1,000자 근처부터 단락 반복 | **DRY 샘플러**(생성 토큰의 구절 연속만 벌점, 기본 0.8/1.75/2 — bench: rep12 3.7→1.3, distinct3 0.77→0.87) + 생성 후 게이트 -> 하드 문제만 1회 재생성 |
| 말투 LoRA 가 서술을 짧게 만듦 | 소설은 항상 기본 모델, 어댑터는 채팅 전용 |

흐름: 이야기 지도(8 beat, 1회) -> 장별 과제(핵심 서사 기능 + 소비된 비트) -> 장 생성
-> 게이트 -> 메모리 기록(별도 짧은 JSON 호출) -> draft/memory/ledger/outline/state 원자 저장.
목표 분량에 닿는 턴은 결말 장까지 쓰고 `completed_at` 을 찍는다.

## 모델 서버 API (`model_server/server.py`)

OpenAI 호환. `POST /v1/chat/completions` (`stream: true` 면 SSE), `GET /health`,
`GET /v1/models`, `POST /v1/count_tokens`. 확장 필드: `adapter`(예: `"cute"`),
`adapter_scale`(기본 0.6), `korean_filter`(`true` | `false` | `"strict"`),
`repetition_penalty`, `no_repeat_ngram_size`, `seed`, `stop`, `min_p`,
`dry_multiplier`/`dry_base`/`dry_allowed_length`(구절 반복 억제),
`engine`(`auto` | `graph` | `eager`).
단일 GPU 이므로 생성은 직렬화되며, 스트리밍 중 연결이 끊기면 생성을 멈춘다.
문맥 상한은 110K 토큰(실측 한계 120K 에 여유).

## 설정 (`configs/default.yaml`)

- `llm`: 백엔드(`local` | `ollama` 레거시), 서버 주소, 채팅/소설 샘플링, 어댑터 배율.
  환경변수 `NOVEL_LLM_BASE_URL`, `NOVEL_LLM_CHAT_ADAPTER`, `NOVEL_LLM_DRY_RUN`(테스트).
- `consumer`: 포트·보관 기간·턴/목표 분량 선택지. `target_chars` 10,000~50,000.
- `generation`: 장 길이(`section_min_chars` 1800), 이야기 지도 beat 수, 메모리 예산 등.
- `ollama`/`data`/`training`: 관리자 연구 UI(`app.py`)용 레거시 설정.

## 저장 위치

- `.runtime/consumer.sqlite3` — 계정·세션·작품·큐·지표 + 일반 채팅(`chats`, `chat_messages`)
  + 작품 템플릿(`story_templates`)
- `data/consumer_stories/<story_id>/` — `draft.md`, `memory.jsonl`, `ledger.json`, `outline.json`, `state.json`, `live.txt`
- `.runtime/model_server.*.log`, `.runtime/model_server.pid` — 모델 서버

## 테스트

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

생성기·클라이언트·채팅 저장소·워커·프론트(AppTest, `NOVEL_LLM_DRY_RUN=1`)를
모델 서버 없이 검증한다. 실제 모델로 한 턴을 돌려보려면 서버를 띄운 뒤
`python -m src.service.worker --once` 로 큐의 작업 하나를 처리한다.

## 레거시: JEPA 연구 도구

`app.py`(관리자, `run_admin.bat`, :8502)는 Ollama 기반 합성 데이터 -> 임베딩 ->
JEPA 예측기 학습 -> 평가 파이프라인을 그대로 품고 있다. 소비자 서비스는 더 이상
이 산출물(`artifacts/active.json`)을 요구하지 않는다. 연구 배경과 평가 프로토콜은
`docs/research/jepa_regularization_longform_stability.md`, `EVALUATION.md`.
`llm.backend: ollama` 로 두면 워커가 예전 생성기(`generate_with_controlled_hallucination`)로
돌아간다.
