# AGENTS.md

## Project Goal (2026-09-03 부터)

Ollama 의 **Gemma4 26B-A4B**(MoE 25.2B, 활성 4B, Q4_K_M, 17GB) 위에서 돌아가는 한국어 장편 소설 서비스.
소비자 앱은 장편 소설 하나만 한다 (일반 채팅은 2026-09-03 에 제거). JEPA/임베딩/RAG 인덱스는 소비자
경로에서 쓰지 않는다 (레거시 연구 도구는 `app.py` 에 남아 있다).

```text
consumer_app.py (Streamlit :8501)
  └─ 장편 소설: 기획 대화(작품 카드 JSON) → ConsumerStore 큐 → src/service/worker.py
       → src/generation/longform.py
            이야기 지도 → 장 설계 → 장 생성 → 기계 게이트 → 개연성 검토·고쳐 쓰기 → 메모리 기록
       → src/llm/ollama_chat.py → Ollama (:11434, Gemma4 26B-A4B)
```

핵심 설계 근거:

- 랩 실측(4B, `C:\연구_프로젝트\ai_아키텍처\applied\README.md`): 직전 장 전문을 문맥에 두면 복사 루프. 26B 로
  바꾼 뒤에도 앞 장은 요약·상태 원장으로만 넣고 직전 장은 꼬리만 준다.
- 09-03 실측(Gemma4 26B-A4B, RTX 4060 8GB, ollama 0.33.2): `think=false` 필수(없으면 답이 thinking 필드로만 감),
  한국어 1.5자/토큰, 디코드 ~28 tok/s, 프리필 533 tok/s, 같은 접두사의 재요청은 KV 캐시로 16.4s → 0.4s. 그래서
  모든 장 단위 호출은 같은 시스템 프롬프트 + 같은 순서의 접두사로 시작하고 역할 지시를 끝에 붙인다.
- 개연성은 프롬프트 지시가 아니라 장치로 지킨다: 쓰기 전 인과 설계, 쓴 뒤 연속성 편집자 검토, 문제 시 고쳐 쓰기.

## Constraints

- Target environment: Windows, RTX 4060 8GB VRAM, 32GB RAM. Ollama 가 모델을 VRAM(~4.6 GiB)과 RAM 에 나눠 올린다.
- Keep implementation simple and robust.
- LLM 호출은 `src/service/runtime.make_llm_client` 를 통해서만 (`llm.backend: ollama` 기본, `local` 은 레거시
  Qwen3.5-4B 모델 서버).
- `llm.num_ctx` 는 요청마다 명시되며 서비스 안에서 고정 — 바꾸면 Ollama 가 17GB 모델을 다시 올린다.
- 장 단위 모델 호출(설계·생성·검토·기록)은 `longform.NOVEL_SYSTEM_PROMPT` 와 `_canon_prefix` 를 공유한다.
  접두사 앞에 무언가를 끼우거나 순서를 바꾸면 캐시가 깨져 호출마다 15초씩 늘어난다.
- 레거시 모델 서버 코드(`model_server/`)는 랩 환경(torch 2.13, transformers 5.15, bitsandbytes)에서 돈다 —
  프로젝트 venv 에 그 의존성을 넣지 않는다. CUDA 그래프 캡처/재생은 서버의 단일 GPU 스레드에서만.
- Use Korean prompts and Korean output examples.
- Save all intermediate artifacts to data/, checkpoints/, and reports/.
- Every pipeline stage must be restartable.
- Avoid hidden global state.
- Add clear error messages.

## Commands

Install. Use 3.11 explicitly: the launchers require it, and `faiss-cpu` lags new
CPython minors, so a newer interpreter fails at install time rather than with a
clear message.

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Model. Ollama (https://ollama.com) 를 설치하고 모델을 받는다 (17GB). 다른 태그를 쓰려면
`configs/default.yaml` 의 `llm.model` 또는 `.env` 의 `NOVEL_LLM_MODEL`.

```bash
ollama pull hf.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M
```

Run. Use the launchers, not `streamlit run` directly — `.streamlit/config.toml`
binds port 8501, which is the consumer port, so running `app.py` by hand puts the
admin UI where the consumer app belongs.

```bash
.\run_service.bat        # Ollama 확인·모델 적재 + 워커 + 소비자 웹
```

```bash
.\run_ollama.bat         # Ollama 와 모델만 확인하고 미리 올려 둔다
```

```bash
.\run_model_server.bat   # 레거시 Qwen 모델 서버 (llm.backend: local 일 때만)
```

```bash
.\run_admin.bat          # 레거시 연구 UI (Ollama 의 gemma4:e4b + embeddinggemma 필요)
```

Test. This is what CI runs, and there is no pytest in the venv.

```bash
.venv\Scripts\python.exe -m unittest discover -s tests
```

```bash
.venv\Scripts\python.exe scripts/smoke_jepa.py
```

레거시 Qwen 가중치는 레포에 없다. `NOVEL_QWEN_MODEL_DIR`/`NOVEL_QWEN_ADAPTERS` 로 위치를 준다
(기본: `C:\llm_files`). JEPA 산출물 설치(`scripts/share_artifact.py`)는 레거시 관리자 UI 에만 필요하다.

Recommended git safety:

```bash
git init
git add .
git commit -m "initial scaffold"
```

## Coding Rules

- Use type hints for public functions.
- Keep modules small.
- Use pydantic for schema validation.
- Use YAML config values from configs/default.yaml.
- Log every long-running step.
- Do not add complex dependencies unless necessary.
- Do not assume internet access at runtime.
- Avoid hardcoded absolute paths.

## Definition of Done

A task is done only when:
1. The GUI launches.
2. The relevant pipeline step runs on a small sample.
3. Output files are saved in the expected directory.
4. Errors are displayed clearly in Streamlit.
5. README/TASKS are updated if behavior changes.
