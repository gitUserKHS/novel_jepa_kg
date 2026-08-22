# AGENTS.md

## Project Goal (2026-08-23 부터)

로컬 Qwen3.5-4B(4bit) 모델 서버 위에서 돌아가는 한국어 장편 소설 서비스 + 일반 채팅.
소비자 경로는 JEPA/임베딩/RAG 인덱스를 쓰지 않는다 (레거시 연구 도구는 `app.py` 에 남아 있다).

```text
consumer_app.py (Streamlit :8501)
  ├─ 일반 채팅: src/llm/local_client.py → model_server/server.py (:8765) 로 SSE 스트리밍
  └─ 장편 소설: 기획 대화(작품 카드 JSON) → ConsumerStore 큐 → src/service/worker.py
                → src/generation/longform.py (이야기 지도·스토리 메모리·게이트) → 모델 서버
```

핵심 설계 근거는 랩 실측(`C:\연구_프로젝트\ai_아키텍처\applied\README.md`): 직전 장 전문을 문맥에
두면 복사 루프, 반복 페널티는 언어 이탈, 말투 LoRA 는 서술 단축. 생성기는 이 셋을 피하도록 짜였다.

## Constraints

- Target environment: Windows, RTX 4060 8GB VRAM, 32GB RAM.
- Keep implementation simple and robust.
- LLM 호출은 `src/service/runtime.make_llm_client` 를 통해서만 (로컬 서버 기본, Ollama 는 레거시).
- 모델 서버 코드(`model_server/`)는 랩 환경(torch 2.13, transformers 5.15, bitsandbytes)에서 돈다 —
  프로젝트 venv 에 그 의존성을 넣지 않는다.
- 디코드는 CUDA 그래프(`graph_decode.py`)가 기본. 그래프 캡처/재생은 반드시 서버의 단일 GPU
  스레드에서만 — 다른 스레드에서 캡처하면 프로세스가 죽는다 (08-23 실측).
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

Run. Use the launchers, not `streamlit run` directly — `.streamlit/config.toml`
binds port 8501, which is the consumer port, so running `app.py` by hand puts the
admin UI where the consumer app belongs.

```bash
.\run_service.bat        # 모델 서버(자동 기동) + 워커 + 소비자 웹
```

```bash
.\run_model_server.bat   # 모델 서버만
```

```bash
.\run_admin.bat          # 레거시 연구 UI (Ollama 필요)
```

Test. This is what CI runs, and there is no pytest in the venv.

```bash
.venv\Scripts\python.exe -m unittest discover -s tests
```

```bash
.venv\Scripts\python.exe scripts/smoke_jepa.py
```

모델 가중치는 레포에 없다. `NOVEL_QWEN_MODEL_DIR`/`NOVEL_QWEN_ADAPTERS` 로 위치를 준다
(기본: 랩 디렉터리). JEPA 산출물 설치(`scripts/share_artifact.py`)는 레거시 관리자 UI 에만 필요하다.

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
