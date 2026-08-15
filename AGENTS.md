# AGENTS.md

## Project Goal

This project is a Streamlit web GUI for a JEPA-inspired latent planner combined with a local LLM for Korean novel generation.

The system should:
1. Generate synthetic narrative transition data using a local LLM.
2. Validate and filter generated JSONL data.
3. Embed scene summaries.
4. Train a small predictor model in embedding space.
5. Use the predictor as a latent next-scene planner.
6. Generate novel prose using a local LLM.
7. Compare LLM-only, RAG, and JEPA-planner generation modes.

## Core Idea

Do not fine-tune the local LLM. Train only a small PyTorch predictor that maps current scene embeddings to next scene embeddings.

Architecture:

```text
사용자가 웹 GUI에서 버튼 클릭
→ Gemma/Ollama가 합성 서사 데이터 생성
→ JSONL 검증 및 필터링
→ 장면 요약 임베딩 생성
→ JEPA-inspired latent predictor 학습
→ 예측 벡터로 유사 장면 검색
→ Local LLM이 다음 장면/소설 본문 생성
→ LLM-only / RAG / JEPA-planner 비교 평가
```

## Constraints

- Target environment: Windows, RTX 4060 8GB VRAM, 32GB RAM.
- Keep implementation simple and robust.
- Prefer CPU FAISS first.
- Use local Ollama APIs for LLM chat and embeddings.
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
.\run_admin.bat
```

```bash
.\run_service.bat
```

Test. This is what CI runs, and there is no pytest in the venv.

```bash
.venv\Scripts\python.exe -m unittest discover -s tests
```

```bash
.venv\Scripts\python.exe scripts/smoke_jepa.py
```

Get the trained model. A clone has none: `data/`, `checkpoints/`, and
`artifacts/versions/` are all gitignored, and there is no RAG-only fallback, so
consumer generation stays disabled until a version is installed.

```bash
python scripts/share_artifact.py import <bundle>.zip --activate
```

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
