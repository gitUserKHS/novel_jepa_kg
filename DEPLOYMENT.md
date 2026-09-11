# Novel Studio 소비자 서비스 배포

이 프로젝트는 소비자 앱, 단일 GPU worker, 관리자 앱을 분리해서 실행한다.
지원 범위는 신뢰할 수 있는 LAN 또는 VPN 파일럿이다. 소비자 회원가입과
로그인은 포함하며, 공용 인터넷 공개, 결제, HTTPS, 콘텐츠 모더레이션은
포함하지 않는다.

```text
소비자 브라우저 -> consumer_app.py :8501 -> SQLite FIFO queue -> 단일 GPU worker -> Ollama 127.0.0.1:11434
                                                                                     (Gemma4 26B-A4B Q4_K_M)
관리자 브라우저 -> app.py :8502 (127.0.0.1 전용, 레거시 JEPA 연구 UI, Ollama 의 gemma4:e4b 필요)
```

Ollama `11434`, 레거시 모델 서버 `8765`, 관리자 `8502`는 외부에 공개하지 않는다. 소비자에게는
`8501`만 열고, 공유기 포트 포워딩은 사용하지 않는다.

## 1. 최초 설치

1. [Ollama](https://ollama.com) 를 설치한다. Windows 설치판은 로그인 시 자동으로 서버를 띄운다.
   런처(`run_service.bat`)는 서버가 없으면 `ollama serve` 를 숨겨서 띄우고, 모델이 없으면 pull 명령을 알려 준다.

2. 소설 모델을 받는다 (17GB, Q4_K_M GGUF). 다른 태그를 쓰려면 `configs/default.yaml` 의 `llm.model`
   또는 `.env` 의 `NOVEL_LLM_MODEL`.

   ```powershell
   ollama pull hf.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M
   ```

3. 프로젝트 venv (웹·워커), Python 3.11 권장. Ollama 백엔드는 이 파일 외의 의존성이 없다.

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

RTX 4060 8GB 에서 Ollama 는 모델의 약 4.6 GiB 를 VRAM 에, 나머지를 RAM(32GB 권장)에 올린다.
`llm.num_ctx`(16384) 를 바꾸면 모델을 다시 올리므로 운영 중에는 고정한다.

## 1.1 레거시: 자체 Qwen3.5-4B 모델 서버

`llm.backend: local` 로 두면 `model_server/server.py` 를 쓴다. 별도 Python 환경
(`requirements-model-server.txt`: torch 2.13+cu126, transformers 5.15, bitsandbytes, peft)이 필요하고,
기본값은 `C:\연구_프로젝트\ai_아키텍처\.venv`, `NOVEL_QWEN_PYTHON` 으로 바꾼다. 가중치·어댑터 위치는
`NOVEL_QWEN_MODEL_DIR`, `NOVEL_QWEN_ADAPTERS` (기본: `C:\llm_files\qwen35-4b`, `C:\llm_files\adapters\cute`).
팀원이 가중치를 받는 절차는 `scripts/setup_model.py --namespace <팀-네임스페이스>` (베이스 12GB, 어댑터는
팀 비공개 저장소), 올리는 쪽은 `scripts/share_model.py`.

## 2. 관리자와 소비자 실행

관리자·학습 UI는 호스트 PC에서만 연다.

```powershell
.\run_admin.bat
# http://127.0.0.1:8502
```

`run_server.bat`도 호환성을 위해 같은 관리자 앱을 연다. 소비자 웹과 worker, 모델 백엔드는
다음 명령으로 함께 실행한다.

```powershell
.\run_service.bat
# http://호스트-PC-IP:8501
```

`run_service.bat` 은 `configs/default.yaml` 의 `llm.backend` 를 읽어 Ollama(기본) 또는 레거시 모델
서버를 준비한다. Ollama 쪽은 `scripts/run_ollama.ps1`: 서버 응답 확인(없으면 기동) → 설정된 모델 설치
확인 → 백그라운드 사전 적재. 모델만 확인·적재하려면 `.\run_ollama.bat`.

소비자는 `8501` 주소에서 직접 회원가입하고 로그인한다. 비밀번호는 사용자별
scrypt 해시로 저장되고, 로그인 세션의 비밀값도 원문으로 저장되지 않는다.
작품은 계정에 귀속되며 사이드바에서 다시 열 수 있다. 공통 초대코드나 별도
작품키는 사용하지 않는다.

Windows 방화벽 요청에서는 **개인 네트워크**만 허용한다. 외부 접속은 Tailscale
같은 VPN을 사용하고 공용 인터넷에 Streamlit을 직접 노출하지 않는다.

## 3. 레거시: JEPA 서비스 모델

관리자 연구 UI(`app.py`)의 JEPA predictor 는 소비자 서비스가 더 이상 요구하지 않는다. 직접 학습해
승격하거나(관리자 `Service` 탭, maintenance 필요) 팀원과 공유(`scripts/share_artifact.py export/import`,
GitHub Release 자산, SHA-256 검증)하는 절차는 그대로 남아 있다. 연구 UI 가 쓰는 Ollama 모델:

```bash
ollama pull gemma4:e4b
ollama pull embeddinggemma
```

## 4. 작품 저장과 복구

- 메타데이터·큐·지표: `.runtime/consumer.sqlite3` (SQLite WAL)
- 작품 원고: `data/consumer_stories/<story_id>/draft.md`
- 장기기억: 같은 작품 디렉터리의 `memory.jsonl`, `ledger.json`, `outline.json`, `state.json`
- 기본 보관: 30일, worker가 만료 작품을 정리
- 즉시 삭제: 소비자 작품 관리 메뉴에서 DB 행과 작품 폴더를 함께 삭제

한 작품에는 대기 또는 실행 작업이 하나만 존재할 수 있고, 전체 worker도 한
작업만 실행한다. 브라우저를 닫아도 생성은 계속된다. worker가 중단되면 stale
작업은 `failed_recoverable`이 되고 이미 저장된 섹션은 그대로 유지된다.

## 5. Windows 예약 작업

현재 사용자 로그인 시 consumer 웹과 worker를 독립적으로 자동 시작한다. Ollama 는 자체 자동 시작을
쓰고, 워커는 모델이 준비될 때까지 `model_unavailable` 상태로 대기한다.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_local_service.ps1 -StartNow
```

등록되는 작업은 다음 두 개다.

- `Novel JEPA Consumer Web`
- `Novel JEPA Consumer Worker`

관리자 앱은 자동 시작하지 않는다. 예약 작업 제거:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\uninstall_local_service.ps1
```

## 6. 상태 확인

consumer 웹, 설정된 모델 백엔드(Ollama 의 `llm.model` 설치 여부 또는 레거시 모델 서버), worker
heartbeat 를 한 번에 검사한다. 연구 UI 의 Ollama 모델·JEPA 산출물 검사는 `--check-ollama`,
`--check-active-jepa` 로만.

```powershell
.\.venv\Scripts\python.exe .\scripts\health_check.py
```

maintenance 제어와 queue 상태 확인:

```powershell
.\.venv\Scripts\python.exe .\scripts\service_control.py status
.\.venv\Scripts\python.exe .\scripts\service_control.py enter
.\.venv\Scripts\python.exe .\scripts\service_control.py wait-idle
.\.venv\Scripts\python.exe .\scripts\service_control.py resume
```

## 7. 환경 변수

| 변수 | 기본값 | 용도 |
|---|---|---|
| `NOVEL_LLM_BACKEND` | `ollama` | 소설 백엔드 (`ollama` | `local`) |
| `NOVEL_LLM_MODEL` | `hf.co/HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced:Q4_K_M` | Ollama 모델 태그 |
| `NOVEL_LLM_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | 소설 백엔드 Ollama 주소 (`OLLAMA_BASE_URL` 도 인정) |
| `NOVEL_LLM_DRY_RUN` | `false` | 모델 없이 고정 응답 (테스트) |
| `NOVEL_LLM_BASE_URL` | `http://127.0.0.1:8765` | 레거시 로컬 모델 서버 주소 |
| `NOVEL_QWEN_PYTHON` | 랩 venv | 레거시 모델 서버를 돌릴 python.exe |
| `NOVEL_QWEN_MODEL_DIR` | `C:\llm_files\qwen35-4b` | 레거시 Qwen3.5-4B 가중치 |
| `NOVEL_QWEN_ADAPTERS` | `cute=C:\llm_files\adapters\cute` | `이름=경로;...` |
| `NOVEL_JEPA_OLLAMA_BASE_URL` | `http://localhost:11434` | 레거시 관리자 UI 용 Ollama 주소 |
| `NOVEL_JEPA_CONSUMER_BIND_HOST` | `0.0.0.0` | 소비자 바인드 주소 |
| `NOVEL_JEPA_CONSUMER_PORT` | `8501` | 소비자 포트 |
| `NOVEL_JEPA_CONSUMER_DB` | `.runtime/consumer.sqlite3` | 서비스 DB |
| `NOVEL_JEPA_CONSUMER_STORY_ROOT` | `data/consumer_stories` | 격리 작품 루트 |
| `NOVEL_JEPA_BIND_HOST` | `127.0.0.1` | 관리자 바인드 주소 |
| `NOVEL_JEPA_PORT` | `8502` | 관리자 포트 |

`.env`는 Git에서 제외된다.

## 8. CI/CD

`CI`는 Windows/Python 3.11에서 단위 테스트와 dry-run pipeline smoke test를
수행한다. 테스트에는 계정·세션·소유권 격리, FIFO, worker 복구, maintenance, 품질 게이트,
개연성 장치(설계 → 검토 → 고쳐 쓰기 계약), Ollama 클라이언트, fake 생성기 3만 자 누적,
Streamlit 관리자/소비자 권한 분리가 포함된다.

`Deploy to local Windows host`는 **수동 실행(`workflow_dispatch`) 전용**이다. 배포 순서는
maintenance drain, queue idle, 서비스 중지, 설치·테스트, consumer/worker 재시작,
웹·백엔드·worker health check, 서비스 재개다. 검증 실패 시 maintenance를 유지한다.

전에는 CI 성공 뒤 자동으로 걸렸지만, `novel-jepa` 라벨의 self-hosted runner가 등록돼 있지
않아 매 푸시마다 시작되지 못하는 실행이 Actions 탭에 영구히 쌓였다(포크한 사람에게도 그대로
간다). runner 를 등록한 뒤 `.github/workflows/deploy-local.yml` 주석의 `workflow_run`
트리거를 되살리면 자동 배포로 돌아간다.

runner에는 `novel-jepa` 라벨을 붙이고 Ollama를 쓰는 같은 Windows 계정으로
실행한다. 공개 저장소의 신뢰하지 않는 pull request에는 self-hosted runner를
절대 제공하지 않는다.
