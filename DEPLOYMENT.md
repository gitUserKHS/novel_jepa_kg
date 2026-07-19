# Novel JEPA 소비자 서비스 배포

이 프로젝트는 소비자 앱, 단일 GPU worker, 관리자 앱을 분리해서 실행한다.
지원 범위는 신뢰할 수 있는 LAN 또는 VPN 파일럿이다. 소비자 회원가입과
로그인은 포함하며, 공용 인터넷 공개, 결제, HTTPS, 콘텐츠 모더레이션은
포함하지 않는다.

```text
소비자 브라우저 -> consumer_app.py :8501 -> SQLite FIFO queue
                                            -> 단일 GPU worker
                                            -> Ollama 127.0.0.1:11434
관리자 브라우저 -> app.py :8502 (127.0.0.1 전용)
```

Ollama의 `11434`와 관리자 `8502`는 외부에 공개하지 않는다. 소비자에게는
`8501`만 열고, 공유기 포트 포워딩은 사용하지 않는다.

## 1. 최초 설치

Python 3.11과 Ollama를 설치한 뒤 프로젝트 루트 PowerShell에서 실행한다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
ollama pull gemma4:e4b
ollama pull embeddinggemma
```

Python 3.14보다 3.11을 권장한다. PyTorch, FAISS, Streamlit 조합이 더 안정적이다.
소비자 모델은 `gemma4:e4b`, 임베딩 모델은 `embeddinggemma:latest`로 고정된다.

## 2. 관리자와 소비자 실행

관리자·학습 UI는 호스트 PC에서만 연다.

```powershell
.\run_admin.bat
# http://127.0.0.1:8502
```

`run_server.bat`도 호환성을 위해 같은 관리자 앱을 연다. 소비자 웹과 worker는
다음 명령으로 함께 실행한다.

```powershell
.\run_service.bat
# http://호스트-PC-IP:8501
```

소비자는 `8501` 주소에서 직접 회원가입하고 로그인한다. 비밀번호는 사용자별
scrypt 해시로 저장되고, 로그인 세션의 비밀값도 원문으로 저장되지 않는다.
작품은 계정에 귀속되며 `내 작품`에서 다시 열 수 있다. 공통 초대코드나 별도
작품키는 사용하지 않는다.

Windows 방화벽 요청에서는 **개인 네트워크**만 허용한다. 외부 접속은 Tailscale
같은 VPN을 사용하고 공용 인터넷에 Streamlit을 직접 노출하지 않는다.

## 3. JEPA 서비스 모델 준비

현재 연구 체크포인트는 샘플 10개, 검증 1개라 소비자 모델로 승격되지 않는다.
기존 파일은 삭제하거나 덮어쓰지 않는다.

1. 관리자 `Dataset`에서 최소 40개, 권장 96개 이상의 유효 샘플을 준비한다.
2. 관리자 `Service` 탭에서 `Enter maintenance`를 누른다.
3. 상태가 `draining`이면 기존 작업이 끝나 `active`가 될 때까지 기다린다.
4. `Candidate versions`에서 후보를 학습하고 품질 게이트를 확인한다.
5. 모든 기준을 통과한 후보만 승격한다.

승격 기준은 샘플 40, 검증 6, validation cosine 0.60, hit@5 0.50,
정규화 top-1 다양성 0.40, JEPA가 RAG-next보다 0.03 이상 높은 검색 점수,
dataset fingerprint·벡터 차원·모델명 일치다. 통과한 버전만 SHA-256과 함께
`artifacts/active.json`에 원자 기록된다. 실패하면 기존 active 버전이 유지된다.

## 4. 작품 저장과 복구

- 메타데이터·큐·지표: `.runtime/consumer.sqlite3` (SQLite WAL)
- 작품 원고: `data/consumer_stories/<story_id>/draft.md`
- 장기기억: 같은 작품 디렉터리의 `memory.jsonl`, `ledger.json`, `state.json`
- 기본 보관: 30일, worker가 만료 작품을 정리
- 즉시 삭제: 소비자 작품 관리 메뉴에서 DB 행과 작품 폴더를 함께 삭제

한 작품에는 대기 또는 실행 작업이 하나만 존재할 수 있고, 전체 worker도 한
작업만 실행한다. 브라우저를 닫아도 생성은 계속된다. worker가 중단되면 stale
작업은 `failed_recoverable`이 되고 이미 저장된 섹션은 그대로 유지된다.

## 5. Windows 예약 작업

현재 사용자 로그인 시 consumer 웹과 worker를 독립적으로 자동 시작한다.

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

consumer 웹, worker heartbeat, Ollama의 두 모델, active JEPA 파일 fingerprint를
한 번에 검사한다.

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
| `NOVEL_JEPA_OLLAMA_BASE_URL` | `http://localhost:11434` | 로컬 Ollama 주소 |
| `NOVEL_JEPA_CONSUMER_BIND_HOST` | `0.0.0.0` | 소비자 바인드 주소 |
| `NOVEL_JEPA_CONSUMER_PORT` | `8501` | 소비자 포트 |
| `NOVEL_JEPA_CONSUMER_DB` | `.runtime/consumer.sqlite3` | 서비스 DB |
| `NOVEL_JEPA_CONSUMER_STORY_ROOT` | `data/consumer_stories` | 격리 작품 루트 |
| `NOVEL_JEPA_BIND_HOST` | `127.0.0.1` | 관리자 바인드 주소 |
| `NOVEL_JEPA_PORT` | `8502` | 관리자 포트 |

소비자 모델명은 환경 변수로 바꾸지 않는다. `.env`는 Git에서 제외된다.

## 8. CI/CD

`CI`는 Windows/Python 3.11에서 단위 테스트와 dry-run pipeline smoke test를
수행한다. 테스트에는 계정·세션·소유권 격리, FIFO, worker 복구, maintenance, 품질 게이트,
fake 생성기 3만 자 누적, Streamlit 관리자/소비자 권한 분리가 포함된다.

`Deploy to local Windows host`는 `master`의 CI가 성공한 뒤에만 self-hosted
Windows runner에서 실행된다. 배포 순서는 maintenance drain, queue idle,
서비스 중지, 설치·테스트, consumer/worker 재시작, 웹·worker·Ollama·active
JEPA health check, 서비스 재개다. 검증 실패 시 maintenance를 유지한다.

runner에는 `novel-jepa` 라벨을 붙이고 Ollama를 쓰는 같은 Windows 계정으로
실행한다. 공개 저장소의 신뢰하지 않는 pull request에는 self-hosted runner를
절대 제공하지 않는다.
