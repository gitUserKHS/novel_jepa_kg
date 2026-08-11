from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

from lxml import etree
from PIL import Image, ImageDraw, ImageFont


TEMPLATE = Path("26-1 CG_capstone design_final report_template.hwpx")
OUT = Path("캡스톤디자인_최종보고서_Novel_JEPA_Lab_작성본.hwpx")
MD_OUT = Path("캡스톤디자인_최종보고서_Novel_JEPA_Lab_작성본.md")
VISUAL_DIR = Path("capstone_visuals_14w")

HP = "http://www.hancom.co.kr/hwpml/2011/paragraph"
NS = {"hp": HP}
RUN = f"{{{HP}}}run"
TEXT = f"{{{HP}}}t"


REPORT_ITEMS = [
    ("title", "26-1 캡스톤디자인 결과보고서"),
    ("meta", "팀명: [기입 필요]"),
    ("meta", "팀원: [리더 및 조원 기입 필요]"),
    ("blank", ""),
    ("h1", "0. 요약"),
    ("body", "본 프로젝트는 한국어 장편소설 생성을 위한 Streamlit 기반 실험 GUI이다."),
    ("body", "핵심 목표는 로컬 LLM을 미세조정하지 않고, 작은 JEPA-inspired latent predictor만 학습하여 다음 장면 방향을 계획하는 것이다."),
    ("body", "전체 파이프라인은 합성 서사 데이터 생성, JSONL 검증, 임베딩, FAISS 검색, PyTorch 예측기 학습, 장문 생성, 평가 보고서 작성으로 구성된다."),
    ("body", "최종 결과물은 Creative Hallucination + JEPA 단일 생성 모드와 Story-memory RAG를 결합한 장편 소설 생성 도구이다."),
    ("body", "생성 과정은 5,000자, 10,000자, 사용자 지정 턴 단위로 분리되어 RTX 4060 8GB 환경에서도 중단 복구가 가능하도록 설계되었다."),
    ("body", "결과적으로 데이터 생성부터 평가까지 재시작 가능한 로컬 AI 창작 실험 환경을 구축하였다."),
    ("blank", ""),
    ("h1", "1. 개요"),
    ("h2", "(1) 프로젝트 명"),
    ("body", "Novel JEPA Lab: JEPA-inspired latent planner와 로컬 LLM을 결합한 한국어 장편소설 생성 시스템"),
    ("h2", "(2) 배경 및 필요성"),
    ("body", "최근 로컬 LLM은 짧은 장면 작성에는 유용하지만, 장편 서사에서는 복선 반복, 설정 충돌, 장기 기억 손실 문제가 쉽게 발생한다."),
    ("body", "LLM 전체를 fine-tuning하는 방식은 데이터와 GPU 비용이 크므로 캡스톤 프로젝트 환경에는 부담이 있다."),
    ("body", "본 프로젝트는 작은 예측기만 학습하여 현재 장면 임베딩에서 다음 장면 임베딩을 예측하는 경량 대안을 검토하였다."),
    ("body", "또한 교수 피드백에서 요구된 hallucination을 무작위 오류가 아니라 통제된 창작 확장으로 정의하고 측정 대상으로 삼았다."),
    ("h2", "(3) 목표"),
    ("body", "1) 로컬 Ollama 모델만으로 합성 서사 전환 데이터를 생성한다."),
    ("body", "2) pydantic 검증과 다양성 축을 이용하여 학습 가능한 JSONL 데이터를 정제한다."),
    ("body", "3) 현재 장면 표현에서 다음 장면 표현을 예측하는 PyTorch MLP predictor를 학습한다."),
    ("body", "4) FAISS 검색과 JEPA 예측 방향을 결합하여 다음 장면 계획을 제공한다."),
    ("body", "5) Story-memory RAG, 지식 그래프, 압축 요약으로 장편 생성의 개연성과 일관성을 보강한다."),
    ("body", "6) controlled hallucination, 반복률, 구조 적합도, 이름 일관성, 진행성 등을 평가 보고서로 산출한다."),
    ("blank", ""),
    ("h1", "2. 요구사항 정의"),
    ("h2", "(1) 기능 요구사항"),
    ("body", "1) 합성 데이터 생성: 장르와 장면 프리셋을 입력받아 scene_t와 scene_t_plus_1 구조의 JSONL 데이터를 생성한다."),
    ("body", "2) 데이터 검증: pydantic schema로 누락 필드와 품질 조건을 검사하고 유효 샘플만 필터링한다."),
    ("body", "3) 임베딩 생성: 현재 장면 구조 텍스트와 다음 장면 목표 텍스트를 embeddinggemma 모델로 벡터화한다."),
    ("body", "4) FAISS 인덱싱: current_context.faiss와 next_scene.faiss를 구축하여 검색 기반 비교가 가능하게 한다."),
    ("body", "5) predictor 학습: 입력 임베딩에서 목표 임베딩 또는 delta를 예측하는 residual MLP를 학습한다."),
    ("body", "6) 장문 생성: JEPA 방향, 검색 예시, 장면 프리셋, hallucination contract를 결합해 한국어 소설 섹션을 생성한다."),
    ("body", "7) 장기 기억: 각 섹션의 요약, 사실, 단서, 위치, 관계, 상태 변화를 JSONL 메모리로 저장한다."),
    ("body", "8) 이어쓰기: 저장된 초안, 메모리, KG/state ledger, run state를 불러와 다음 턴을 이어서 생성한다."),
    ("body", "9) 반복 방지: consumed beat ledger가 이미 사용된 폭로, 경고, 이동, 단서 회수를 추적하고 반복 시 한 번 재작성한다."),
    ("body", "10) 평가: 생성 결과에 대해 길이, 반복, 창의 확장, 이름 일관성, 섹션 구조, 인접 섹션 유사도를 계산한다."),
    ("h2", "(2) 사용성 및 성능 요구사항"),
    ("body", "1) Streamlit GUI에서 데이터 생성, 임베딩, 학습, 생성, 평가를 버튼 중심으로 실행할 수 있어야 한다."),
    ("body", "2) Dry-run mode를 제공하여 Ollama 없이도 전체 흐름을 점검할 수 있어야 한다."),
    ("body", "3) RTX 4060 8GB VRAM 환경을 고려하여 context length, GPU layer, batch, max token을 조정할 수 있어야 한다."),
    ("body", "4) 장문 생성은 한 번에 30,000자를 요청하지 않고 섹션 단위로 나누어 실패 시 중간 결과를 보존해야 한다."),
    ("body", "5) 모델 로딩 실패나 HTTP 500 오류가 발생하면 보수적 fallback 옵션으로 1회 복구를 시도해야 한다."),
    ("body", "6) 생성 중에는 실시간 스트리밍과 trace table을 제공하여 긴 대기 시간을 사용자가 확인할 수 있어야 한다."),
    ("h2", "(3) 개발환경"),
    ("body", "운영체제: Windows 환경을 기준으로 개발하였다."),
    ("body", "하드웨어 목표: RTX 4060 8GB VRAM, 32GB RAM 환경에서 동작하도록 설정하였다."),
    ("body", "언어 및 프레임워크: Python 3.11 계열, Streamlit, PyTorch, FAISS CPU, pydantic, PyYAML을 사용하였다."),
    ("body", "LLM 런타임: Ollama API를 사용하며 기본 chat model은 gemma4:12b-it-q4_K_M, embedding model은 embeddinggemma:latest로 설정하였다."),
    ("body", "데이터셋 구조: world, characters, scene_t, scene_t_plus_1, metadata로 구성된 JSONL 전환 샘플을 사용한다."),
    ("body", "전처리: JSON object 추출, schema validation, 최소 길이 검사, 장르 프리셋 다양성 축 부여, embedding cache 재사용을 적용하였다."),
    ("body", "학습 모델: 입력 차원은 임베딩 차원이며 hidden dim 1024, layer 4, dropout 0.1의 residual MLP predictor를 사용하였다."),
    ("body", "학습 설정: epochs 80, batch size 32, learning rate 1e-4, weight decay 0.01, validation ratio 0.15, early stopping patience 12를 기본값으로 두었다."),
    ("body", "손실 함수: cosine alignment를 중심으로 MSE 보조 항을 결합하고, normalize_prediction 설정 시 norm regularization을 비활성화하였다."),
    ("h2", "(4) 인터페이스 정의"),
    ("body", "입력 인터페이스: 장르, 장면 프리셋, 세계관, 인물표, 이전 장면, 생성 턴 길이, 이어쓰기 지시문, Ollama 설정을 입력한다."),
    ("body", "출력 인터페이스: filtered.jsonl, scenes.npz, FAISS index, predictor checkpoint, long-form markdown, memory JSONL, ledger JSON, evaluation report를 저장한다."),
    ("body", "사용자 인터페이스: Project, Chat, Dataset, Embedding, Train, Generate, Evaluate, Reports 탭으로 구성하였다."),
    ("body", "공유 인터페이스: GitHub repository와 Hugging Face artifact bundle을 통해 코드와 모델 산출물을 분리 공유할 수 있다."),
    ("blank", ""),
    ("h1", "3. 전체 구성"),
    ("h2", "(1) 시스템 구성도"),
    ("body", "그림 1. 시스템 구성도"),
    ("figure", "architecture"),
    ("body", "Streamlit GUI -> Ollama Chat/Embedding -> JSONL Filter -> Embedding Cache -> FAISS Index -> JEPA Predictor -> Story-memory RAG -> Korean Novel Output -> Evaluation Report"),
    ("body", "사용자는 GUI에서 버튼을 클릭하고, 시스템은 단계별 산출물을 data, checkpoints, reports 디렉터리에 저장한다."),
    ("body", "Ollama는 합성 데이터 생성과 소설 생성, 임베딩 생성을 담당하며, PyTorch predictor는 다음 장면 잠재 표현만 학습한다."),
    ("h2", "(2) 처리 흐름도"),
    ("body", "그림 2. 처리 흐름도"),
    ("figure", "flow"),
    ("body", "1) 장르와 프리셋 선택 -> 2) 합성 샘플 생성 -> 3) JSONL 검증 -> 4) 임베딩 생성 -> 5) FAISS 인덱스 구축 -> 6) predictor 학습"),
    ("body", "7) 현재 장면 분석 -> 8) 다음 장면 임베딩 예측 -> 9) 유사 목표 장면 검색 -> 10) 섹션별 소설 생성 -> 11) 메모리 저장 -> 12) 평가 보고서 작성"),
    ("h2", "(3) 주요 기능 설명"),
    ("body", "합성 데이터 생성 기능은 genre preset과 diversity plan을 결합하여 다양한 서사 전환 샘플을 만든다."),
    ("body", "Scene analyzer는 사용자가 입력한 현재 장면을 요약, 감정, 갈등, 상태, plot function으로 구조화한다."),
    ("body", "JEPA-inspired predictor는 현재 장면 표현에서 다음 장면 표현을 예측하여 생성 방향을 검색 가능한 벡터로 바꾼다."),
    ("body", "Creative Hallucination + JEPA 모드는 planner 방향을 유지하면서 단서, 상징, 감각 묘사, 감정 추론을 창의적으로 확장한다."),
    ("body", "Story-memory RAG는 이전 섹션의 사실, 미해결 단서, 최신 상태, 관계 triple을 검색하여 장편 일관성을 보강한다."),
    ("body", "Consumed beat ledger는 이미 사용된 서사 기능을 canon으로 기록하고, 새 섹션이 같은 폭로를 반복하지 않도록 한다."),
    ("body", "Portable bundle 기능은 초안, memory, KG/state, turn progress, 세계관, 인물표를 ZIP으로 내보내고 복원한다."),
    ("blank", ""),
    ("h1", "4. 세부 추진전략"),
    ("h2", "(1) 프로젝트 진행 전략"),
    ("body", "첫째, 복잡한 LLM fine-tuning 대신 소규모 predictor 학습으로 연구 범위를 제한하였다."),
    ("body", "둘째, 모든 중간 산출물을 파일로 저장하여 실패 후에도 동일 단계부터 재시작할 수 있게 하였다."),
    ("body", "셋째, Dry-run mode와 smoke test를 구축하여 실제 Ollama 호출 없이도 UI와 pipeline 오류를 검출하였다."),
    ("body", "넷째, 교수 피드백의 hallucination 요구를 반영하여 창의 확장을 정량 지표로 분리하였다."),
    ("body", "다섯째, 로컬 자원 한계를 고려하여 30,000자 목표를 여러 섹션과 여러 턴으로 나누어 생성하였다."),
    ("h2", "(2) 개발 일정"),
    ("body", "표 1. 14주 개발 일정표"),
    ("figure", "schedule"),
    ("body", "세부 일정 문서의 수업 계획과 GitHub 커밋 38건을 대조하여 팀 구성부터 최종 보고서·시연 준비까지 14주 과정으로 재구성하였다."),
    ("h2", "(3) 팀원 역할"),
    ("body", "[기입 필요] 리더: 일정 관리, 전체 구조 설계, 발표 자료 취합"),
    ("body", "[기입 필요] 팀원 1: 데이터 생성, JSON schema, 프리셋 설계"),
    ("body", "[기입 필요] 팀원 2: predictor 학습, FAISS 검색, 평가 지표 구현"),
    ("body", "[기입 필요] 팀원 3: Streamlit UI, 장문 생성, 문서화 및 시연 준비"),
    ("body", "AI agent 사용: Codex는 코드 수정, 오류 분석, 문서 초안 작성, GitHub 공유 절차 보조에 사용하였다."),
    ("body", "Local LLM 사용: Gemma 계열 모델은 합성 데이터 생성과 한국어 소설 본문 생성에 사용하였다."),
    ("blank", ""),
    ("h1", "5. 결과"),
    ("h2", "(1) 최종 결과물"),
    ("body", "최종 결과물은 로컬 PC에서 실행 가능한 Streamlit 기반 Novel JEPA Lab GUI이다."),
    ("body", "Project 탭은 데이터 생성부터 학습, 생성, 평가까지 한 번에 실행하는 full pipeline을 제공한다."),
    ("body", "Generate 탭은 5,000자, 10,000자, 사용자 지정 길이로 장문 소설을 생성하고 이어쓸 수 있다."),
    ("body", "Chat 탭은 세션별 memory summary와 knowledge graph를 유지하며 대화형 장면 생성을 지원한다."),
    ("body", "Reports 탭은 cache, artifacts, evaluation report를 확인할 수 있게 하였다."),
    ("body", "생성 결과는 creative_longform_latest.md, creative_longform_memory.jsonl, creative_longform_ledger.json, creative_longform_state.json으로 저장된다."),
    ("body", "그림 3. 최종 결과물 화면"),
    ("figure", "result"),
    ("body", "GUI 화면에는 stage table, live output, retrieval/planner diagnostics, memory count, retry count, checkpoint path가 표시된다."),
    ("h2", "(2) 성능 분석 결과"),
    ("body", "정량 검증은 scripts/smoke_jepa.py를 통해 dry-run pipeline 전체가 통과하는지 확인하였다."),
    ("body", "검증 항목은 synthetic data 생성, filtering, embedding, FAISS index, predictor training, JEPA generation, long-form continuation, bundle import/export이다."),
    ("body", "평가 보고서는 embedding continuity, keyword consistency, novelty, lexical diversity, length fit, progression score, dialogue ratio를 계산한다."),
    ("body", "Controlled hallucination 평가는 creative expansion rate, target alignment, useful hallucination score, hallucination risk를 제공한다."),
    ("body", "장문 구조 평가는 section count fit, section body coverage, average section body chars로 확인한다."),
    ("body", "반복 방지 평가는 repeated subtitle count, repeated narrative beat count, adjacent section similarity, retry success rate를 제공한다."),
    ("body", "자원 측면에서는 Ollama 8K context, num_gpu 24, num_batch 32를 기본값으로 두고 실패 시 4K context와 낮은 GPU layer fallback을 사용한다."),
    ("body", "최근 개선에서는 반복 가드가 켜져도 섹션 초안을 즉시 스트리밍하고, 반복 판정 시 같은 자리에서 재작성본으로 교체하도록 수정하였다."),
    ("body", "그림 4. 평가 지표 구조도"),
    ("figure", "metrics"),
    ("h2", "(3) 구현 세부 결과"),
    ("body", "합성 데이터 모듈은 장르별 scene preset을 순환 적용하여 동일한 장르 안에서도 감정 변화, 단서 유형, 압박 원천이 다르게 나타나도록 하였다."),
    ("body", "각 샘플은 현재 장면 scene_t와 다음 장면 scene_t_plus_1을 분리하여 저장하므로 predictor 학습에서 입력과 목표 표현을 명확히 나눌 수 있다."),
    ("body", "데이터 필터링 단계는 JSON 문법 오류, 필수 필드 누락, 지나치게 짧은 요약, schema 불일치를 제거하여 학습 데이터 품질을 유지한다."),
    ("body", "임베딩 단계는 동일 텍스트와 동일 모델 조합을 cache key로 저장하여 반복 실행 시 Ollama embedding 호출 횟수를 줄인다."),
    ("body", "FAISS 단계는 current-context index와 next-scene index를 모두 구축하여 단순 RAG와 JEPA 예측 검색을 비교할 수 있게 하였다."),
    ("body", "학습 단계는 validation split을 checkpoint에 저장하여 훈련 데이터 검색 결과를 성능으로 과대 해석하지 않도록 하였다."),
    ("body", "predictor는 입력 표현과 목표 표현의 cosine alignment를 높이는 방향으로 학습되며, delta prediction 옵션으로 전환 벡터를 예측할 수 있다."),
    ("body", "생성 단계는 검색된 다음 장면 예시 전체를 그대로 넣지 않고, 요약된 beat card만 넣어 prompt 길이와 잡음을 줄였다."),
    ("body", "Controlled hallucination contract는 새로운 단서, 감각 묘사, 상징, 감정 추론을 허용하되 세계관 규칙과 인물명은 보존하도록 제한한다."),
    ("body", "장문 생성은 섹션 하나마다 private story-memory block을 함께 생성하여 별도의 요약 LLM 호출 없이 memory ledger를 갱신한다."),
    ("body", "Story-memory record에는 summary, facts, open clues, resolved clues, locations, state changes, state updates, relations, keywords가 포함된다."),
    ("body", "Knowledge graph는 relation triple을 사용하여 소유, 신뢰, 은폐, 추적, 위치, 원인 관계를 추적한다."),
    ("body", "Hierarchical summary는 여러 섹션을 묶어 압축 timeline을 만들고, 8K context 안에서 장기 흐름을 유지하는 역할을 한다."),
    ("body", "Consumed beat ledger는 reveal, alliance shift, system warning, location move, clue resolution, new threat, emotional turn 유형을 구분한다."),
    ("body", "반복 판정은 단순 키워드 중복이 아니라 동일 유형의 trigger와 고유 내용이 함께 겹칠 때만 재시도를 실행하도록 조정하였다."),
    ("body", "Streamlit 출력부는 섹션 단위 renderer를 사용하여 재작성 전 초안이 채택본과 중복 표시되지 않도록 구현하였다."),
    ("body", "Ollama client는 이미 unload된 모델을 다시 unload하지 않도록 상태를 기억하여 섹션 사이의 불필요한 API 호출을 줄였다."),
    ("body", "오류 처리 측면에서는 checkpoint, memory, ledger, state를 매 섹션마다 저장하므로 runner failure가 발생해도 직전 완료 섹션까지 보존된다."),
    ("h2", "(4) 검증 시나리오"),
    ("body", "검증 시나리오 1은 dry-run mode에서 전체 pipeline이 실행되는지 확인하는 것이다."),
    ("body", "이 시나리오는 실제 모델 호출 없이 synthetic data, filtering, embedding, FAISS, training, generation, continuation 경로를 점검한다."),
    ("body", "검증 시나리오 2는 predictor checkpoint가 생성되고 validation diagnostics가 available 상태로 보고되는지 확인하는 것이다."),
    ("body", "검증 시나리오 3은 장문 생성 결과에 MEMORY_START marker가 노출되지 않고, 두 개 이상의 제목 섹션이 생성되는지 확인하는 것이다."),
    ("body", "검증 시나리오 4는 continuation bundle을 ZIP으로 내보낸 뒤 다른 output_root에 가져와 본문과 memory count가 유지되는지 확인하는 것이다."),
    ("body", "검증 시나리오 5는 markdown draft를 가져왔을 때 memory가 prose로부터 재구축되는지 확인하는 것이다."),
    ("body", "검증 시나리오 6은 consumed beat ledger가 같은 폭로를 반복할 때 감지하고, 서로 다른 새 폭로는 오탐하지 않는지 확인하는 것이다."),
    ("body", "검증 시나리오 7은 교체형 stream renderer가 반복 초안을 최종 화면에 남기지 않고 채택 본문만 남기는지 확인하는 것이다."),
    ("body", "검증 시나리오 8은 evaluation report가 repeated subtitle, repeated narrative beat, adjacent similarity를 별도 지표로 계산하는지 확인하는 것이다."),
    ("h2", "(5) 산출물 목록"),
    ("body", "data/synthetic/generated.jsonl은 Ollama 또는 dry-run이 생성한 원본 합성 서사 전환 샘플이다."),
    ("body", "data/filtered/filtered.jsonl은 schema validation과 품질 필터를 통과한 학습 대상 샘플이다."),
    ("body", "data/embeddings/scenes.npz는 현재 장면과 다음 장면 목표의 임베딩 행렬 및 backend 정보를 저장한다."),
    ("body", "data/indexes/current_context.faiss와 data/indexes/next_scene.faiss는 검색 비교와 JEPA 방향 검색에 사용된다."),
    ("body", "checkpoints/predictor/best.pt는 residual MLP predictor의 최적 가중치와 학습 metadata를 저장한다."),
    ("body", "checkpoints/predictor/model_card.json은 최신 학습 결과와 validation 중심 지표를 사람이 읽기 쉽게 정리한다."),
    ("body", "reports/runs/creative_longform_latest.md는 장문 생성 중 가장 최신의 전체 초안을 저장한다."),
    ("body", "reports/runs/creative_longform_memory.jsonl은 섹션별 memory record를 줄 단위 JSON으로 저장한다."),
    ("body", "reports/runs/creative_longform_ledger.json은 최신 상태, 지식 그래프, 단서 ledger, 압축 timeline을 저장한다."),
    ("body", "reports/runs/creative_longform_state.json은 현재 턴, 전체 글자 수, 섹션 수, 반복 재시도 통계를 기록한다."),
    ("body", "reports/runs/comparison_*.md는 생성 결과 평가 보고서이며 발표와 최종보고서 근거 자료로 활용할 수 있다."),
    ("blank", ""),
    ("h1", "6. 고찰"),
    ("body", "첫 번째 어려움은 로컬 모델의 VRAM 한계였다. 12B Q4 모델은 RTX 4060 8GB에서 context와 batch 설정에 따라 runner failure가 발생할 수 있었다."),
    ("body", "이를 해결하기 위해 GPU layer, context length, fallback 옵션, 모델 unload 관리, section 단위 생성을 도입하였다."),
    ("body", "두 번째 어려움은 장편 생성의 기억 손실이었다. 최근 문맥만 넣으면 과거 단서나 관계 변화가 쉽게 사라졌다."),
    ("body", "이를 해결하기 위해 섹션별 story memory, 최신 state ledger, unresolved clue ledger, knowledge graph, hierarchical summary를 결합하였다."),
    ("body", "세 번째 어려움은 hallucination의 의미 정의였다. 단순 오류로 보면 프로젝트 가치가 약해지므로, 창의적 확장과 위험한 drift를 구분하였다."),
    ("body", "네 번째 어려움은 동일한 폭로와 감정 beat가 반복되는 문제였다. consumed beat ledger와 고신뢰 반복 판정으로 같은 사건의 재서술을 줄였다."),
    ("body", "한계점은 synthetic data 규모가 작을 경우 predictor의 일반화 성능이 제한된다는 점이다."),
    ("body", "또한 자동 평가는 문체의 아름다움이나 독자 몰입감을 완전히 측정하지 못하므로 사람 평가 또는 LLM judge가 추가될 필요가 있다."),
    ("body", "향후에는 더 큰 genre별 데이터셋, 실제 사용자 평가, KG 기반 제약 강화, 시각적 story graph 편집 기능을 추가할 수 있다."),
    ("blank", ""),
    ("h1", "7. 결론"),
    ("body", "본 프로젝트는 한국어 장편소설 생성을 위해 로컬 LLM, JEPA-inspired latent predictor, RAG memory를 통합한 실험 시스템을 구현하였다."),
    ("body", "LLM 자체를 fine-tuning하지 않고도 작은 predictor와 검색 기반 planning으로 다음 장면 방향을 제시할 수 있음을 확인하였다."),
    ("body", "Story-memory RAG와 consumed beat ledger는 장문 생성에서 개연성 유지와 반복 감소에 실질적으로 기여하였다."),
    ("body", "최종적으로 데이터 생성, 학습, 생성, 이어쓰기, 평가, 공유가 가능한 캡스톤 시연용 GUI와 산출물 관리 구조를 완성하였다."),
    ("blank", ""),
    ("h1", "8. 참고문헌"),
    ("body", "1) Yann LeCun et al., A Path Towards Autonomous Machine Intelligence, 2022."),
    ("body", "2) Meta AI Research, I-JEPA: Self-supervised learning from images, 2023."),
    ("body", "3) Ollama Documentation, https://ollama.com"),
    ("body", "4) Streamlit Documentation, https://docs.streamlit.io"),
    ("body", "5) PyTorch Documentation, https://pytorch.org/docs"),
    ("body", "6) FAISS Documentation, https://faiss.ai"),
    ("body", "7) Pydantic Documentation, https://docs.pydantic.dev"),
]


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/malgunbd.ttf" if bold else "C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/gulim.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: str,
    outline: str = "#D7DEE8",
    radius: int = 18,
    width: int = 2,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def _arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#2F5E8E") -> None:
    draw.line([start, end], fill=color, width=3)
    x1, y1 = start
    x2, y2 = end
    if abs(x2 - x1) >= abs(y2 - y1):
        direction = 1 if x2 >= x1 else -1
        points = [(x2, y2), (x2 - 12 * direction, y2 - 7), (x2 - 12 * direction, y2 + 7)]
    else:
        direction = 1 if y2 >= y1 else -1
        points = [(x2, y2), (x2 - 7, y2 - 12 * direction), (x2 + 7, y2 - 12 * direction)]
    draw.polygon(points, fill=color)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    fill: str,
    accent: str,
) -> None:
    _rounded_box(draw, xy, fill=fill)
    x1, y1, x2, _y2 = xy
    draw.rounded_rectangle((x1, y1, x1 + 8, _y2), radius=6, fill=accent)
    title_font = _font(18, bold=True)
    body_font = _font(13)
    draw.text((x1 + 20, y1 + 14), title, font=title_font, fill="#172033")
    y = y1 + 44
    for line in _wrap_text(draw, body, body_font, x2 - x1 - 36)[:3]:
        draw.text((x1 + 20, y), line, font=body_font, fill="#4D5B6B")
        y += 18


def _save_architecture(path: Path) -> None:
    img = Image.new("RGB", (736, 572), "#F7FAFC")
    draw = ImageDraw.Draw(img)
    draw.text((28, 24), "Novel JEPA Lab 시스템 구성도", font=_font(28, bold=True), fill="#162033")
    draw.text((30, 62), "로컬 LLM은 고정하고 작은 predictor만 학습하는 장편 소설 생성 구조", font=_font(15), fill="#52606D")
    boxes = [
        ((32, 120, 190, 218), "Streamlit GUI", "Project, Generate, Chat, Reports", "#FFFFFF", "#6C8AE4"),
        ((290, 108, 452, 206), "Ollama", "Gemma chat + EmbeddingGemma", "#FFFFFF", "#1F9D8A"),
        ((546, 120, 704, 218), "Artifacts", "data / checkpoints / reports", "#FFFFFF", "#D97706"),
        ((32, 318, 190, 430), "Synthetic JSONL", "scene_t -> scene_t+1 전환 데이터", "#FFFFFF", "#7C3AED"),
        ((290, 310, 452, 438), "JEPA Predictor", "현재 장면 벡터에서 다음 장면 벡터 예측", "#FFFFFF", "#2563EB"),
        ((546, 318, 704, 430), "Story Memory RAG", "상태 ledger, KG, consumed beat", "#FFFFFF", "#DC2626"),
    ]
    for box in boxes:
        _draw_label(draw, *box)
    _arrow(draw, (190, 168), (290, 158))
    _arrow(draw, (452, 158), (546, 168))
    _arrow(draw, (110, 218), (110, 318))
    _arrow(draw, (190, 374), (290, 374))
    _arrow(draw, (452, 374), (546, 374))
    _arrow(draw, (625, 318), (625, 218))
    draw.rounded_rectangle((132, 472, 604, 530), radius=18, fill="#E8F1FF", outline="#B6C9F5", width=2)
    draw.text((158, 488), "최종 출력: 한국어 장편소설 초안 + 메모리 원장 + 평가 보고서", font=_font(18, bold=True), fill="#1F3763")
    img.save(path)


def _save_flow(path: Path) -> None:
    img = Image.new("RGB", (748, 1146), "#F8FAFC")
    draw = ImageDraw.Draw(img)
    draw.text((32, 28), "처리 흐름도", font=_font(34, bold=True), fill="#162033")
    draw.text((34, 76), "데이터 생성부터 장문 이어쓰기와 평가까지의 실행 순서", font=_font(17), fill="#52606D")
    steps = [
        ("1. 입력 설정", "장르, 프리셋, 세계관, 인물표, 이전 장면"),
        ("2. 합성 데이터", "Ollama JSON mode로 scene transition 생성"),
        ("3. 검증/필터", "pydantic schema와 최소 품질 조건 검사"),
        ("4. 임베딩/FAISS", "현재 맥락과 다음 장면 목표를 벡터화"),
        ("5. Predictor 학습", "Residual MLP로 다음 장면 표현 예측"),
        ("6. JEPA 계획", "예측 벡터로 유사 목표 장면 검색"),
        ("7. 장문 생성", "Creative Hallucination + JEPA 섹션 작성"),
        ("8. Memory 갱신", "facts, clues, states, KG, consumed beat 저장"),
        ("9. 평가/공유", "보고서 작성, bundle export, GitHub/HF 공유"),
    ]
    y = 132
    for i, (title, body) in enumerate(steps):
        accent = "#2563EB" if i < 5 else "#C2410C" if i < 7 else "#047857"
        _draw_label(draw, (92, y, 656, y + 78), title, body, "#FFFFFF", accent)
        if i < len(steps) - 1:
            _arrow(draw, (374, y + 78), (374, y + 110), "#52606D")
        y += 112
    draw.rounded_rectangle((72, 1058, 676, 1110), radius=18, fill="#FFF7ED", outline="#FDBA74", width=2)
    draw.text((106, 1074), "중간 산출물을 매 단계 저장하여 실패 후에도 재시작 가능", font=_font(18, bold=True), fill="#9A3412")
    img.save(path)


def _save_schedule(path: Path) -> None:
    img = Image.new("RGB", (748, 1146), "#F8FAFC")
    draw = ImageDraw.Draw(img)
    draw.text((28, 26), "14주 개발 일정표", font=_font(29, bold=True), fill="#162033")
    draw.text((30, 66), "수업 세부 일정과 GitHub 커밋 38건을 대조한 실제 추진 내역", font=_font(15), fill="#52606D")

    x0, y0 = 24, 108
    col_widths = [48, 122, 280, 184, 66]
    headers = ["주차", "단계", "주요 활동 및 구현", "산출물 / Git 근거", "상태"]
    rows = [
        ("1주", "문제 탐색", "캡스톤 오리엔테이션, 생성형 AI·소설 생성 문제 조사", "아이디어 후보 목록", "완료"),
        ("2주", "팀 구성", "팀 구성, 역할 분담, 협업 방식과 개발 환경 조사", "세부 일정.docx", "완료"),
        ("3주", "주제 확정", "로컬 LLM과 JEPA를 결합한 한국어 소설 생성 주제 확정", "프로젝트 주제·목표", "완료"),
        ("4주", "계획 수립", "요구사항, 파이프라인, 산출물, 위험요소를 계획서로 정리", "계획서·구성도", "완료"),
        ("5주", "피드백", "계획서 피드백 반영, LLM fine-tuning 대신 predictor 학습 범위 확정", "수정 계획서", "완료"),
        ("6주", "개발 시작", "Streamlit scaffold, YAML config, Ollama client, 재시작 구조 구현", "a220ce3, d1fac12", "완료"),
        ("7주", "환경/발표", "CUDA 학습, Windows launcher, live pipeline dashboard 구축", "2f8645a, acd49db", "완료"),
        ("8주", "LLM+RAG", "합성 데이터, 장문 메모리 세션, 캐시·평가 파이프라인 구현", "e7bc4cc, 0926264", "완료"),
        ("9주", "단편 테스트", "JSON 추출·schema 안정화, 장르 프리셋, beat card, 이름 일관성 검증", "aa5801d, 8a9dbd5", "완료"),
        ("10주", "코드 고도화", "Ollama 복구, 다양성 계획, trace, cache 분리, 저장소 정리 기능 보강", "5012dd3, 4b062d3", "완료"),
        ("11주", "JEPA 삽입", "Residual predictor, retrieval baseline, planner 진단과 checkpoint metadata 구현", "d0f335a, dbf8c65", "완료"),
        ("12주", "장편 생성", "섹션별 생성, 사용자 길이 설정, 저장 파일 이어쓰기와 장편 초안 생성", "0926264, b0ea768", "완료"),
        ("13주", "Hallucination", "Creative Hallucination + JEPA 모드와 창의 확장·위험 평가 지표 추가", "b0ea768, cf5f712", "완료"),
        ("14주", "최종 고도화", "Story-memory RAG, KG/state ledger, portable bundle, 일관성·대기시간 개선, 보고서 작성", "de64f3e~141fca3", "진행"),
    ]

    header_h = 42
    row_h = 66
    table_w = sum(col_widths)
    draw.rounded_rectangle((x0, y0, x0 + table_w, y0 + header_h + row_h * len(rows)), radius=14, fill="#FFFFFF", outline="#CBD5E1", width=2)
    draw.rounded_rectangle((x0, y0, x0 + table_w, y0 + header_h), radius=14, fill="#1E3A8A")
    draw.rectangle((x0, y0 + 20, x0 + table_w, y0 + header_h), fill="#1E3A8A")

    x = x0
    for header, width in zip(headers, col_widths):
        draw.text((x + 8, y0 + 11), header, font=_font(12, bold=True), fill="#FFFFFF")
        x += width

    y = y0 + header_h
    for idx, row in enumerate(rows):
        fill = "#FFFFFF" if idx % 2 == 0 else "#F1F5F9"
        draw.rectangle((x0, y, x0 + table_w, y + row_h), fill=fill)
        x = x0
        for col, (text, width) in enumerate(zip(row, col_widths)):
            font = _font(11, bold=col in (0, 1, 4))
            color = "#0F172A" if col != 4 else "#047857" if text == "완료" else "#C2410C"
            lines = _wrap_text(draw, text, font, width - 12)[:3]
            text_h = len(lines) * 15
            yy = y + max(8, (row_h - text_h) // 2)
            for line in lines:
                draw.text((x + 6, yy), line, font=font, fill=color)
                yy += 15
            x += width
        y += row_h

    x = x0
    for width in col_widths[:-1]:
        x += width
        draw.line((x, y0, x, y0 + header_h + row_h * len(rows)), fill="#CBD5E1", width=1)
    for i in range(len(rows) + 1):
        yy = y0 + header_h + i * row_h
        draw.line((x0, yy, x0 + table_w, yy), fill="#CBD5E1", width=1)

    draw.rounded_rectangle((62, 1090, 686, 1130), radius=14, fill="#E0F2FE", outline="#7DD3FC", width=2)
    draw.text((92, 1100), "현재 상태: 핵심 기능 완료 · 최종 보고서/시연 자료 보완 단계", font=_font(15, bold=True), fill="#075985")
    img.save(path)


def _save_result(path: Path) -> None:
    img = Image.new("RGB", (736, 572), "#F6F8FB")
    draw = ImageDraw.Draw(img)
    draw.text((28, 24), "최종 결과물 화면", font=_font(28, bold=True), fill="#162033")
    draw.text((30, 62), "Streamlit GUI에서 장문 생성, 메모리, 평가, 산출물 저장을 통합 관리", font=_font(15), fill="#52606D")
    _rounded_box(draw, (30, 98, 706, 528), "#FFFFFF", "#CBD5E1", 20, 2)
    draw.rounded_rectangle((30, 98, 706, 142), radius=20, fill="#172033")
    draw.text((52, 110), "Novel JEPA Lab", font=_font(20, bold=True), fill="#FFFFFF")

    tabs = [("Project", False), ("Generate", True), ("Chat", False), ("Evaluate", False), ("Reports", False)]
    x = 224
    for tab, active in tabs:
        fill = "#2563EB" if active else "#334155"
        draw.rounded_rectangle((x, 110, x + 86, 134), radius=8, fill=fill)
        draw.text((x + 9, 115), tab, font=_font(11, bold=active), fill="#E2E8F0")
        x += 92

    _rounded_box(draw, (50, 162, 214, 498), "#F8FAFC", "#D7DEE8", 12, 1)
    draw.text((66, 178), "실행 설정", font=_font(17, bold=True), fill="#162033")
    settings = [
        ("Model", "gemma4:12b q4_K_M"),
        ("Context", "8K"),
        ("Turn", "10,000자"),
        ("Mode", "Creative + JEPA"),
        ("Memory RAG", "ON"),
    ]
    y = 216
    for label, value in settings:
        draw.text((66, y), label, font=_font(11), fill="#64748B")
        draw.rounded_rectangle((66, y + 17, 196, y + 42), radius=7, fill="#E2E8F0")
        draw.text((76, y + 22), value, font=_font(10, bold=True), fill="#0F172A")
        y += 54

    _rounded_box(draw, (234, 162, 490, 498), "#FFFFFF", "#D7DEE8", 12, 1)
    draw.text((252, 178), "Live Output", font=_font(17, bold=True), fill="#162033")
    output_lines = [
        "### 4장. 사라진 별의 지도",
        "복도 끝의 창문은 푸른빛으로 떨렸다.",
        "이전 장면의 단서는 memory block에서",
        "불러와 인물의 감정선과 세계관 규칙을",
        "유지한 채 다음 사건으로 확장된다.",
        "",
        "[저장] creative_longform_latest.md",
    ]
    yy = 214
    for line in output_lines:
        fill = "#2563EB" if line.startswith("###") else "#334155"
        draw.text((252, yy), line, font=_font(12, bold=line.startswith("###")), fill=fill)
        yy += 24
    draw.rounded_rectangle((252, 446, 472, 472), radius=8, fill="#DCFCE7", outline="#86EFAC")
    draw.text((268, 452), "Section 4/8 saved · 이어쓰기 가능", font=_font(11, bold=True), fill="#166534")

    _rounded_box(draw, (510, 162, 684, 318), "#F8FAFC", "#D7DEE8", 12, 1)
    draw.text((526, 178), "Planner 진단", font=_font(16, bold=True), fill="#162033")
    diag = [("retrieval hit", "0.82"), ("memory hits", "6"), ("retry count", "1")]
    yy = 214
    for label, value in diag:
        draw.text((526, yy), label, font=_font(11), fill="#64748B")
        draw.text((644, yy), value, font=_font(12, bold=True), fill="#0F172A")
        yy += 28

    _rounded_box(draw, (510, 340, 684, 498), "#F8FAFC", "#D7DEE8", 12, 1)
    draw.text((526, 356), "Reports", font=_font(16, bold=True), fill="#162033")
    report_lines = ["evaluation.md", "memory.jsonl", "ledger.json", "bundle.zip"]
    yy = 392
    for line in report_lines:
        draw.ellipse((526, yy + 6, 534, yy + 14), fill="#D97706")
        draw.text((542, yy), line, font=_font(12), fill="#334155")
        yy += 24
    img.save(path)


def _save_metrics(path: Path) -> None:
    img = Image.new("RGB", (736, 572), "#F8FAFC")
    draw = ImageDraw.Draw(img)
    draw.text((28, 24), "평가 지표 구조도", font=_font(28, bold=True), fill="#162033")
    draw.text((30, 62), "창의적 확장과 개연성 유지 여부를 분리해서 측정", font=_font(15), fill="#52606D")
    groups = [
        ((40, 116, 332, 238), "Continuity", ["embedding continuity", "keyword consistency", "name consistency"], "#2563EB"),
        ((404, 116, 696, 238), "Creative Hallucination", ["creative expansion rate", "target alignment", "hallucination risk"], "#7C3AED"),
        ((40, 294, 332, 416), "Structure", ["section count fit", "body coverage", "length fit"], "#047857"),
        ((404, 294, 696, 416), "Repetition Guard", ["repeated subtitle", "repeated beat", "retry success rate"], "#DC2626"),
    ]
    for xy, title, bullets, accent in groups:
        _rounded_box(draw, xy, "#FFFFFF", "#D7DEE8", 18, 2)
        x1, y1, x2, y2 = xy
        draw.rounded_rectangle((x1, y1, x2, y1 + 34), radius=18, fill=accent)
        draw.text((x1 + 18, y1 + 7), title, font=_font(16, bold=True), fill="#FFFFFF")
        y = y1 + 50
        for bullet in bullets:
            draw.ellipse((x1 + 20, y + 5, x1 + 28, y + 13), fill=accent)
            draw.text((x1 + 38, y), bullet, font=_font(14), fill="#334155")
            y += 24
    _arrow(draw, (332, 177), (404, 177), "#64748B")
    _arrow(draw, (332, 355), (404, 355), "#64748B")
    draw.rounded_rectangle((144, 472, 592, 528), radius=18, fill="#E0F2FE", outline="#7DD3FC", width=2)
    draw.text((178, 488), "최종 보고서: 정량 지표 + 산출물 경로 + 한계 및 개선 방향", font=_font(17, bold=True), fill="#075985")
    img.save(path)


def create_visuals() -> dict[str, Path]:
    VISUAL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "architecture": VISUAL_DIR / "capstone_architecture.png",
        "flow": VISUAL_DIR / "capstone_flow.png",
        "schedule": VISUAL_DIR / "capstone_schedule.png",
        "result": VISUAL_DIR / "capstone_result.png",
        "metrics": VISUAL_DIR / "capstone_metrics.png",
    }
    _save_architecture(paths["architecture"])
    _save_flow(paths["flow"])
    _save_schedule(paths["schedule"])
    _save_result(paths["result"])
    _save_metrics(paths["metrics"])
    return paths


def _make_para(template: etree._Element, text: str, char_pr_id: str) -> etree._Element:
    para = deepcopy(template)
    for child in list(para):
        para.remove(child)
    run = etree.Element(RUN, charPrIDRef=char_pr_id)
    if text:
        t = etree.SubElement(run, TEXT)
        t.text = text
    para.append(run)
    return para


def _make_title_para(template: etree._Element, text: str) -> etree._Element:
    para = deepcopy(template)
    for child in list(para):
        para.remove(child)
    sec_run = deepcopy(template.xpath("./hp:run", namespaces=NS)[0])
    para.append(sec_run)
    run = etree.Element(RUN, charPrIDRef="7")
    t = etree.SubElement(run, TEXT)
    t.text = text
    para.append(run)
    return para


def _make_figure_para(
    template: etree._Element,
    *,
    caption: str,
    image_id: str,
    figure_num: int,
) -> etree._Element:
    para = deepcopy(template)
    for caption_el in para.xpath(".//hp:caption", namespaces=NS):
        parent = caption_el.getparent()
        if parent is not None:
            parent.remove(caption_el)
    for img in para.xpath(".//*[local-name()='img']"):
        if "binaryItemIDRef" in img.attrib:
            img.set("binaryItemIDRef", image_id)
    for shape_comment in para.xpath(".//hp:shapeComment", namespaces=NS):
        shape_comment.text = f"그림입니다.\n원본 그림의 이름: {image_id}.png"

    tbl_counter = 0
    pic_counter = 0
    for elem in para.iter():
        local_name = etree.QName(elem).localname
        if local_name == "tbl":
            tbl_counter += 1
            elem.set("id", str(2009065800 + figure_num * 10 + tbl_counter))
            elem.set("zOrder", str(figure_num * 2))
        elif local_name == "pic":
            pic_counter += 1
            elem.set("id", str(2009065900 + figure_num * 10 + pic_counter))
            elem.set("instid", str(935324000 + figure_num * 10 + pic_counter))
            elem.set("zOrder", str(figure_num * 2 + pic_counter))
    return para


def _register_manifest_item(content_hpf: bytes, item_id: str, href: str) -> bytes:
    root = etree.fromstring(content_hpf)
    ns = {"opf": "http://www.idpf.org/2007/opf/"}
    manifest = root.xpath(".//opf:manifest", namespaces=ns)[0]
    existing = manifest.xpath(f"./opf:item[@id='{item_id}']", namespaces=ns)
    if existing:
        item = existing[0]
    else:
        item = etree.Element("{http://www.idpf.org/2007/opf/}item")
        manifest.insert(max(0, len(manifest) - 2), item)
    item.set("id", item_id)
    item.set("href", href)
    item.set("media-type", "image/png")
    item.set("isEmbeded", "1")
    return etree.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)


def build_report() -> None:
    visual_paths = create_visuals()
    with ZipFile(TEMPLATE, "r") as zin:
        infos = zin.infolist()
        payloads = {info.filename: zin.read(info.filename) for info in infos}
        root = etree.fromstring(payloads["Contents/section0.xml"])
        p_nodes = root.xpath("./hp:p", namespaces=NS)
        title_template = p_nodes[0]
        meta_template = p_nodes[3]
        h1_template = p_nodes[13]
        h2_template = p_nodes[16]
        body_template = p_nodes[18]
        architecture_figure_template = p_nodes[31]
        flow_figure_template = p_nodes[37]

        for child in list(root):
            root.remove(child)

        figure_specs = {
            "architecture": (architecture_figure_template, "시스템 구성도", "image1", 1),
            "flow": (flow_figure_template, "처리 흐름도", "image2", 2),
            "schedule": (flow_figure_template, "14주 개발 일정표", "image3", 3),
            "result": (architecture_figure_template, "최종 결과물 화면", "image4", 4),
            "metrics": (architecture_figure_template, "평가 지표 구조도", "image5", 5),
        }

        for kind, text in REPORT_ITEMS:
            if kind == "title":
                root.append(_make_title_para(title_template, text))
            elif kind == "meta":
                root.append(_make_para(meta_template, text, "0"))
            elif kind == "h1":
                root.append(_make_para(h1_template, text, "7"))
            elif kind == "h2":
                root.append(_make_para(h2_template, text, "7"))
            elif kind == "blank":
                root.append(_make_para(body_template, "", "0"))
            elif kind == "figure":
                template, caption, image_id, num = figure_specs[text]
                root.append(
                    _make_figure_para(
                        template,
                        caption=caption,
                        image_id=image_id,
                        figure_num=num,
                    )
                )
            else:
                root.append(_make_para(body_template, text, "0"))

        payloads["Contents/section0.xml"] = etree.tostring(
            root,
            encoding="UTF-8",
            xml_declaration=True,
            standalone=True,
        )
        payloads["Preview/PrvText.txt"] = "\r\n".join(
            text for kind, text in REPORT_ITEMS if text and kind != "figure"
        ).encode("utf-8")
        payloads["BinData/image1.png"] = visual_paths["architecture"].read_bytes()
        payloads["BinData/image2.png"] = visual_paths["flow"].read_bytes()
        payloads["BinData/image3.png"] = visual_paths["schedule"].read_bytes()
        payloads["BinData/image4.png"] = visual_paths["result"].read_bytes()
        payloads["BinData/image5.png"] = visual_paths["metrics"].read_bytes()
        payloads["Contents/content.hpf"] = _register_manifest_item(
            payloads["Contents/content.hpf"],
            "image3",
            "BinData/image3.png",
        )
        payloads["Contents/content.hpf"] = _register_manifest_item(
            payloads["Contents/content.hpf"],
            "image4",
            "BinData/image4.png",
        )
        payloads["Contents/content.hpf"] = _register_manifest_item(
            payloads["Contents/content.hpf"],
            "image5",
            "BinData/image5.png",
        )

        with ZipFile(OUT, "w") as zout:
            written = set()
            for info in infos:
                new_info = type(info)(info.filename, date_time=info.date_time)
                new_info.compress_type = info.compress_type
                new_info.external_attr = info.external_attr
                zout.writestr(new_info, payloads[info.filename])
                written.add(info.filename)
            for name in ["BinData/image3.png", "BinData/image4.png", "BinData/image5.png"]:
                if name not in written:
                    zout.writestr(name, payloads[name])

    md_lines: list[str] = []
    for kind, text in REPORT_ITEMS:
        if not text:
            md_lines.append("")
        elif kind == "title":
            md_lines.append("# " + text)
        elif kind == "h1":
            md_lines.append("## " + text)
        elif kind == "h2":
            md_lines.append("### " + text)
        elif kind == "figure":
            md_lines.append(f"![{text}]({visual_paths[text].as_posix()})")
        else:
            md_lines.append(text)
    MD_OUT.write_text("\n".join(md_lines), encoding="utf-8")
    print(OUT.resolve())
    print(MD_OUT.resolve())


if __name__ == "__main__":
    build_report()
