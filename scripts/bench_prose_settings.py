"""소설 본문 샘플링 설정 비교 벤치 (모델 서버 필요).

장 내부 구절 반복(rep12)·다양성(distinct3)·언어 누출·속도를 설정별로 잰다.
python scripts/bench_prose_settings.py --settings baseline,dry,minp_dry,rep10_dry --out .runtime/bench_prose.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generation.longform import SYSTEM_PROMPT, _distinct3, _max_repeated_span, _section_body  # noqa: E402
from src.llm.local_client import LocalLLMClient  # noqa: E402

SETTINGS = {
    "baseline": dict(temperature=0.75, top_p=0.9, top_k=20, repetition_penalty=1.05),
    "dry": dict(temperature=0.75, top_p=0.9, top_k=20, repetition_penalty=1.05, dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=2),
    "minp_dry": dict(temperature=0.75, top_p=1.0, top_k=0, min_p=0.08, repetition_penalty=1.05, dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=2),
    "rep10_dry": dict(temperature=0.75, top_p=0.9, top_k=20, repetition_penalty=1.0, dry_multiplier=0.8, dry_base=1.75, dry_allowed_length=2),
    "dry_strong": dict(temperature=0.75, top_p=0.9, top_k=20, repetition_penalty=1.05, dry_multiplier=1.2, dry_base=1.75, dry_allowed_length=2),
}
PROMPTS = [
    ("등대", "[작품 설정]\n제목: 등대지기의 딸\n장르: 미스터리 드라마\n세계관: 외딴 섬 검은돌섬. 낡은 등대와 폭풍, 섬 사람들이 숨기는 20년 전 사건\n[인물 — 이 이름들만 쓴다]\n서하린: 스물여섯 살 등대지기\n콩떡: 하린의 반려견\n박 노인: 섬의 어부\n[이번 장(1장)의 과제]\n- 폭풍이 오는 밤, 하린이 아버지의 항해일지에서 좌표를 처음 발견한다.\n- 분량: 약 1,800자. 대화와 묘사를 섞는다.\n[작성 규칙]\n- 첫 줄은 '### 소제목' 한 줄. 그 뒤는 본문 문단만.\n- 인물표에 없는 새 이름을 만들지 않는다. 한국어로만 쓴다. 마지막 문장을 완전히 끝맺는다."),
    ("연구동", "[작품 설정]\n제목: 기억의 잔향\n장르: SF 미스터리\n세계관: 기억 잔향이 물리적 흔적으로 남는 근미래 서울. 폐쇄 연구동의 실험 기록은 사라진 사람들의 마지막 선택을 보존한다.\n[인물 — 이 이름들만 쓴다]\n서윤: 동생을 찾는 기록 복원가\n민재: 실험의 진실을 숨긴 연구원\n[이번 장(1장)의 과제]\n- 서윤이 폐쇄 연구동에서 동생의 이름이 적힌 손상된 로그를 발견하고 민재와 처음 부딪힌다.\n- 분량: 약 1,800자. 대화와 묘사를 섞는다.\n[작성 규칙]\n- 첫 줄은 '### 소제목' 한 줄. 그 뒤는 본문 문단만.\n- 인물표에 없는 새 이름을 만들지 않는다. 한국어로만 쓴다. 마지막 문장을 완전히 끝맺는다."),
    ("헌책방", "[작품 설정]\n제목: 같은 페이지\n장르: 로맨스\n세계관: 비 오는 골목의 오래된 헌책방. 주인은 손님의 얼굴을 기억하지 못하지만 책은 기억한다.\n[인물 — 이 이름들만 쓴다]\n지우: 편집자\n태오: 책방 주인\n[이번 장(1장)의 과제]\n- 지우와 태오가 같은 시집을 동시에 집어 들며 처음 만난다.\n- 분량: 약 1,800자. 대화와 묘사를 섞는다.\n[작성 규칙]\n- 첫 줄은 '### 소제목' 한 줄. 그 뒤는 본문 문단만.\n- 인물표에 없는 새 이름을 만들지 않는다. 한국어로만 쓴다. 마지막 문장을 완전히 끝맺는다."),
]


def measure(text: str) -> dict:
    body = _section_body(text) if text.startswith("###") else text
    return dict(
        chars=len(text),
        rep12=_max_repeated_span(body),
        distinct3=round(_distinct3(body), 3),
        latin=len(re.findall(r"[A-Za-z]{2,}", body)),
        cjk=len(re.findall(r"[一-鿿぀-ヿ]", body)),
        ended=bool(body) and body.rstrip()[-1] in ".!?…\"'”’",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="baseline,dry,minp_dry,rep10_dry")
    ap.add_argument("--base-url", default="http://127.0.0.1:8765")
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--out", default=".runtime/bench_prose.json")
    args = ap.parse_args()
    client = LocalLLMClient(args.base_url, timeout_sec=900)
    rows = []
    for name in [s for s in args.settings.split(",") if s]:
        params = SETTINGS[name]
        for label, prompt in PROMPTS:
            t = time.time()
            text = client.chat(prompt, system=SYSTEM_PROMPT, max_tokens=args.max_tokens, korean_filter="strict", seed=11, **params)
            seconds = time.time() - t
            m = measure(text)
            m.update(setting=name, prompt=label, seconds=round(seconds, 1), text=text)
            rows.append(m)
            print(f"[{name:10s} {label:4s}] {m['chars']:5d}자 rep12={m['rep12']:2d} d3={m['distinct3']:.3f} latin={m['latin']} cjk={m['cjk']} ended={m['ended']} {seconds:.0f}s", flush=True)
    summary = {}
    for name in {r["setting"] for r in rows}:
        rs = [r for r in rows if r["setting"] == name]
        summary[name] = {k: round(sum(r[k] for r in rs) / len(rs), 3) for k in ("chars", "rep12", "distinct3", "latin", "cjk", "seconds")}
        summary[name]["ended"] = sum(r["ended"] for r in rs)
    print("\n=== 설정별 평균 ===")
    for name, m in summary.items():
        print(name, json.dumps(m, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=2), encoding="utf-8")
    print("[saved]", args.out)


if __name__ == "__main__":
    main()
