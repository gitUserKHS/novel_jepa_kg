"""말투 LoRA 어댑터를 Hugging Face Hub 에 올린다 (팀 공유용, 올리는 쪽에서 1회 실행).

베이스 가중치(Qwen3.5-4B, 12GB)는 올리지 않는다 — Qwen 공식 저장소에서 받으면 된다.
공유 대상은 우리가 학습한 어댑터(약 82MB)뿐이다.

    hf auth login                                       # 먼저 로그인 (토큰은 이 스크립트가 다루지 않는다)
    python scripts/share_model.py --namespace <사용자/조직> --private
    python scripts/share_model.py --namespace <사용자/조직> --private --only cute
    python scripts/share_model.py --namespace <사용자/조직> --dry-run   # 올릴 목록만 확인

받는 쪽은 `scripts/setup_model.py --namespace <같은 값>`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)

DEFAULT_SOURCE = Path(os.environ.get("NOVEL_QWEN_ADAPTER_SRC", r"C:\llm_files\adapters"))
REPOS = {"cute": "qwen3.5-4b-ko-cute", "tone": "qwen3.5-4b-ko-tone"}
# 어댑터 디렉터리에서 이것만 올린다 (체크포인트·옵티마이저 상태는 제외).
ALLOW = {"adapter_model.safetensors", "adapter_config.json", "README.md", "train_args.json"}
# 학습 데이터도 같은 저장소의 data/ 아래 둔다 (재학습·확장용). --no-data 로 끈다.
DATA_SRC = Path(os.environ.get("NOVEL_QWEN_DATA_SRC",
                               r"C:\연구_프로젝트\ai_아키텍처\applied\data"))
DATA_FILES = {"cute": "cute.jsonl", "tone": "tone.jsonl"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--namespace", default=os.environ.get("NOVEL_HF_NAMESPACE", ""),
                    help="HF 사용자명 또는 조직명")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="어댑터가 있는 로컬 디렉터리")
    ap.add_argument("--only", choices=sorted(REPOS), help="하나만 올린다")
    ap.add_argument("--private", action="store_true", help="비공개 저장소로 만든다 (팀 공유 권장)")
    ap.add_argument("--no-data", action="store_true", help="학습 데이터(jsonl)를 올리지 않는다")
    ap.add_argument("--dry-run", action="store_true", help="올리지 않고 목록만 출력")
    args = ap.parse_args()

    if not args.namespace:
        print("--namespace 가 필요합니다 (HF 사용자명 또는 조직명).", file=sys.stderr)
        raise SystemExit(1)

    from huggingface_hub import HfApi
    api = HfApi()
    if not args.dry_run:
        try:
            who = api.whoami()
            print(f"[auth] {who.get('name')} 로 로그인됨")
        except Exception:
            print("로그인이 필요합니다:  hf auth login", file=sys.stderr)
            raise SystemExit(1)

    source = Path(args.source)
    names = [args.only] if args.only else list(REPOS)
    for name in names:
        local = source / name
        if not local.is_dir():
            print(f"[skip] {name}: {local} 없음", file=sys.stderr)
            continue
        files = sorted(f for f in local.iterdir() if f.is_file() and f.name in ALLOW)
        if not any(f.name == "adapter_model.safetensors" for f in files):
            print(f"[skip] {name}: adapter_model.safetensors 없음", file=sys.stderr)
            continue
        # (로컬 경로, 저장소 내 경로) 쌍
        uploads = [(f, f.name) for f in files]
        data_file = DATA_SRC / DATA_FILES[name]
        if not args.no_data and data_file.is_file():
            uploads.append((data_file, f"data/{data_file.name}"))

        repo_id = f"{args.namespace}/{REPOS[name]}"
        size = sum(f.stat().st_size for f, _ in uploads) / 2**20
        print(f"\n[{name}] -> {repo_id} ({'private' if args.private else 'PUBLIC'}, {size:.1f} MB)")
        for f, path_in_repo in uploads:
            print(f"    {path_in_repo}  ({f.stat().st_size/2**20:.1f} MB)")
        if args.dry_run:
            continue
        api.create_repo(repo_id=repo_id, repo_type="model", private=args.private, exist_ok=True)
        for f, path_in_repo in uploads:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=path_in_repo,
                            repo_id=repo_id, repo_type="model")
        print(f"[{name}] https://huggingface.co/{repo_id}")

    if args.dry_run:
        print("\n(dry-run: 아무것도 올리지 않았습니다)")
    else:
        print(f"\n팀원 설치:  python scripts/setup_model.py --namespace {args.namespace}")


if __name__ == "__main__":
    main()
