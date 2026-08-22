"""팀원용 모델 설치: 베이스 가중치(Qwen3.5-4B) + 말투 LoRA 어댑터를 내려받는다.

레포를 클론해도 모델은 따라오지 않는다(12GB). 이 스크립트가 그 간극을 메운다.

    python scripts/setup_model.py                      # 베이스 + 어댑터 전부
    python scripts/setup_model.py --skip-base          # 어댑터만 (베이스가 이미 있을 때)
    python scripts/setup_model.py --dest D:/models     # 설치 위치 지정

끝나면 `.env` 에 넣을 값을 출력한다. 비공개 어댑터 저장소라면 먼저 로그인해야 한다:

    hf auth login          # (구버전 CLI 는 huggingface-cli login)

함정 메모 (Windows 실측):
- `snapshot_download(local_dir=...)` 를 쓴다. 심링크 캐시는 개발자 모드가 아니면 권한 오류.
- 다운로드 중에는 HF_HUB_OFFLINE 를 켜면 안 된다 (서버 실행 스크립트는 켜 둔다).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 다운로드가 목적이므로 오프라인 플래그를 명시적으로 끈다 (실행 스크립트는 1 로 켠다).
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

BASE_REPO = "Qwen/Qwen3.5-4B"
BASE_DIRNAME = "qwen35-4b"
# 팀 어댑터 저장소. 다른 네임스페이스를 쓰면 --namespace 또는 NOVEL_HF_NAMESPACE 로 바꾼다.
DEFAULT_NAMESPACE = os.environ.get("NOVEL_HF_NAMESPACE", "")
ADAPTERS = {"cute": "qwen3.5-4b-ko-cute", "tone": "qwen3.5-4b-ko-tone"}


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def fetch(repo_id: str, dest: Path, *, label: str) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    print(f"\n[{label}] {repo_id} -> {dest}")
    if dest.exists() and any(dest.iterdir()):
        print(f"[{label}] already present, resuming/verifying")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),          # 심링크 캐시 회피 (Windows 권한 함정)
            ignore_patterns=["*.pt", "*.pth", "*.msgpack", "*.h5"],
        )
    except RepositoryNotFoundError:
        print(f"[{label}] NOT FOUND: {repo_id}", file=sys.stderr)
        print("        비공개 저장소라면 `hf auth login` 후 다시 실행하세요.", file=sys.stderr)
        raise SystemExit(2)
    except GatedRepoError:
        print(f"[{label}] 접근 권한이 없습니다: {repo_id} (팀 관리자에게 초대를 요청하세요)", file=sys.stderr)
        raise SystemExit(2)
    total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    print(f"[{label}] done, {human(total)}")
    return Path(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parent.parent / "models"),
                    help="설치 루트 (기본: <repo>/models)")
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE,
                    help="어댑터 저장소 네임스페이스 (HF 사용자 또는 조직)")
    ap.add_argument("--skip-base", action="store_true", help="베이스 가중치를 건너뛴다 (12GB)")
    ap.add_argument("--skip-adapters", action="store_true", help="말투 어댑터를 건너뛴다")
    args = ap.parse_args()

    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("huggingface_hub 가 없습니다:  pip install huggingface_hub", file=sys.stderr)
        raise SystemExit(1)

    dest_root = Path(args.dest).resolve()
    dest_root.mkdir(parents=True, exist_ok=True)
    env: dict[str, str] = {}

    if not args.skip_base:
        base_dir = dest_root / BASE_DIRNAME
        fetch(BASE_REPO, base_dir, label="base")
        env["NOVEL_QWEN_MODEL_DIR"] = str(base_dir)

    if not args.skip_adapters:
        if not args.namespace:
            print("\n[adapters] 네임스페이스가 없어 건너뜁니다."
                  "  --namespace <HF 사용자/조직> 또는 NOVEL_HF_NAMESPACE 를 지정하세요.", file=sys.stderr)
        else:
            specs = []
            for name, repo in ADAPTERS.items():
                target = dest_root / "adapters" / name
                fetch(f"{args.namespace}/{repo}", target, label=f"adapter:{name}")
                specs.append(f"{name}={target}")
            env["NOVEL_QWEN_ADAPTERS"] = ";".join(specs)

    if not env:
        return
    print("\n" + "=" * 72)
    print(".env 에 아래 줄을 넣으세요 (또는 환경변수로 설정):\n")
    for k, v in env.items():
        print(f"{k}={v}")
    print("\n그리고 모델 서버 파이썬 환경을 지정하세요 (torch/transformers/bitsandbytes 설치본):")
    print("NOVEL_QWEN_PYTHON=<...>\\.venv\\Scripts\\python.exe")
    print("=" * 72)
    print("\n확인:  .\\run_model_server.bat   그리고   python scripts\\health_check.py")


if __name__ == "__main__":
    main()
