"""Recalibrate the JEPA plausibility threshold against the active model.

`jepa_coherence_min_cosine` is not a portable number. Cosine values depend on
the embedding model and on how well the predictor was trained, so the shipped
default is only valid for the model it was measured on. Re-run this after
changing the embedding model or promoting a retrained predictor.

The script continues a real draft twice per story state: once asking for a
proper continuation and once asking for a deliberate causal break. It then
scores both with the same gate the generator uses and reports the thresholds
that separate them.

    python scripts/calibrate_coherence.py --samples 3

Generation is slow: each sample is one full section from the local chat model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation.coherence import realized_state_text
from src.memory.story_rag import split_story_memory, story_memory_instruction
from src.planner.jepa_dataset import build_generation_context_text
from src.planner.jepa_predict import predict_from_context_embeddings
from src.service.artifacts import load_active_manifest
from src.service.runtime import make_ollama_client
from src.utils.config import AppConfig, load_config


PROMPT_HEAD = """당신은 한국어 장편 소설 작가입니다.

[세계관]
{world}

[인물표]
{characters}

[직전 장면]
{previous}

"""

GENUINE_RULES = """위 장면에서 곧바로 이어지는 다음 섹션 하나를 쓰세요.
- 직전 장면이 남긴 상황과 인물의 목표에서 인과적으로 출발할 것
- 구체적 행동과 장소 변화를 하나 포함할 것
- {flavor}
- {min_chars}자 이상, `###` 한국어 소제목 하나
"""

BROKEN_RULES = """의도적으로 앞 장면과 인과가 끊긴 섹션 하나를 쓰세요. 이것은 게이트를 보정하기 위한 결함 표본입니다.
- 직전 장면의 상황, 목표, 위치를 전혀 이어받지 말 것
- {flavor}
- 앞에서 쌓인 긴장과 단서를 언급하지 말 것
- {min_chars}자 이상, `###` 한국어 소제목 하나
"""

GENUINE_FLAVORS = [
    "새 단서를 하나 구체적인 물건으로 얻을 것",
    "인물 관계가 한 단계 변할 것",
    "적대적 압박이 먼저 움직일 것",
    "직전 선택의 대가가 물리적 사건으로 나타날 것",
]

BROKEN_FLAVORS = [
    "전혀 다른 시공간에서 무관한 사건이 일어날 것",
    "다른 장르의 이야기처럼 무관한 인물들의 일상을 그릴 것",
    "이 작품과 관계없는 회상이나 설명을 나열할 것",
    "작품의 인물이 등장하지 않는 별개의 장면을 쓸 것",
]


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else 0.0


def score_sample(
    config: AppConfig,
    client,
    *,
    world: str,
    characters: str,
    previous: str,
    predicted: np.ndarray,
    rules: str,
    flavor: str,
    min_chars: int,
) -> tuple[float, str]:
    prompt = PROMPT_HEAD.format(world=world, characters=characters, previous=previous)
    prompt += rules.format(flavor=flavor, min_chars=min_chars)
    prompt += story_memory_instruction(1)
    raw = client.chat(
        prompt,
        system="You write long-form Korean novel prose in sequential sections.",
        temperature=0.95,
        max_tokens=config.generation.max_tokens,
    )
    text, memory = split_story_memory(raw, 1)
    realized = np.asarray(
        client.embed(
            [realized_state_text(memory, text)],
            unload_chat=not config.generation.jepa_coherence_keep_chat_loaded,
        ),
        dtype="float32",
    )[0]
    title = text.splitlines()[0][:60] if text.strip() else "(empty)"
    return cosine(predicted, realized), title


def report(genuine: list[float], broken: list[float]) -> float | None:
    print()
    print(
        f"genuine n={len(genuine)} min={min(genuine):.3f} "
        f"mean={statistics.fmean(genuine):.3f} max={max(genuine):.3f}"
    )
    print(
        f"broken  n={len(broken)} min={min(broken):.3f} "
        f"mean={statistics.fmean(broken):.3f} max={max(broken):.3f}"
    )
    print(f"\n{'threshold':>10} {'false rewrites':>16} {'breaks caught':>15}")
    best: tuple[float, int] | None = None
    for step in range(41):
        threshold = round(0.50 + 0.01 * step, 3)
        false_rewrites = sum(1 for value in genuine if value < threshold)
        caught = sum(1 for value in broken if value < threshold)
        if false_rewrites == 0 and (best is None or caught > best[1]):
            best = (threshold, caught)
        if round(threshold * 100) % 2 == 0:
            print(
                f"{threshold:>10.2f} {false_rewrites:>8d}/{len(genuine):<7d} "
                f"{caught:>8d}/{len(broken):<6d}"
            )
    if best is None:
        print("\nNo threshold avoids rewriting a genuine section.")
        print("The predictor cannot separate these yet; train it on more samples.")
        return None
    print(
        f"\nRecommended jepa_coherence_min_cosine: {best[0]:.2f} "
        f"({best[1]}/{len(broken)} breaks caught, 0/{len(genuine)} genuine rewritten)"
    )
    return best[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--draft",
        default="reports/runs/creative_longform_latest.md",
        help="An existing draft to continue from. Needs at least one section.",
    )
    parser.add_argument("--world", default="", help="World text; defaults to the draft's own.")
    parser.add_argument("--characters", default="")
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Genuine and broken sections per story state (each is one chat call).",
    )
    parser.add_argument("--min-chars", type=int, default=1400)
    parser.add_argument("--out", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / args.config)
    manifest = load_active_manifest(config, verify_files=True)
    config.training.checkpoint_path = manifest["paths"]["checkpoint"]
    config.ollama.chat_model = config.consumer.chat_model
    config.ollama.embed_model = config.consumer.embed_model
    client = make_ollama_client(config)

    from src.service.story_workspace import read_draft, split_sections

    draft_path = Path(args.draft)
    if not draft_path.is_absolute():
        draft_path = root / draft_path
    sections = split_sections(read_draft(draft_path))
    if not sections:
        print(f"No draft sections found at {draft_path}.")
        print("Generate a story first, then point --draft at its draft.md.")
        return 1

    world = args.world or f"장르: {config.generation.genre or '미지정'}"
    characters = args.characters or "(인물표 미지정)"
    count = max(1, args.samples)
    print(f"active version : {manifest['version']}")
    print(f"embedding model: {config.ollama.embed_model}")
    print(f"draft sections : {len(sections)}")
    print(f"samples        : {count} genuine + {count} broken per story state\n")

    genuine: list[float] = []
    broken: list[float] = []
    records: list[dict] = []

    for section_index, previous in enumerate(sections[-2:], start=1):
        context_text = build_generation_context_text(world, characters, previous)
        predicted = predict_from_context_embeddings(
            config,
            np.asarray(
                client.embed(
                    [context_text],
                    unload_chat=not config.generation.jepa_coherence_keep_chat_loaded,
                ),
                dtype="float32",
            ),
        )[0]
        for kind, rules, flavors, bucket in (
            ("genuine", GENUINE_RULES, GENUINE_FLAVORS, genuine),
            ("broken", BROKEN_RULES, BROKEN_FLAVORS, broken),
        ):
            for index in range(count):
                score, title = score_sample(
                    config,
                    client,
                    world=world,
                    characters=characters,
                    previous=previous,
                    predicted=predicted,
                    rules=rules,
                    flavor=flavors[index % len(flavors)],
                    min_chars=args.min_chars,
                )
                bucket.append(score)
                records.append(
                    {"state": section_index, "kind": kind, "score": score, "title": title}
                )
                print(f"  state {section_index} {kind:8s} {score:.3f}  {title}")

    recommended = report(genuine, broken)
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "version": manifest["version"],
                    "embed_model": config.ollama.embed_model,
                    "records": records,
                    "recommended": recommended,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
