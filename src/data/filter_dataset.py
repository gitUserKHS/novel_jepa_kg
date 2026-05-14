from __future__ import annotations

import json

from src.data.validate_jsonl import validate_sample
from src.utils.config import AppConfig
from src.utils.paths import ensure_parent, resolve_path

BLOCKED_NAMES = {
    "해리포터",
    "호그와트",
    "마블",
    "아이언맨",
    "스파이더맨",
    "스타워즈",
    "다스베이더",
    "나루토",
    "원피스",
}


def has_blocked_name(sample_text: str) -> bool:
    lowered = sample_text.lower()
    return any(name.lower() in lowered for name in BLOCKED_NAMES)


def is_quality_sample(sample: dict, min_summary_chars: int) -> tuple[bool, str]:
    try:
        validated = validate_sample(sample)
    except ValueError as exc:
        return False, str(exc)
    current = validated.scene_t.summary.strip()
    next_scene = validated.scene_t_plus_1.summary.strip()
    if len(current) < min_summary_chars or len(next_scene) < min_summary_chars:
        return False, "summary too short"
    if current == next_scene:
        return False, "current and next summaries are identical"
    if not validated.characters or any(not character.goal.strip() for character in validated.characters):
        return False, "missing character goal"
    if has_blocked_name(json.dumps(sample, ensure_ascii=False)):
        return False, "blocked copyrighted franchise name"
    return True, "ok"


def filter_jsonl(config: AppConfig) -> dict[str, int]:
    input_path = resolve_path(config, config.data.synthetic_path)
    output_path = resolve_path(config, config.data.filtered_path)
    ensure_parent(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {input_path}")

    kept = 0
    rejected = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                ok, _reason = is_quality_sample(sample, config.data.min_summary_chars)
            except Exception:
                ok = False
            if ok:
                dst.write(json.dumps(sample, ensure_ascii=False) + "\n")
                kept += 1
            else:
                rejected += 1
    return {"kept": kept, "rejected": rejected}
