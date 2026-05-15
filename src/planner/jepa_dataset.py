from __future__ import annotations

import random
from typing import Any

MASK_TOKEN = "[MASKED]"


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _line(label: str, value: Any) -> str:
    items = _as_list(value)
    if not items:
        return f"- {label}: (없음)"
    return f"- {label}: {' / '.join(items)}"


def _drop_value(
    field_name: str,
    value: Any,
    active: bool,
    rng: random.Random,
    field_dropout_prob: float,
) -> Any:
    if not active:
        return value
    if field_name in {"emotion", "conflict", "state", "plot_function", "relationship_tension", "scene_preset"}:
        if rng.random() < field_dropout_prob:
            return MASK_TOKEN
    return value


def _world_lines(world: dict[str, Any]) -> list[str]:
    return [
        _line("장르", world.get("genre")),
        _line("세계 전제", world.get("premise")),
        _line("세계 규칙", world.get("rules")),
    ]


def _character_lines(characters: list[dict[str, Any]]) -> list[str]:
    lines = []
    for character in characters:
        name = str(character.get("name", "인물")).strip() or "인물"
        detail = "; ".join(
            part
            for part in [
                f"역할={character.get('role', '')}".strip(),
                f"목표={character.get('goal', '')}".strip(),
                f"두려움={character.get('fear', '')}".strip(),
                f"관계={character.get('relationship', '')}".strip(),
            ]
            if part and not part.endswith("=")
        )
        lines.append(f"- {name}: {detail or '(설정 없음)'}")
    return lines or ["- 인물표 없음"]


def _scene_preset_context(metadata: dict[str, Any]) -> list[str]:
    plan = metadata.get("diversity_plan") or {}
    return [
        _line("프리셋", metadata.get("scene_preset_label") or plan.get("label")),
        _line("프리셋 기능", plan.get("plot_function")),
        _line("프리셋 모티프", plan.get("motif")),
        _line("관계 긴장", plan.get("relationship")),
    ]


def build_context_text(
    sample: dict[str, Any],
    use_dropout: bool = False,
    rng: random.Random | None = None,
    context_dropout_prob: float = 0.15,
    field_dropout_prob: float = 0.20,
) -> str:
    rng = rng or random.Random()
    active_dropout = use_dropout and rng.random() < context_dropout_prob
    world = sample.get("world", {})
    characters = sample.get("characters", [])
    scene = sample.get("scene_t", {})
    metadata = sample.get("metadata", {})
    preset_lines = _scene_preset_context(metadata)
    if active_dropout and rng.random() < field_dropout_prob:
        preset_lines = [_line("프리셋", MASK_TOKEN)]

    return "\n".join(
        [
            "[Context Encoder Input]",
            "[World]",
            *_world_lines(world),
            "",
            "[Characters]",
            *_character_lines(characters),
            "",
            "[Current Scene]",
            _line("요약", scene.get("summary")),
            _line("감정", _drop_value("emotion", scene.get("emotion"), active_dropout, rng, field_dropout_prob)),
            _line("갈등", _drop_value("conflict", scene.get("conflict"), active_dropout, rng, field_dropout_prob)),
            _line("상태", _drop_value("state", scene.get("state"), active_dropout, rng, field_dropout_prob)),
            _line(
                "장면 기능",
                _drop_value("plot_function", scene.get("plot_function"), active_dropout, rng, field_dropout_prob),
            ),
            "",
            "[Scene Preset Metadata]",
            *preset_lines,
        ]
    )


def build_target_text(sample: dict[str, Any]) -> str:
    scene = sample.get("scene_t_plus_1", {})
    metadata = sample.get("metadata", {})
    plan = metadata.get("diversity_plan") or {}
    return "\n".join(
        [
            "[Target Encoder Input]",
            _line("다음 장면 요약", scene.get("summary")),
            _line("다음 감정", scene.get("emotion")),
            _line("다음 갈등", scene.get("conflict")),
            _line("다음 상태", scene.get("state")),
            _line("다음 장면 기능", scene.get("plot_function")),
            _line("다음 훅", plan.get("next_hook")),
        ]
    )


def build_generation_context_text(
    world: str,
    characters: str,
    previous_scene: str,
    scene_preset: dict[str, str] | None = None,
) -> str:
    preset = scene_preset or {}
    return "\n".join(
        [
            "[Context Encoder Input]",
            "[World]",
            _line("세계 설정", world),
            "",
            "[Characters]",
            _line("인물표", characters),
            "",
            "[Current Scene]",
            _line("요약", previous_scene),
            "",
            "[Scene Preset Metadata]",
            _line("프리셋", preset.get("label")),
            _line("프리셋 기능", preset.get("plot_function")),
            _line("프리셋 모티프", preset.get("motif")),
            _line("관계 긴장", preset.get("relationship")),
        ]
    )


def build_context_target_texts(
    samples: list[dict[str, Any]],
    use_dropout: bool = False,
    seed: int = 42,
    context_dropout_prob: float = 0.15,
    field_dropout_prob: float = 0.20,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    contexts = [
        build_context_text(
            sample,
            use_dropout=use_dropout,
            rng=rng,
            context_dropout_prob=context_dropout_prob,
            field_dropout_prob=field_dropout_prob,
        )
        for sample in samples
    ]
    targets = [build_target_text(sample) for sample in samples]
    return contexts, targets
