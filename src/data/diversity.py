from __future__ import annotations

from collections import Counter
from typing import Any

from src.llm.scene_presets import AUTO_SCENE_PRESET, presets_for_genre


DIVERSITY_AXES: dict[str, list[str]] = {
    "pacing": ["느린 긴장 축적", "빠른 위기 전환", "조용한 단서 발견", "폭발 직전의 정지"],
    "pov_distance": ["주인공 밀착", "조력자 관찰", "대립자 압박", "집단 상황 조망"],
    "stakes_scale": ["개인 비밀", "관계 붕괴", "공동체 위험", "세계 규칙 흔들림"],
    "clue_type": ["물리 증거", "기억 단서", "증언/고백", "장소 변화", "기록 오류"],
    "relationship_shift": ["불신 심화", "임시 동맹", "보호와 통제 충돌", "배신 암시", "책임 전가"],
    "transition_shape": ["발견에서 선택으로", "추적에서 반전으로", "대화에서 위기로", "실패에서 새 단서로", "의심에서 검증으로"],
    "pressure_source": ["시간 제한", "감시 접근", "동료의 위험", "금기 위반", "증거 소실"],
}

REPORT_AXES = [
    "label",
    "plot_function",
    "emotion_arc",
    "pacing",
    "pov_distance",
    "stakes_scale",
    "clue_type",
    "relationship_shift",
    "transition_shape",
    "pressure_source",
]


def enhance_diversity_plan(plan: dict[str, str], sample_index: int) -> dict[str, str]:
    enhanced = dict(plan)
    for offset, (axis, values) in enumerate(DIVERSITY_AXES.items()):
        step = 2 * offset + 3
        enhanced[axis] = values[((sample_index - 1) * step + offset) % len(values)]
    enhanced["variation_id"] = f"v{sample_index:04d}"
    enhanced["diversity_signature"] = " | ".join(
        str(enhanced.get(axis, "")) for axis in ["label", "pacing", "stakes_scale", "clue_type", "transition_shape"]
    )
    return enhanced


def training_scale_recommendations(
    genre: str | None,
    preset_label: str | None,
    bucket_count: int,
) -> dict[str, Any]:
    preset_count = 1
    if not preset_label or preset_label == AUTO_SCENE_PRESET:
        preset_count = min(max(1, bucket_count), max(1, len(presets_for_genre(genre))))
    quick = max(8, preset_count * 2)
    balanced = max(32, preset_count * 8)
    research = max(96, preset_count * 18)
    robust = max(192, preset_count * 36)
    return {
        "preset_count": preset_count,
        "quick": quick,
        "balanced": balanced,
        "research": research,
        "robust": robust,
        "rationale": (
            "Quick은 UI/파이프라인 확인용, balanced는 JEPA validation 지표 관찰용, "
            "research는 장르별 전이 패턴 비교용, robust는 더 안정적인 retrieval 실험용입니다."
        ),
    }


def _plan_from_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return sample.get("metadata", {}).get("diversity_plan") or {}


def diversity_report_from_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(samples)
    plans = [_plan_from_sample(sample) for sample in samples]
    axis_reports: dict[str, dict[str, Any]] = {}
    for axis in REPORT_AXES:
        values = [str(plan.get(axis, "")).strip() for plan in plans if str(plan.get(axis, "")).strip()]
        counts = Counter(values)
        axis_reports[axis] = {
            "unique": len(counts),
            "coverage_ratio": len(counts) / max(1, total),
            "top_values": [{"value": value, "count": count} for value, count in counts.most_common(5)],
        }
    signatures = [str(plan.get("diversity_signature", "")).strip() for plan in plans]
    signature_count = len({signature for signature in signatures if signature})
    return {
        "sample_count": total,
        "unique_signatures": signature_count,
        "signature_ratio": signature_count / max(1, total),
        "axes": axis_reports,
    }
