from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from src.data.diversity import diversity_report_from_samples
from src.data.validate_jsonl import parse_json_object, validate_sample
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import compact_plan_text, diversity_plan, synthetic_sample_prompt
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.paths import ensure_parent, resolve_path

logger = get_logger(__name__)


def _json_hash(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            sample = json.loads(line)
            key = sample.get("metadata", {}).get("dataset_key")
            if key:
                cache[key] = sample
        except json.JSONDecodeError:
            continue
    return cache


def _compatible_cache_key(genre: str, sample_id: int, plan: dict[str, Any]) -> str:
    return _json_hash(
        {
            "genre": genre,
            "sample_id": sample_id,
            "scene_preset_id": plan.get("id") or "",
            "scene_preset_label": plan.get("label") or "",
            "subgenre": plan.get("subgenre") or "",
            "plot_function": plan.get("plot_function") or "",
        }
    )


def _sample_compatible_cache_key(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata", {}) or {}
    plan = metadata.get("diversity_plan") or {}
    genre = metadata.get("genre_input") or sample.get("world", {}).get("genre", "")
    sample_id = int(sample.get("id", 0) or 0)
    merged_plan = {
        **plan,
        "id": metadata.get("scene_preset_id") or plan.get("id") or "",
        "label": metadata.get("scene_preset_label") or plan.get("label") or "",
    }
    return _compatible_cache_key(str(genre), sample_id, merged_plan)


def _build_compatible_cache(cache: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    compatible: dict[str, dict[str, Any]] = {}
    for sample in cache.values():
        key = _sample_compatible_cache_key(sample)
        compatible.setdefault(key, sample)
    return compatible


def _write_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for sample in cache.values():
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _cache_key(config: AppConfig, genre: str, sample_id: int, plan: dict[str, str]) -> str:
    return _json_hash(
        {
            "schema": "scene-transition-v4-diversity-axes",
            "chat_model": config.ollama.chat_model,
            "genre": genre,
            "sample_id": sample_id,
            "scene_preset_id": plan.get("id"),
            "scene_preset_label": plan.get("label"),
            "plan": plan,
        }
    )


def _attach_metadata(
    sample: dict[str, Any],
    key: str,
    genre: str,
    sample_id: int,
    plan: dict[str, str],
    cache_source: str = "generated",
) -> dict[str, Any]:
    sample["id"] = sample_id
    sample["metadata"] = {
        "dataset_key": key,
        "genre_input": genre,
        "genre_preset": plan.get("genre_key", ""),
        "scene_preset_id": plan.get("id", ""),
        "scene_preset_label": plan.get("label", ""),
        "diversity_plan": plan,
        "diversity_label": compact_plan_text(plan),
        "cache_source": cache_source,
    }
    return sample


def generate_synthetic_dataset(
    config: AppConfig,
    client: OllamaClient,
    genre: str,
    count: int,
    scene_preset: str | None = None,
) -> dict[str, Any]:
    output_path = resolve_path(config, config.data.synthetic_path)
    cache_path = resolve_path(config, config.data.sample_cache_path)
    ensure_parent(output_path)

    cache = _load_cache(cache_path) if config.data.reuse_existing else {}
    compatible_cache = _build_compatible_cache(cache) if config.data.reuse_existing else {}
    written = 0
    reused = 0
    exact_reused = 0
    compatible_reused = 0
    generated = 0
    failures = 0
    candidate_limit = max(count, math.ceil(count * max(1.0, config.data.synthetic_candidate_multiplier)))
    logger.info("Generating %s synthetic samples at %s with candidate limit %s", count, output_path, candidate_limit)

    selected_samples: list[dict[str, Any]] = []
    candidates_checked = 0
    for sample_id in range(1, candidate_limit + 1):
        if len(selected_samples) >= count:
            break
        candidates_checked += 1
        plan = diversity_plan(sample_id, config.data.diversity_buckets, genre=genre, preset_label=scene_preset)
        key = _cache_key(config, genre, sample_id, plan)
        cached = cache.get(key)
        cache_source = "exact"
        if cached is None and config.data.reuse_existing and config.data.allow_legacy_sample_cache:
            cached = compatible_cache.get(_compatible_cache_key(genre, sample_id, plan))
            cache_source = "legacy-compatible"
        if cached and config.data.reuse_existing:
            sample = dict(cached)
            sample = _attach_metadata(sample, key, genre, sample_id, plan, cache_source=cache_source)
            selected_samples.append(sample)
            cache[key] = sample
            reused += 1
            if cache_source == "exact":
                exact_reused += 1
            else:
                compatible_reused += 1
            written += 1
            continue

        last_error: Exception | None = None
        for _attempt in range(config.data.max_retries):
            try:
                text = client.chat(
                    synthetic_sample_prompt(genre, sample_id, plan),
                    system="당신은 한국어 서사 데이터셋을 JSON으로만 작성하는 도우미입니다.",
                    temperature=config.data.synthetic_temperature,
                    max_tokens=config.data.synthetic_max_tokens,
                    json_mode=True,
                )
                payload = parse_json_object(text)
                sample = validate_sample(payload).to_jsonable(sample_id)
                sample = _attach_metadata(sample, key, genre, sample_id, plan, cache_source="generated")
                selected_samples.append(sample)
                cache[key] = sample
                generated += 1
                written += 1
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001 - retry path records parser/model failures.
                last_error = exc
        if last_error is not None:
            failures += 1
            logger.warning("Failed to generate sample %s: %s", sample_id, last_error)

    with output_path.open("w", encoding="utf-8") as f:
        for sample in selected_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    if config.data.reuse_existing:
        _write_cache(cache_path, cache)
    return {
        "requested": count,
        "candidate_limit": candidate_limit,
        "candidates_checked": candidates_checked,
        "written": written,
        "generated": generated,
        "reused": reused,
        "exact_reused": exact_reused,
        "compatible_reused": compatible_reused,
        "failures": failures,
        "diversity": diversity_report_from_samples(selected_samples),
    }
