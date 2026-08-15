from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from src.embedding.embed_scenes import embed_dataset
from src.embedding.vector_store import build_current_context_index, build_next_scene_index
from src.llm.ollama_client import OllamaClient
from src.planner.jepa_predict import evaluate_planner_diagnostics
from src.planner.jepa_train import train_predictor
from src.utils.config import AppConfig, ConsumerConfig
from src.utils.paths import ensure_parent, resolve_path


REQUIRED_ARTIFACT_PATHS = (
    "dataset",
    "embeddings",
    "current_index",
    "next_index",
    "checkpoint",
    "model_card",
    "history",
    "diagnostics",
)
VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")


class ActiveModelUnavailable(RuntimeError):
    pass


def _require_maintenance_idle(config: AppConfig) -> None:
    from src.service.consumer_store import ConsumerStore

    store = ConsumerStore(config)
    if store.maintenance_status() != "1":
        raise RuntimeError("Service candidate changes require active maintenance mode.")
    stats = store.queue_stats()
    if stats["queued"] or stats["running"]:
        raise RuntimeError("Wait for the consumer generation queue to become idle.")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def quality_gate(facts: dict[str, Any], thresholds: ConsumerConfig) -> dict[str, Any]:
    checks = [
        ("sample_count", int(facts.get("sample_count", 0)), ">=", thresholds.min_samples),
        ("validation_count", int(facts.get("validation_count", 0)), ">=", thresholds.min_validation_samples),
        (
            "validation_cosine",
            float(facts.get("validation_cosine", 0.0)),
            ">=",
            thresholds.min_validation_cosine,
        ),
        ("hit_at_5", float(facts.get("hit_at_5", 0.0)), ">=", thresholds.min_hit_at_5),
        (
            "normalized_top1_diversity",
            float(facts.get("normalized_top1_diversity", 0.0)),
            ">=",
            thresholds.min_top1_diversity,
        ),
        (
            "effective_rank_ratio",
            float(facts.get("effective_rank_ratio", 0.0)),
            ">=",
            thresholds.min_effective_rank_ratio,
        ),
        (
            "jepa_gain_over_rag_next",
            float(facts.get("jepa_score", 0.0)) - float(facts.get("rag_next_score", 0.0)),
            ">=",
            thresholds.min_jepa_gain_over_rag_next,
        ),
        ("file_fingerprint_match", bool(facts.get("file_fingerprint_match")), "is", True),
        ("vector_dimensions_match", bool(facts.get("vector_dimensions_match")), "is", True),
        ("model_names_match", bool(facts.get("model_names_match")), "is", True),
    ]
    results: list[dict[str, Any]] = []
    for name, actual, operator, expected in checks:
        passed = actual >= expected if operator == ">=" else actual is expected
        results.append(
            {
                "name": name,
                "actual": actual,
                "operator": operator,
                "expected": expected,
                "passed": bool(passed),
            }
        )
    failures = [row for row in results if not row["passed"]]
    return {
        "passed": not failures,
        "checks": results,
        "failures": failures,
        "recommended_sample_count": thresholds.recommended_samples,
        "recommended_sample_count_met": int(facts.get("sample_count", 0)) >= thresholds.recommended_samples,
    }


def _faiss_metadata(path: Path) -> tuple[int, int]:
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required to inspect JEPA artifacts.") from exc
    with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        shutil.copyfile(path, temporary_path)
        index = faiss.read_index(str(temporary_path))
        return int(index.d), int(index.ntotal)
    finally:
        temporary_path.unlink(missing_ok=True)


def inspect_artifact_set(
    config: AppConfig,
    paths: dict[str, str | Path],
    *,
    expected_dataset_sha256: str | None,
    expected_file_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    resolved = {key: Path(value).expanduser().resolve() for key, value in paths.items()}
    missing = [key for key in REQUIRED_ARTIFACT_PATHS[:-1] if key not in resolved or not resolved[key].exists()]
    if missing:
        return {"available": False, "reason": f"missing artifacts: {', '.join(missing)}", "missing": missing}

    dataset_sha = sha256_file(resolved["dataset"])
    with np.load(resolved["embeddings"], allow_pickle=False) as embeddings:
        current = np.asarray(embeddings["current_embeddings"], dtype="float32")
        next_vectors = np.asarray(embeddings["next_embeddings"], dtype="float32")
        embed_model = str(embeddings["embed_model"])
    checkpoint = torch.load(resolved["checkpoint"], map_location="cpu")
    current_dim, current_total = _faiss_metadata(resolved["current_index"])
    next_dim, next_total = _faiss_metadata(resolved["next_index"])
    model_card = json.loads(resolved["model_card"].read_text(encoding="utf-8"))
    diagnostics = evaluate_planner_diagnostics(_config_for_paths(config, resolved), top_k=5)
    validation = diagnostics.get("validation", {})
    baselines = validation.get("baselines", {})
    jepa = baselines.get("jepa_next_index", {})
    rag_next = baselines.get("rag_next_index", {})

    file_hashes = {
        key: sha256_file(path)
        for key, path in resolved.items()
        if key in REQUIRED_ARTIFACT_PATHS and path.exists()
    }
    hashes_match = bool(expected_file_hashes)
    if expected_file_hashes:
        hashes_match = all(file_hashes.get(key) == value for key, value in expected_file_hashes.items())
    dataset_lines = sum(1 for line in resolved["dataset"].read_text(encoding="utf-8").splitlines() if line.strip())
    sample_count = int(current.shape[0])
    facts = {
        "sample_count": sample_count,
        "validation_count": int(checkpoint.get("val_size", model_card.get("val_size", 0))),
        "validation_cosine": float(diagnostics.get("validation_pred_target_cosine", 0.0)),
        "hit_at_5": float(diagnostics.get("validation_retrieval_hit_at_k", 0.0)),
        "normalized_top1_diversity": float(
            diagnostics.get("validation_normalized_top1_diversity", 0.0)
        ),
        "effective_rank_ratio": float(
            model_card.get("representation_regularization", {}).get(
                "effective_rank_ratio",
                checkpoint.get("representation_diagnostics", {}).get("effective_rank_ratio", 0.0),
            )
        ),
        "jepa_score": float(jepa.get("retrieval_mean_score", 0.0)),
        "rag_next_score": float(rag_next.get("retrieval_mean_score", 0.0)),
        "file_fingerprint_match": bool(expected_dataset_sha256)
        and expected_dataset_sha256 == dataset_sha
        and hashes_match,
        "vector_dimensions_match": bool(
            current.ndim == 2
            and next_vectors.shape == current.shape
            and int(checkpoint.get("dim", -1)) == current.shape[1]
            and current_dim == current.shape[1]
            and next_dim == current.shape[1]
            and current_total == sample_count
            and next_total == sample_count
            and dataset_lines == sample_count
            and int(checkpoint.get("dataset_size", -1)) == sample_count
        ),
        "model_names_match": embed_model == config.consumer.embed_model,
    }
    return {
        "available": True,
        "facts": facts,
        "diagnostics": diagnostics,
        "dataset_sha256": dataset_sha,
        "file_hashes": file_hashes,
        "dimensions": {
            "embedding": int(current.shape[1]),
            "checkpoint": int(checkpoint.get("dim", -1)),
            "current_index": current_dim,
            "next_index": next_dim,
        },
        "models": {"chat": config.consumer.chat_model, "embedding": embed_model},
    }


def _config_for_paths(config: AppConfig, paths: dict[str, Path]) -> AppConfig:
    candidate = config.model_copy(deep=True)
    candidate.ollama.chat_model = candidate.consumer.chat_model
    candidate.ollama.embed_model = candidate.consumer.embed_model
    candidate.data.filtered_path = str(paths["dataset"])
    candidate.data.embeddings_path = str(paths["embeddings"])
    candidate.data.current_context_index_path = str(paths["current_index"])
    candidate.data.faiss_index_path = str(paths["next_index"])
    candidate.training.checkpoint_path = str(paths["checkpoint"])
    candidate.training.model_card_path = str(paths["model_card"])
    candidate.training.history_path = str(paths["history"])
    return candidate


def current_research_artifact_status(config: AppConfig) -> dict[str, Any]:
    paths = {
        "dataset": resolve_path(config, config.data.filtered_path),
        "embeddings": resolve_path(config, config.data.embeddings_path),
        "current_index": resolve_path(config, config.data.current_context_index_path),
        "next_index": resolve_path(config, config.data.faiss_index_path),
        "checkpoint": resolve_path(config, config.training.checkpoint_path),
        "model_card": resolve_path(config, config.training.model_card_path),
        "history": resolve_path(config, config.training.history_path),
    }
    inspected = inspect_artifact_set(
        config,
        paths,
        expected_dataset_sha256=None,
        expected_file_hashes=None,
    )
    if inspected.get("available"):
        inspected["gate"] = quality_gate(inspected["facts"], config.consumer)
    return inspected


def build_candidate(
    config: AppConfig,
    client: OllamaClient,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _require_maintenance_idle(config)
    if client.chat_model != config.consumer.chat_model or client.embed_model != config.consumer.embed_model:
        raise ValueError("Service candidates must use the fixed consumer chat and embedding models.")
    source_dataset = resolve_path(config, config.data.filtered_path)
    if not source_dataset.exists():
        raise FileNotFoundError(f"Filtered dataset not found: {source_dataset}")

    version = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    candidate_root = resolve_path(config, config.consumer.candidates_dir) / version
    candidate_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "dataset": candidate_root / "dataset.jsonl",
        "embeddings": candidate_root / "scenes.npz",
        "current_index": candidate_root / "current_context.faiss",
        "next_index": candidate_root / "next_scene.faiss",
        "checkpoint": candidate_root / "predictor.pt",
        "model_card": candidate_root / "model_card.json",
        "history": candidate_root / "train_history.json",
        "diagnostics": candidate_root / "diagnostics.json",
    }
    shutil.copy2(source_dataset, paths["dataset"])
    candidate_config = _config_for_paths(config, paths)
    candidate_config.data.embedding_cache_path = str(candidate_root / "embedding_cache.jsonl")
    candidate_config.data.reuse_existing = False
    candidate_config.ollama.chat_model = config.consumer.chat_model
    candidate_config.ollama.embed_model = config.consumer.embed_model
    if progress_callback:
        progress_callback({"stage": "embedding", "status": "running"})
    embed_dataset(candidate_config, client)
    build_current_context_index(candidate_config)
    build_next_scene_index(candidate_config)
    if progress_callback:
        progress_callback({"stage": "training", "status": "running"})
    train_predictor(candidate_config, progress_callback=progress_callback)
    diagnostics = evaluate_planner_diagnostics(candidate_config, top_k=5)
    _atomic_json(paths["diagnostics"], diagnostics)

    initial_hashes = {key: sha256_file(path) for key, path in paths.items() if path.exists()}
    inspected = inspect_artifact_set(
        candidate_config,
        paths,
        expected_dataset_sha256=initial_hashes["dataset"],
        expected_file_hashes=initial_hashes,
    )
    if not inspected.get("available"):
        raise RuntimeError(str(inspected.get("reason", "Candidate inspection failed.")))
    gate = quality_gate(inspected["facts"], config.consumer)
    manifest = {
        "schema_version": 1,
        "version": version,
        "status": "candidate",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
        "models": {
            "chat": config.consumer.chat_model,
            "embedding": config.consumer.embed_model,
        },
        "dataset_sha256": inspected["dataset_sha256"],
        "file_hashes": inspected["file_hashes"],
        "facts": inspected["facts"],
        "quality_gate": gate,
    }
    _atomic_json(candidate_root / "manifest.json", manifest)
    return manifest


def promote_candidate(config: AppConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    _require_maintenance_idle(config)
    gate = manifest.get("quality_gate", {})
    if not gate.get("passed"):
        raise ValueError("Candidate did not pass the service quality gate.")
    version = str(manifest.get("version", "")).strip()
    if not VERSION_RE.fullmatch(version):
        raise ValueError("Candidate manifest has no version.")
    candidate_root = resolve_path(config, config.consumer.candidates_dir) / version
    version_root = resolve_path(config, config.consumer.versions_dir) / version
    if not candidate_root.exists():
        raise FileNotFoundError(f"Candidate directory not found: {candidate_root}")
    models = manifest.get("models", {})
    if models.get("chat") != config.consumer.chat_model or models.get("embedding") != config.consumer.embed_model:
        raise ValueError("Candidate model names do not match the fixed consumer models.")
    candidate_root = candidate_root.resolve()
    manifest_paths = manifest.get("paths", {})
    for key in REQUIRED_ARTIFACT_PATHS:
        value = manifest_paths.get(key)
        if not value or Path(value).expanduser().resolve().parent != candidate_root:
            raise ValueError(f"Candidate artifact path is outside its version directory: {key}.")
    inspected = inspect_artifact_set(
        config,
        manifest.get("paths", {}),
        expected_dataset_sha256=manifest.get("dataset_sha256"),
        expected_file_hashes=manifest.get("file_hashes", {}),
    )
    if not inspected.get("available"):
        raise RuntimeError(str(inspected.get("reason", "Candidate inspection failed.")))
    fresh_gate = quality_gate(inspected["facts"], config.consumer)
    if not fresh_gate.get("passed"):
        raise ValueError("Candidate no longer passes the service quality gate.")
    manifest = json.loads(json.dumps(manifest))
    manifest["facts"] = inspected["facts"]
    manifest["quality_gate"] = fresh_gate
    version_root.parent.mkdir(parents=True, exist_ok=True)
    if version_root.exists():
        raise FileExistsError(f"Version already exists: {version_root}")
    os.replace(candidate_root, version_root)

    promoted = json.loads(json.dumps(manifest))
    promoted["status"] = "active"
    promoted["promoted_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    promoted["paths"] = {
        key: str((version_root / Path(value).name).resolve())
        for key, value in manifest["paths"].items()
    }
    _atomic_json(version_root / "manifest.json", promoted)
    _atomic_json(resolve_path(config, config.consumer.active_manifest_path), promoted)
    return promoted


def load_active_manifest(config: AppConfig, *, verify_files: bool = True) -> dict[str, Any]:
    path = resolve_path(config, config.consumer.active_manifest_path)
    if not path.exists():
        raise ActiveModelUnavailable("승격된 JEPA 서비스 모델이 아직 없어.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ActiveModelUnavailable("Active JEPA manifest is invalid.") from exc
    if manifest.get("status") != "active" or not manifest.get("quality_gate", {}).get("passed"):
        raise ActiveModelUnavailable("Active JEPA model has not passed the quality gate.")
    models = manifest.get("models", {})
    if models.get("chat") != config.consumer.chat_model or models.get("embedding") != config.consumer.embed_model:
        raise ActiveModelUnavailable("Active model names do not match the fixed consumer models.")
    paths = manifest.get("paths", {})
    hashes = manifest.get("file_hashes", {})
    version = str(manifest.get("version", "")).strip()
    if not VERSION_RE.fullmatch(version):
        raise ActiveModelUnavailable("Active JEPA version is invalid.")
    version_root = (resolve_path(config, config.consumer.versions_dir) / version).resolve()
    if not version_root.exists():
        raise ActiveModelUnavailable("Active JEPA version directory is missing.")
    for key in REQUIRED_ARTIFACT_PATHS:
        artifact_value = paths.get(key)
        if not artifact_value:
            raise ActiveModelUnavailable(f"Active manifest is missing paths.{key}.")
        # A manifest records the absolute path of the machine that built it, so
        # honouring it verbatim makes an artifact unusable anywhere else -- the
        # parent check below would reject every shared copy. Take only the file
        # name and rebind it to this machine's version directory, which also
        # makes escaping the directory impossible by construction.
        name = Path(str(artifact_value)).name
        if name in {"", ".", ".."}:
            raise ActiveModelUnavailable(f"Active JEPA artifact path is invalid: {key}.")
        artifact = (version_root / name).resolve()
        if artifact.parent != version_root:
            raise ActiveModelUnavailable(f"Active JEPA artifact escaped its version directory: {key}.")
        if not artifact.exists():
            raise ActiveModelUnavailable(f"Active JEPA artifact is missing: {key}.")
        if verify_files and hashes.get(key) != sha256_file(artifact):
            raise ActiveModelUnavailable(f"Active JEPA artifact fingerprint mismatch: {key}.")
    return manifest


def active_model_status(config: AppConfig, *, verify_files: bool = False) -> dict[str, Any]:
    try:
        manifest = load_active_manifest(config, verify_files=verify_files)
        return {"ready": True, "manifest": manifest, "reason": ""}
    except ActiveModelUnavailable as exc:
        return {"ready": False, "manifest": None, "reason": str(exc)}


def list_candidate_manifests(config: AppConfig) -> list[dict[str, Any]]:
    root = resolve_path(config, config.consumer.candidates_dir)
    if not root.exists():
        return []
    manifests: list[dict[str, Any]] = []
    for path in root.glob("*/manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload["manifest_path"] = str(path)
        manifests.append(payload)
    return sorted(manifests, key=lambda item: str(item.get("created_at", "")), reverse=True)
