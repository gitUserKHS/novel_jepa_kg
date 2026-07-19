from __future__ import annotations


def normalize_model_name(name: str) -> str:
    """Normalize Ollama model names while preserving explicit non-latest tags."""
    normalized = name.strip().lower()
    if normalized.endswith(":latest"):
        return normalized[: -len(":latest")]
    return normalized


def model_is_available(requested: str, available: list[str]) -> bool:
    requested_name = normalize_model_name(requested)
    return requested_name in {normalize_model_name(name) for name in available}
