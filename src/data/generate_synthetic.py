from __future__ import annotations

import json

from src.data.validate_jsonl import parse_json_object, validate_sample
from src.llm.ollama_client import OllamaClient
from src.llm.prompts import synthetic_sample_prompt
from src.utils.config import AppConfig
from src.utils.logging import get_logger
from src.utils.paths import ensure_parent, resolve_path

logger = get_logger(__name__)


def generate_synthetic_dataset(config: AppConfig, client: OllamaClient, genre: str, count: int) -> dict[str, int]:
    output_path = resolve_path(config, config.data.synthetic_path)
    ensure_parent(output_path)
    written = 0
    failures = 0
    logger.info("Generating %s synthetic samples at %s", count, output_path)
    with output_path.open("w", encoding="utf-8") as f:
        for sample_id in range(1, count + 1):
            last_error: Exception | None = None
            for _attempt in range(config.data.max_retries):
                try:
                    text = client.chat(
                        synthetic_sample_prompt(genre, sample_id),
                        system="당신은 한국어 서사 데이터셋을 JSON으로만 작성하는 도우미입니다.",
                        temperature=0.9,
                        max_tokens=1200,
                    )
                    payload = parse_json_object(text)
                    sample = validate_sample(payload).to_jsonable(sample_id)
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1
                    last_error = None
                    break
                except Exception as exc:  # noqa: BLE001 - retry path records every parser/model failure.
                    last_error = exc
            if last_error is not None:
                failures += 1
                logger.warning("Failed to generate sample %s: %s", sample_id, last_error)
    return {"requested": count, "written": written, "failures": failures}
