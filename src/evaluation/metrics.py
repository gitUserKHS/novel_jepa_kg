from __future__ import annotations

import re
from collections import Counter

import numpy as np

from src.llm.ollama_client import OllamaClient


def repetition_rate(text: str, ngram: int = 4) -> float:
    tokens = re.findall(r"\w+", text)
    if len(tokens) < ngram:
        return 0.0
    grams = [tuple(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def keyword_consistency(reference: str, generated: str) -> float:
    ref_words = {word for word in re.findall(r"[가-힣A-Za-z0-9]+", reference) if len(word) >= 2}
    gen_words = {word for word in re.findall(r"[가-힣A-Za-z0-9]+", generated) if len(word) >= 2}
    if not ref_words:
        return 0.0
    return len(ref_words & gen_words) / len(ref_words)


def embedding_continuity(client: OllamaClient, reference: str, generated: str) -> float:
    vectors = client.embed([reference, generated])
    a = vectors[0]
    b = vectors[1]
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def contradiction_check(text: str) -> list[str]:
    issues = []
    patterns = [("죽었다", "살아났다"), ("모른다", "알고 있었다"), ("없었다", "있었다")]
    for left, right in patterns:
        if left in text and right in text:
            issues.append(f"'{left}'와 '{right}'가 함께 등장합니다. 맥락상 모순인지 확인하세요.")
    return issues
