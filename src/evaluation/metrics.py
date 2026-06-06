from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

import numpy as np

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
SENTENCE_RE = re.compile(r"[^.!?。！？\n]+[.!?。！？]?")

SECTION_HEADING_RE = re.compile(r"(?m)^#{2,3}\s+(.+?)\s*$")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def repetition_rate(text: str, ngram: int = 4) -> float:
    tokens = tokenize(text)
    if len(tokens) < ngram:
        return 0.0
    grams = [tuple(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def repetition_profile(text: str) -> dict[str, float]:
    return {f"{n}gram": repetition_rate(text, n) for n in (2, 3, 4)}


def lexical_diversity(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def sentence_stats(text: str) -> dict[str, float]:
    sentences = [sentence.strip() for sentence in SENTENCE_RE.findall(text) if sentence.strip()]
    if not sentences:
        return {"sentence_count": 0, "avg_sentence_chars": 0.0}
    return {
        "sentence_count": len(sentences),
        "avg_sentence_chars": sum(len(sentence) for sentence in sentences) / len(sentences),
    }


def section_structure_metrics(text: str, expected_sections: int = 4, min_chars_per_section: int = 450) -> dict[str, float | int]:
    matches = list(SECTION_HEADING_RE.finditer(text))
    if not matches:
        return {
            "section_count": 0,
            "section_count_fit": 0.0,
            "sections_with_body": 0,
            "section_body_coverage": 0.0,
            "avg_section_body_chars": 0.0,
        }

    body_lengths: list[int] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body_lengths.append(len(text[start:end].strip()))

    expected = max(1, expected_sections)
    min_chars = max(1, min_chars_per_section)
    sections_with_body = sum(1 for length in body_lengths if length >= min_chars)
    count_fit = 1.0 - min(1.0, abs(len(matches) - expected) / expected)
    return {
        "section_count": len(matches),
        "section_count_fit": round(max(0.0, count_fit), 4),
        "sections_with_body": sections_with_body,
        "section_body_coverage": round(sections_with_body / max(1, len(matches)), 4),
        "avg_section_body_chars": round(sum(body_lengths) / max(1, len(body_lengths)), 2),
    }


def dialogue_ratio(text: str) -> float:
    if not text:
        return 0.0
    dialogue_chars = text.count('"') + text.count("'") + text.count("“") + text.count("”") + text.count("‘") + text.count("’")
    return min(1.0, dialogue_chars / max(1, len(text)) * 20)


def keyword_consistency(reference: str, generated: str) -> float:
    ref_words = {word for word in tokenize(reference) if len(word) >= 2}
    gen_words = {word for word in tokenize(generated) if len(word) >= 2}
    if not ref_words:
        return 0.0
    return len(ref_words & gen_words) / len(ref_words)


def novelty_from_previous(reference: str, generated: str) -> float:
    ref_words = {word for word in tokenize(reference) if len(word) >= 2}
    gen_words = {word for word in tokenize(generated) if len(word) >= 2}
    if not gen_words:
        return 0.0
    overlap = len(ref_words & gen_words) / max(1, len(gen_words))
    return max(0.0, min(1.0, 1.0 - overlap))


def hallucination_metrics(
    reference: str,
    generated: str,
    *,
    target_novelty: float = 0.35,
    contradictions: list[str] | None = None,
    name_consistency_score: float = 1.0,
) -> dict[str, float | str]:
    ref_words = {word.lower() for word in tokenize(reference) if len(word) >= 2}
    gen_words = [word.lower() for word in tokenize(generated) if len(word) >= 2]
    if not gen_words:
        return {
            "supported_token_ratio": 0.0,
            "creative_expansion_rate": 0.0,
            "hallucination_presence": "none",
            "hallucination_target_alignment": 0.0,
            "useful_hallucination_score": 0.0,
            "hallucination_risk": 0.0,
        }

    novel_count = sum(1 for word in gen_words if word not in ref_words)
    creative_expansion = novel_count / max(1, len(gen_words))
    supported_ratio = 1.0 - creative_expansion
    contradiction_count = len(contradictions or [])
    contradiction_penalty = min(1.0, contradiction_count * 0.25)
    consistency_factor = max(0.0, min(1.0, name_consistency_score)) * (1.0 - contradiction_penalty)
    progression_factor = 0.5 + 0.5 * progression_score(generated)
    useful_score = creative_expansion * consistency_factor * progression_factor
    target = max(0.0, min(1.0, target_novelty))
    target_alignment = 1.0 - min(1.0, abs(creative_expansion - target))
    risk = max(0.0, creative_expansion - keyword_consistency(reference, generated)) + contradiction_penalty
    risk = max(0.0, min(1.0, risk))
    if creative_expansion < 0.15:
        presence = "low"
    elif creative_expansion < 0.35:
        presence = "moderate"
    else:
        presence = "high"
    return {
        "supported_token_ratio": round(supported_ratio, 4),
        "creative_expansion_rate": round(creative_expansion, 4),
        "hallucination_presence": presence,
        "hallucination_target_alignment": round(target_alignment, 4),
        "useful_hallucination_score": round(max(0.0, min(1.0, useful_score)), 4),
        "hallucination_risk": round(risk, 4),
    }


def length_fit(text: str, target_min_chars: int, target_max_chars: int) -> float:
    size = len(text.strip())
    if size == 0:
        return 0.0
    if target_min_chars <= size <= target_max_chars:
        return 1.0
    if size < target_min_chars:
        return size / max(1, target_min_chars)
    return max(0.0, 1.0 - ((size - target_max_chars) / max(1, target_max_chars)))


def progression_score(text: str) -> float:
    markers = [
        "단서",
        "선택",
        "결심",
        "드러",
        "밝혀",
        "변했",
        "향해",
        "위기",
        "대가",
        "관계",
        "목표",
        "진실",
    ]
    hits = sum(1 for marker in markers if marker in text)
    return min(1.0, hits / 5)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def pairwise_diversity(vectors: dict[str, np.ndarray]) -> float:
    similarities = []
    for left, right in combinations(vectors.values(), 2):
        similarities.append(cosine_similarity(left, right))
    if not similarities:
        return 0.0
    return max(0.0, min(1.0, 1.0 - float(np.mean(similarities))))


def contradiction_check(text: str) -> list[str]:
    issues = []
    patterns = [
        ("죽었다", "살아났다"),
        ("죽은", "살아 있는"),
        ("모른다", "알고 있었다"),
        ("처음 만난", "예전부터 알고"),
        ("없었다", "있었다"),
        ("떠났다", "계속 곁에"),
    ]
    for left, right in patterns:
        if left in text and right in text:
            issues.append(f"'{left}'와 '{right}'가 함께 등장합니다. 맥락상 모순인지 확인하세요.")
    return issues


def overall_score(metrics: dict[str, float | int | list | dict]) -> float:
    continuity = max(0.0, min(1.0, (float(metrics.get("embedding_continuity", 0.0)) + 1.0) / 2.0))
    keyword = float(metrics.get("keyword_consistency", 0.0))
    novelty = float(metrics.get("novelty_from_previous", 0.0))
    progression = float(metrics.get("progression_score", 0.0))
    lexical = min(1.0, float(metrics.get("lexical_diversity", 0.0)) / 0.75)
    length = float(metrics.get("length_fit", 0.0))
    consistency = float(metrics.get("name_consistency_score", 1.0))
    section_fit = float(metrics.get("section_count_fit", 0.0))
    section_body = float(metrics.get("section_body_coverage", 0.0))
    useful_hallucination = float(metrics.get("useful_hallucination_score", 0.0))
    hallucination_risk = float(metrics.get("hallucination_risk", 0.0))
    repetition = float(metrics.get("repetition_rate", 0.0))
    repeated_beats = int(metrics.get("repeated_narrative_beat_count", 0))
    repeated_subtitles = int(metrics.get("repeated_subtitle_count", 0))
    max_adjacent_similarity = float(metrics.get("max_adjacent_section_similarity", 0.0))
    contradictions = metrics.get("contradictions", [])
    contradiction_penalty = min(0.25, 0.08 * len(contradictions)) if isinstance(contradictions, list) else 0.0
    narrative_repetition_penalty = min(
        0.20,
        0.035 * repeated_beats
        + 0.025 * repeated_subtitles
        + max(0.0, max_adjacent_similarity - 0.4) * 0.2,
    )
    score = (
        0.20 * continuity
        + 0.14 * keyword
        + 0.14 * novelty
        + 0.17 * progression
        + 0.10 * lexical
        + 0.13 * length
        + 0.12 * consistency
        + 0.05 * section_body
        + 0.03 * section_fit
        + 0.08 * useful_hallucination
        - 0.25 * repetition
        - 0.08 * hallucination_risk
        - contradiction_penalty
        - narrative_repetition_penalty
    )
    return round(max(0.0, min(1.0, score)), 4)
