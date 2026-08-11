"""JEPA-based plausibility check for a finished long-form section.

The planner is trained to predict the next narrative state in embedding space.
That same predictor can score a section after it is written: embed the state the
section actually reached, and compare it with the state the predictor expected
from the preceding context. A low cosine means the section is not a plausible
continuation of its own setup, which is the failure the rule-based stability
gate cannot see.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.llm.ollama_client import OllamaClient
from src.memory.story_rag import StoryMemory
from src.planner.jepa_dataset import build_generation_context_text
from src.planner.jepa_predict import predict_from_context_embeddings
from src.utils.config import AppConfig


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CoherenceAssessment:
    available: bool
    score: float
    threshold: float
    issue: str = ""

    @property
    def issues(self) -> list[str]:
        return [self.issue] if self.issue else []


UNAVAILABLE = CoherenceAssessment(available=False, score=0.0, threshold=0.0)


def realized_state_text(memory: StoryMemory, section: str) -> str:
    """Describe the state a section reached, shaped like the training target.

    Mirrors ``src.planner.jepa_dataset.build_target_text`` so the realized state
    lands in the same representation space the predictor was trained to hit.
    """
    summary = memory.summary.strip() or _body_excerpt(section)
    emotion = ", ".join(memory.keywords[:4])
    conflict = "; ".join(memory.open_clues[:3])
    state = "; ".join(
        [
            *memory.state_changes[:3],
            *(
                f"{update.entity} {update.attribute}: {update.value}"
                for update in memory.state_updates[:3]
            ),
        ]
    )
    resolved = "; ".join(memory.resolved_clues[:3])
    return "\n".join(
        [
            "[Target Encoder Input]",
            f"다음 장면 요약: {summary}",
            f"다음 감정: {emotion}",
            f"다음 갈등: {conflict}",
            f"다음 상태: {state}",
            f"다음 장면 기능: {resolved}",
            f"다음 훅: {'; '.join(memory.facts[:2])}",
        ]
    )


def assess_section_coherence(
    config: AppConfig,
    client: OllamaClient,
    *,
    world: str,
    characters: str,
    preceding_context: str,
    section: str,
    memory: StoryMemory,
) -> CoherenceAssessment:
    """Score how plausibly a section continues the context that preceded it.

    Never raises: a missing checkpoint, a cold index, or an Ollama failure
    downgrades the check to unavailable instead of losing a generated section.
    """
    if not config.generation.enable_jepa_coherence_gate:
        return UNAVAILABLE
    if not section.strip():
        return UNAVAILABLE

    threshold = float(config.generation.jepa_coherence_min_cosine)
    try:
        context_text = build_generation_context_text(world, characters, preceding_context)
        target_text = realized_state_text(memory, section)
        # One batched embed call. Evicting the multi-GB chat model for the small
        # embedder costs more in the next section's reload than the embed itself
        # (measured 8.96s vs 3.47s per section), so keep it resident by default.
        vectors = np.asarray(
            client.embed(
                [context_text, target_text],
                unload_chat=not config.generation.jepa_coherence_keep_chat_loaded,
            ),
            dtype="float32",
        )
        if vectors.shape[0] != 2:
            return UNAVAILABLE
        predicted = predict_from_context_embeddings(config, vectors[:1])[0]
        score = _cosine(predicted, vectors[1])
    except Exception:
        # Degrading silently would leave the gate permanently scoring 0.0 with
        # no trace, so record why before giving up on this section.
        logger.warning(
            "JEPA plausibility check unavailable for section %s",
            memory.section_index,
            exc_info=True,
        )
        return UNAVAILABLE

    issue = ""
    if score < threshold:
        issue = (
            f"JEPA plausibility below threshold (cosine {score:.3f} < {threshold:.3f}): "
            "the section does not follow causally from its own setup"
        )
    return CoherenceAssessment(
        available=True,
        score=round(float(score), 4),
        threshold=threshold,
        issue=issue,
    )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _body_excerpt(section: str, limit: int = 300) -> str:
    lines = [line.strip() for line in section.splitlines() if not line.strip().startswith("### ")]
    return " ".join(line for line in lines if line)[:limit]
