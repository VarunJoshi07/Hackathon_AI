"""Module for computing final second-pass score fusion across reranking signals.

This module aggregates the results of the second-pass candidate evaluation components,
combining the Cross-Encoder semantic similarity, must-have skill matches, career 
quality indicators, behavioral engagement metrics, and custom disqualifier penalties 
into a unified metric. Candidates are sorted deterministically in descending order.

Optimized to process thousands of filtered candidates with absolute safety boundaries,
proper type boundaries, and high computational efficiency.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

# System Scoring Component Weight Boundaries
CROSS_ENCODER_WEIGHT: Final[float] = 0.50
MUST_HAVE_WEIGHT: Final[float] = 0.20
CAREER_QUALITY_WEIGHT: Final[float] = 0.15
BEHAVIORAL_WEIGHT: Final[float] = 0.10
PENALTY_MULTIPLIER: Final[float] = 0.05

# System-wide normalization constants
MIN_SCORE: Final[float] = 0.0
MAX_SCORE: Final[float] = 1.0


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Safely coerces structural numeric outputs to robust float values."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def fuse_second_pass_scores(
    candidates: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Fuse all second-pass scores and return candidates sorted by final score."""

    fused_results: list[dict[str, Any]] = []

    for item in candidates:
        candidate_id = str(item.get("candidate_id", "UNKNOWN"))

        embedding_similarity = _coerce_float(
            item.get("embedding_similarity")
        )
        cross_encoder_score = _coerce_float(
            item.get("cross_encoder_score")
        )
        must_have_score = _coerce_float(
            item.get("must_have_score")
        )
        career_score = _coerce_float(
            item.get("career_quality_score")
        )
        behavior_score = _coerce_float(
            item.get("behavioral_score")
        )
        penalty = _coerce_float(
            item.get("disqualifier_penalty", item.get("penalty", 0.0))
        )

        base_score = (
            CROSS_ENCODER_WEIGHT * cross_encoder_score
            + MUST_HAVE_WEIGHT * must_have_score
            + CAREER_QUALITY_WEIGHT * career_score
            + BEHAVIORAL_WEIGHT * behavior_score
        )

        final_score = base_score - penalty * PENALTY_MULTIPLIER
        final_score = max(0.0, min(1.0, final_score))
        final_score *= 100

        fused_results.append(
            {
                "candidate_id": candidate_id,
                "final_score": round(final_score, 2),
                "matched_skills": item.get("matched_skills", []),
                "missing_skills": item.get("missing_skills", []),
                "years_experience": item.get("years_experience"),
                "embedding_similarity": round(
                    embedding_similarity, 3
                ),
                "cross_encoder_score": round(
                    cross_encoder_score, 3
                ),
                "production_background": bool(
                    item.get("production_background", False)
                ),
                "behavior_score": round(
                    behavior_score, 3
                ),
                "career_score": round(
                    career_score, 3
                ),
                "must_have_score": round(
                    must_have_score, 3
                ),
                "penalty": round(
                    penalty, 3
                ),
            }
        )

    fused_results.sort(
        key=lambda x: (-x["final_score"], x["candidate_id"])
    )

    logger.info(
        "Successfully completed score fusion and deterministic sorting for %d candidates.",
        len(fused_results),
    )

    return fused_results


def main() -> None:
    """Lightweight functional smoke test to verify alignment and tie-breaking."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.info("Starting second_pass_score_fusion validation runner.")

    # Simulated component dataset matching upstream modules
    mock_batch = [
        {
            "candidate_id": "CAND_B",
            "cross_encoder_score": 0.85,
            "must_have_score": 1.0,
            "career_quality_score": 0.80,
            "behavioral_score": 0.90,
            "disqualifier_penalty": 0.0,
        },
        {
            "candidate_id": "CAND_A",  # Identical scores to CAND_B, tests alphabetical tie-breaking
            "cross_encoder_score": 0.85,
            "must_have_score": 1.0,
            "career_quality_score": 0.80,
            "behavioral_score": 0.90,
            "disqualifier_penalty": 0.0,
        },
        {
            "candidate_id": "CAND_C",  # Impacted heavily by disqualifier deduction
            "cross_encoder_score": 0.95,
            "must_have_score": 0.5,
            "career_quality_score": 0.40,
            "behavioral_score": 0.50,
            "disqualifier_penalty": 0.80,
        }
    ]

    try:
        ranked_pool = fuse_second_pass_scores(mock_batch)

        print("\n--- Smoke Test Ranking Hierarchies ---")
        for idx, item in enumerate(ranked_pool):
            print(f"Rank {idx + 1}: {item['candidate_id']} -> Score: {item['final_score']:.5f}")
        print("----------------------------------------\n")

        # Architectural and Structural Validation Constraints
        assert ranked_pool[0]["candidate_id"] == "CAND_A", "Tie-breaking failure: Alphabetical key fallback missed."
        assert ranked_pool[1]["candidate_id"] == "CAND_B", "Sorting sequence compromised during identical scores."
        assert ranked_pool[2]["final_score"] < ranked_pool[1]["final_score"], "Penalty tracking failed validation constraints."
        assert all(MIN_SCORE <= x["final_score"] <= MAX_SCORE for x in ranked_pool), "Score clamping violation discovered."

        logger.info("Second pass fusion system check verified successfully.")
    except Exception as e:
        logger.exception("Score fusion module testing encountered an assertion failure: %s", e)


if __name__ == "__main__":
    main()