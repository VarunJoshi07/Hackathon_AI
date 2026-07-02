
"""Module for performing first-pass score fusion on candidate feature scores.

This module processes pre-computed scores (embedding, TF-IDF, skill, experience,
hiring, credibility, and penalty scores) for a high volume of candidates. It 
normalizes inputs, calculates a unified weighted score using pre-defined feature 
weights, applies penalty factors, and sorts candidates deterministically.

This module explicitly focuses *only* on score aggregation and does not handle 
embedding generation, text extraction, ranking assignment, or file I/O operations.
It is optimized to handle 100,000+ candidates efficiently.
"""

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

# Import weights and configuration constants
from src.first_pass.weights import (
    CREDIBILITY_WEIGHT,
    EMBEDDING_WEIGHT,
    EPSILON,
    EXPERIENCE_WEIGHT,
    HIRING_WEIGHT,
    MAX_SCORE,
    MIN_SCORE,
    PENALTY_WEIGHT,
    SKILL_WEIGHT,
    TFIDF_WEIGHT,
    TOP_K_FIRST_PASS,
)

# Configure logging
logger = logging.getLogger(__name__)

# Pre-defined map linking candidate keys to their respective module weights
FEATURE_WEIGHTS: Mapping[str, float] = {
    "embedding_score": EMBEDDING_WEIGHT,
    "tfidf_score": TFIDF_WEIGHT,
    "skill_score": SKILL_WEIGHT,
    "experience_score": EXPERIENCE_WEIGHT,
    "hiring_score": HIRING_WEIGHT,
    "credibility_score": CREDIBILITY_WEIGHT,
}

# Pre-compute total primary weight at module level to avoid per-candidate overhead
TOTAL_WEIGHT: float = sum(FEATURE_WEIGHTS.values())

# Validate configurations globally on startup
if TOTAL_WEIGHT <= EPSILON:
    raise ValueError("TOTAL_WEIGHT must be greater than zero.")


def normalize_score(score: Any) -> float:
    """Normalizes an individual feature score.

    Handles missing values, non-numeric types, and NaN values gracefully by
    defaulting them to MIN_SCORE. Valid numeric values are clamped strictly
    within the [MIN_SCORE, MAX_SCORE] range.

    Args:
        score: The raw score value to normalize.

    Returns:
        float: The normalized and clamped score.
    """
    if score is None:
        return MIN_SCORE

    try:
        val = float(score)
    except (ValueError, TypeError):
        logger.debug("Invalid score type encountered: %s. Defaulting to MIN_SCORE.", type(score))
        return MIN_SCORE

    if math.isnan(val) or math.isinf(val):
        return MIN_SCORE

    return max(MIN_SCORE, min(val, MAX_SCORE))


def calculate_first_pass_score(candidate_scores: Mapping[str, Any]) -> float:
    """Calculates the combined first-pass fused score for a single candidate.

    Computes the weighted average of the core feature scores dynamically using
    FEATURE_WEIGHTS, standardizes the total by the pre-computed TOTAL_WEIGHT, 
    applies a scaled penalty score, and clamps the result within system bounds.

    Args:
        candidate_scores: Dictionary containing individual candidate features.

    Returns:
        float: The final fused score.
    """
    weighted_sum = sum(
        normalize_score(candidate_scores.get(name)) * weight
        for name, weight in FEATURE_WEIGHTS.items()
    )

    weighted_base = weighted_sum / TOTAL_WEIGHT

    # Note: Assumes a 'penalties' pipeline module executed prior to score fusion.
    # If penalty_score is missing, normalize_score defaults it cleanly to MIN_SCORE.
    penalty = normalize_score(candidate_scores.get("penalty_score"))
    final_score = weighted_base - (penalty * PENALTY_WEIGHT)

    return max(MIN_SCORE, min(final_score, MAX_SCORE))


def ranking_key(candidate: Mapping[str, Any]) -> tuple[float, str]:
    """Helper function to generate a deterministic sorting key for candidates.

    Primary sort: Fused score descending (simulated via negative value).
    Secondary sort: Candidate ID ascending (alphabetical tie-breaker).
    """
    return (-candidate["final_score"], candidate["candidate_id"])


def rank_candidates(candidate_score_list: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Calculates fused scores and sorts candidates deterministically.

    Validates candidate identifiers and converts incoming records into standard
    dictionaries. Mutates the local structures to optimize memory utilization
    across large candidate batches (e.g., 100,000 items). Sorting is guaranteed 
    to be deterministic (descending by 'final_score', ascending by 'candidate_id').

    Args:
        candidate_score_list: A sequence of candidate mapping structures.

    Returns:
        list[dict[str, Any]]: The sorted list of candidate dictionaries.

    Raises:
        ValueError: If a candidate record lacks a valid string 'candidate_id'.
    """
    processed_candidates: list[dict[str, Any]] = []

    for candidate in candidate_score_list:
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str):
            raise ValueError(f"Candidate missing required string 'candidate_id'. Got: {type(candidate_id)}")

        # Convert to dictionary and calculate rounded final score
        cand_dict = dict(candidate)
        raw_final_score = calculate_first_pass_score(cand_dict)
        cand_dict["final_score"] = round(raw_final_score, 6)
        
        processed_candidates.append(cand_dict)

    # Sort in-place using the explicitly defined function key for clean execution
    processed_candidates.sort(key=ranking_key)

    return processed_candidates


def get_top_candidates(
    candidate_score_list: list[dict[str, Any]],
    top_k: int = TOP_K_FIRST_PASS,
) -> list[dict[str, Any]]:
    """Extracts the top K elements from an already ranked candidate list.

    Args:
        candidate_score_list: Ranked list of candidate dictionaries.
        top_k: Non-negative maximum number of candidates to return.

    Returns:
        list[dict[str, Any]]: The top K ranked candidate dictionaries.

    Raises:
        ValueError: If top_k is negative.
    """
    if top_k < 0:
        raise ValueError(
            f"top_k parameter must be non-negative. Received: {top_k}"
        )

    return candidate_score_list[:top_k]


def main() -> None:
    """Lightweight smoke test for score fusion processing verification."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info("Starting lightweight smoke test for score_fusion.py")

    sample_candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "CAND_002",
            "embedding_score": 0.91,
            "tfidf_score": 0.83,
            "skill_score": 0.75,
            "experience_score": 0.88,
            "hiring_score": 0.72,
            "credibility_score": 0.95,
            "penalty_score": 0.10,
        },
        {
            "candidate_id": "CAND_001",
            "embedding_score": 0.91,
            "tfidf_score": 0.83,
            "skill_score": 0.75,
            "experience_score": 0.88,
            "hiring_score": 0.72,
            "credibility_score": 0.95,
            "penalty_score": 0.10,
        },
    ]

    try:
        ranked = rank_candidates(sample_candidates)
        top_results = get_top_candidates(ranked, top_k=2)

        for rank, cand in enumerate(top_results, start=1):
            logger.info("Rank %d: %s | Final Score: %.6f", rank, cand["candidate_id"], cand["final_score"])

        assert top_results[0]["candidate_id"] == "CAND_001", "Tie-breaking failure!"
        logger.info("Smoke test completed successfully.")

    except Exception as e:
        logger.error("Smoke test failed: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
