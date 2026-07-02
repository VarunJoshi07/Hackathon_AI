"""Module for computing a deterministic behavioral engagement score for candidates.

This module processes key profile engagement metrics (recruiter response rate,
interview completion rate, recent activity, open-to-work status, and notice period duration)
to generate a consolidated behavioral score.

Scoring Rules:
    - High recruiter response rate and interview completion rates scale scores positively.
    - Active or open-to-work candidates receive an engagement premium.
    - Shorter notice periods are favored (immediate/short notice maximizes the metric).
    - Outputs are safely normalized and clamped strictly within the [0.0, 1.0] range.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

# Component Weight Configurations (Must sum to 1.0 for clean normalization)
RESPONSE_RATE_WEIGHT: Final[float] = 0.25
COMPLETION_RATE_WEIGHT: Final[float] = 0.25
RECENT_ACTIVITY_WEIGHT: Final[float] = 0.20
OPEN_TO_WORK_WEIGHT: Final[float] = 0.15
NOTICE_PERIOD_WEIGHT: Final[float] = 0.15

# Global Safety Check
_TOTAL_WEIGHT: Final[float] = (
    RESPONSE_RATE_WEIGHT
    + COMPLETION_RATE_WEIGHT
    + RECENT_ACTIVITY_WEIGHT
    + OPEN_TO_WORK_WEIGHT
    + NOTICE_PERIOD_WEIGHT
)
if not np.isclose(_TOTAL_WEIGHT, 1.0):
    raise ValueError(f"System configuration weights must equal 1.0. Current total: {_TOTAL_WEIGHT}")


def _coerce_percentage_signal(value: Any) -> float:
    """Safely normalizes numeric or percentage-like signals to a [0.0, 1.0] float bounds."""
    if value is None or isinstance(value, bool):
        return 0.0
    try:
        val = float(value)
        # Handle values passed as percentages (e.g., 85.0 instead of 0.85)
        if val > 1.0:
            val = val / 100.0
        return max(0.0, min(val, 1.0))
    except (ValueError, TypeError):
        return 0.0


def _score_notice_period(notice_days: Any) -> float:
    """Scores candidate availability based on notice period days.
    
    Grading Schema:
        - 0 days (Immediate) or None: 1.0
        - <= 15 days: 0.8
        - <= 30 days: 0.6
        - <= 60 days: 0.3
        - > 60 days: 0.1
    """
    if notice_days is None or isinstance(notice_days, bool):
        return 1.0  # Safe default optimization for immediate/unspecified availability
    
    try:
        days = float(notice_days)
    except (ValueError, TypeError):
        return 0.5  # Neutral fallback for unparseable strings

    if days <= 0:
        return 1.0
    if days <= 15:
        return 0.8
    if days <= 30:
        return 0.6
    if days <= 60:
        return 0.3
    return 0.1


def _calculate_single_behavioral_score(candidate: Mapping[str, Any]) -> float:
    """Computes the weighted aggregate behavioral index for a single candidate profile."""
    # Extract and normalize continuous rate factors
    response_score = _coerce_percentage_signal(candidate.get("recruiter_response_rate"))
    completion_score = _coerce_percentage_signal(candidate.get("interview_completion_rate"))
    activity_score = _coerce_percentage_signal(candidate.get("recent_activity"))
    
    # Extract and normalize boolean flag factors
    open_to_work_flag = candidate.get("open_to_work")
    open_to_work_score = 1.0 if bool(open_to_work_flag) and not isinstance(open_to_work_flag, Sequence) else 0.0

    # Extract and normalize availability factors
    notice_score = _score_notice_period(candidate.get("notice_period"))

    # Compute explicit weighted aggregation
    final_score = (
        (response_score * RESPONSE_RATE_WEIGHT) +
        (completion_score * COMPLETION_RATE_WEIGHT) +
        (activity_score * RECENT_ACTIVITY_WEIGHT) +
        (open_to_work_score * OPEN_TO_WORK_WEIGHT) +
        (notice_score * NOTICE_PERIOD_WEIGHT)
    )
    
    return max(0.0, min(final_score, 1.0))


def calculate_behavioral_scores(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Processes candidate collections to compute standard behavioral engagement scores.

    Args:
        candidates: A sequence of candidate records containing behavioral fields.

    Returns:
        A 1D numpy.ndarray of float64 scores normalized between 0.0 and 1.0.
    """
    num_candidates = len(candidates)
    if num_candidates == 0:
        return np.empty(0, dtype=np.float64)

    logger.info("Initializing behavioral scoring pipeline for %d candidate records.", num_candidates)
    
    # Pre-allocate contiguous memory to safely handle high volumes up to 100,000+ records
    scores = np.zeros(num_candidates, dtype=np.float64)

    for idx, candidate in enumerate(candidates):
        scores[idx] = _calculate_single_behavioral_score(candidate)

    return scores


def main() -> None:
    """Lightweight system smoke test to verify behavior pipelines and weights."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.info("Starting behavioral_score validation module.")

    # High engagement profile
    candidate_active = {
        "candidate_id": "CAND_ACTIVE_01",
        "recruiter_response_rate": 0.95,
        "interview_completion_rate": 90.0,  # Test percentage auto-coercion
        "recent_activity": 0.85,
        "open_to_work": True,
        "notice_period": 0  # Immediate
    }

    # Low engagement profile
    candidate_passive = {
        "candidate_id": "CAND_PASSIVE_02",
        "recruiter_response_rate": 0.20,
        "interview_completion_rate": 0.40,
        "recent_activity": 0.10,
        "open_to_work": False,
        "notice_period": 90  # 3-month notice
    }

    sample_pool = [candidate_active, candidate_passive]

    try:
        results = calculate_behavioral_scores(sample_pool)
        
        print("\n--- Smoke Test Score Aggregations ---")
        print(f"Active Profile Score : {results[0]:.4f}")
        print(f"Passive Profile Score: {results[1]:.4f}")
        print("--------------------------------------\n")

        assert results[0] > results[1], "Pipeline logic error: Active profiles must outscore passive profiles."
        assert np.all(results >= 0.0) and np.all(results <= 1.0), "Boundary constraints broken."
        logger.info("Behavioral pipeline smoke check completed successfully.")
        
    except Exception as e:
        logger.exception("Behavioral metric pipeline failed validation: %s", e)


if __name__ == "__main__":
    main()