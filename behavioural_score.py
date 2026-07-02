"""Module for computing a deterministic behavioral engagement score for candidates.

This module processes key profile engagement metrics nested under the `redrob_signals` attribute
(handling both pre-parsed dictionaries and raw JSON strings) to generate a consolidated,
comprehensive behavioral score.

Scoring rules are vectorized across candidates, normalizing multiple production fields
into clean [0.0, 1.0] indicators, and clamping the final composite outputs safely.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, date
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

# Core Reference System Date Anchor for recency evaluations
TODAY: Final[date] = date(2026, 6, 30)


def _safe_parse_signals(candidate: Any) -> dict[str, Any]:
    """Safely extracts and parses the redrob_signals object from a candidate record.

    Guarantees returning a dictionary without raising exceptions on malformed types/strings.
    """
    if not isinstance(candidate, dict):
        return {}
    
    signals_obj = candidate.get("redrob_signals")
    if not signals_obj:
        return {}
        
    if isinstance(signals_obj, dict):
        return signals_obj
        
    if isinstance(signals_obj, str):
        try:
            parsed = json.loads(signals_obj)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            logger.debug("Failed to parse redrob_signals JSON string for a candidate record.")
            return {}
            
    return {}


def _compute_activity_score(last_active_str: Any) -> float:
    """Computes a deterministic activity metric from the last active date string.

    Last active ≤ 7 days   → 1.0
    ≤ 30 days              → 0.8
    ≤ 90 days              → 0.5
    Older                  → 0.2
    Missing/Invalid        → 0.5
    """
    if not isinstance(last_active_str, str):
        return 0.5
        
    try:
        # Standard schema date format parser
        dt = datetime.strptime(last_active_str.strip(), "%Y-%m-%d").date()
        days_diff = (TODAY - dt).days
        
        if days_diff < 0:
            return 1.0  # Handle edge cases gracefully
        if days_diff <= 7:
            return 1.0
        if days_diff <= 30:
            return 0.8
        if days_diff <= 90:
            return 0.5
        return 0.2
    except (ValueError, TypeError):
        return 0.5


def _coerce_float(val: Any, default: float = 0.0) -> float:
    """Safely coerces mixed field types to floating numbers."""
    if val is None or isinstance(val, bool):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def calculate_behavioral_scores(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Computes a deterministic composite behavioral engagement score for candidates.

    Iterates and normalizes multiple rich behavioral indicators, returning a 
    vectorized numpy array strictly bounded between 0.0 and 1.0.

    Args:
        candidates: Sequence of dictionaries representing candidate profiles.

    Returns:
        np.ndarray: Vector of final clamped behavioral floats.
    """
    if not candidates:
        return np.array([], dtype=np.float32)

    num_candidates = len(candidates)
    logger.info("Computing behavioral scores for %d profiles...", num_candidates)

    # Pre-allocate evaluation component lists
    recruiter_response_rates = []
    interview_completion_rates = []
    github_scores = []
    profile_completeness_scores = []
    offer_acceptance_rates = []
    saved_by_recruiters = []
    search_appearances = []
    profile_views = []
    connection_counts = []
    activity_scores = []
    open_to_work_flags = []
    notice_period_scores = []
    verified_emails = []
    verified_phones = []

    for candidate in candidates:
        signals = _safe_parse_signals(candidate)

        # 1. Base Direct Probability Indicators [0, 1]
        recruiter_response_rates.append(max(0.0, min(1.0, _coerce_float(signals.get("recruiter_response_rate"), 0.0))))
        interview_completion_rates.append(max(0.0, min(1.0, _coerce_float(signals.get("interview_completion_rate"), 0.0))))
        offer_acceptance_rates.append(max(0.0, min(1.0, _coerce_float(signals.get("offer_acceptance_rate"), 0.0))))

        # 2. Scaled Platform Engineering and Quality Profiles
        github_val = _coerce_float(signals.get("github_activity_score"), 0.0)
        github_scores.append(max(0.0, min(1.0, github_val / 10.0 if github_val > 0 else 0.0)))

        completeness_val = _coerce_float(signals.get("profile_completeness_score"), 0.0)
        profile_completeness_scores.append(max(0.0, min(1.0, completeness_val / 100.0)))

        # 3. Vector Volume Thresholds Cap Boundaries
        search_appearances.append(min(_coerce_float(signals.get("search_appearance_30d"), 0.0) / 300.0, 1.0))
        connection_counts.append(min(_coerce_float(signals.get("connection_count"), 0.0) / 500.0, 1.0))
        profile_views.append(min(_coerce_float(signals.get("profile_views_received_30d"), 0.0) / 50.0, 1.0))
        saved_by_recruiters.append(min(_coerce_float(signals.get("saved_by_recruiters_30d"), 0.0) / 10.0, 1.0))

        # 4. Computed Recency Decay Indicator
        activity_scores.append(_compute_activity_score(signals.get("last_active_date")))

        # 5. Availability Status Modifier Fields
        open_to_work_flags.append(1.0 if bool(signals.get("open_to_work_flag", False)) else 0.0)

        # Notice period favorability scaling: shorter notice periods score higher
        notice_days = _coerce_float(signals.get("notice_period_days"), 90.0)
        if notice_days <= 0:
            notice_score = 1.0
        elif notice_days <= 30:
            notice_score = 0.8
        elif notice_days <= 60:
            notice_score = 0.5
        elif notice_days <= 90:
            notice_score = 0.3
        else:
            notice_score = 0.1
        notice_period_scores.append(notice_score)

        # 6. Technical Authentication Flags
        verified_emails.append(1.0 if bool(signals.get("verified_email", False)) else 0.0)
        verified_phones.append(1.0 if bool(signals.get("verified_phone", False)) else 0.0)

    # Vectorize and stack metrics across all features via matrix array allocations
    feature_matrix = np.array([
        recruiter_response_rates,
        interview_completion_rates,
        offer_acceptance_rates,
        github_scores,
        profile_completeness_scores,
        search_appearances,
        connection_counts,
        profile_views,
        saved_by_recruiters,
        activity_scores,
        open_to_work_flags,
        notice_period_scores,
        verified_emails,
        verified_phones
    ], dtype=np.float32)

    # Normalized System Component Weight Definitions (Must sum strictly to 1.0)
    weights = np.array([
        0.15,  # recruiter_response_rate
        0.15,  # interview_completion_rate
        0.05,  # offer_acceptance_rate
        0.10,  # github_activity_score
        0.05,  # profile_completeness_score
        0.05,  # search_appearance_30d
        0.05,  # connection_count
        0.05,  # profile_views_received_30d
        0.05,  # saved_by_recruiters_30d
        0.10,  # activity recency metrics
        0.10,  # open_to_work_flag
        0.05,  # notice_period_days favored
        0.02,  # verified_email trust premium
        0.03,  # verified_phone trust premium
    ], dtype=np.float32)

    # Perform highly efficient vectorized dot product aggregation
    aggregated_scores = np.dot(weights, feature_matrix)

    # Safe bounded enforcement constraint clamping within [0.0, 1.0] range
    final_scores = np.clip(aggregated_scores, 0.0, 1.0)

    return final_scores


if __name__ == "__main__":
    """Lightweight smoke test execution to verify schema parsing and alignment layers."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger.info("Executing behavioral_score.py validation run...")

    # High engagement profile using direct dictionary mapping
    candidate_active = {
        "candidate_id": "CAND_ACTIVE_01",
        "redrob_signals": {
            "applications_submitted_30d": 12,
            "avg_response_time_hours": 4.5,
            "connection_count": 480,
            "endorsements_received": 95,
            "github_activity_score": 9.5,
            "interview_completion_rate": 0.95,
            "last_active_date": "2026-06-28",  # Highly Active (<= 7 days)
            "notice_period_days": 0,
            "offer_acceptance_rate": 0.90,
            "open_to_work_flag": True,
            "profile_completeness_score": 98.0,
            "profile_views_received_30d": 45,
            "recruiter_response_rate": 0.92,
            "saved_by_recruiters_30d": 9,
            "search_appearance_30d": 280,
            "verified_email": True,
            "verified_phone": True
        }
    }

    # Low engagement profile using JSON string parsing interface layout
    candidate_passive_json = json.dumps({
        "applications_submitted_30d": 0,
        "avg_response_time_hours": 120.0,
        "connection_count": 12,
        "endorsements_received": 1,
        "github_activity_score": 0.0,
        "interview_completion_rate": 0.10,
        "last_active_date": "2025-01-01",  # Dormant / Outdated (> 90 days)
        "notice_period_days": 90,
        "offer_acceptance_rate": 0.20,
        "open_to_work_flag": False,
        "profile_completeness_score": 35.0,
        "profile_views_received_30d": 1,
        "recruiter_response_rate": 0.05,
        "saved_by_recruiters_30d": 0,
        "search_appearance_30d": 4,
        "verified_email": False,
        "verified_phone": False
    })

    candidate_passive = {
        "candidate_id": "CAND_PASSIVE_02",
        "redrob_signals": candidate_passive_json
    }

    # Malformed profile payload safety simulation validation
    candidate_malformed = {
        "candidate_id": "CAND_MALFORMED_03",
        "redrob_signals": "{ corrupted json break ... }"
    }

    sample_pool = [candidate_active, candidate_passive, candidate_malformed]

    try:
        results = calculate_behavioral_scores(sample_pool)
        
        print("\n--- Smoke Test Score Aggregations ---")
        print(f"Active Profile Score     : {results[0]:.4f}")
        print(f"Passive String-JSON Score: {results[1]:.4f}")
        print(f"Malformed Payload Fallback: {results[2]:.4f}")
        print("--------------------------------------\n")

        assert results[0] > results[1], "Pipeline logic error: Active profiles must outscore passive profiles."
        assert results[2] >= 0.0 and results[2] <= 1.0, "Bounds violation discovered on fallback indices."
        logger.info("All behavioral scorer code assertions and schema integrity runs passed.")

    except Exception as e:
        logger.exception("Smoke test execution aborted due to unhandled pipeline failure.")