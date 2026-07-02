"""Candidate profile verification and compliance validation pipeline module.

This module provides high-performance, deterministic verification functions to audit
and detect fraud, irregularities, or inflated metrics within candidate resumes. 
It screens for keyword stuffing, overlapping timelines, duplicate roles, fabricated
experience durations, and skill/title contradictions.

The module computes a continuous `validation_score` strictly bounded within [0.0, 1.0].
A score of 1.0 represents a fully compliant and authentic profile, with fractional
deductions applied for each anomaly discovered.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

# System-wide timeline anchor
TODAY: Final[date] = date(2026, 6, 30)

# Fractional penalty configuration constraints
PENALTY_KEYWORD_STUFFING: Final[float] = 0.25
PENALTY_OVERLAPPING_DATES: Final[float] = 0.20
PENALTY_IMPOSSIBLE_TIMELINE: Final[float] = 0.30
PENALTY_DUPLICATE_EXP: Final[float] = 0.15
PENALTY_FAKE_YEARS: Final[float] = 0.25
PENALTY_INCONSISTENT_SKILLS: Final[float] = 0.15

# Configuration parameters
KEYWORD_STUFFING_MAX_COUNT: Final[int] = 6
MIN_ALLOWED_BIRTH_YEAR: Final[int] = 1940


def _parse_date(value: Any) -> date | None:
    """Safely converts dynamic timeline entries into date structures."""
    if not value or isinstance(value, bool):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (ValueError, TypeError):
        return None


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Safely normalizes dynamic numerical parameters to precision floats."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def verify_single_candidate(candidate: Mapping[str, Any]) -> float:
    """Audits a single candidate record against compliance and integrity metrics.

    Args:
        candidate: Mapping object containing professional background attributes.

    Returns:
        float: Calculated validation score strictly bounded between 0.0 and 1.0.
    """
    validation_score = 1.0

    # Extract foundational structures cleanly
    history = candidate.get("career_history") or candidate.get("employment_history") or []
    if not isinstance(history, list):
        history = []

    skills_raw = candidate.get("skills") or []
    skills_set: set[str] = set()
    if isinstance(skills_raw, list):
        for s in skills_raw:
            if isinstance(s, dict) and "name" in s:
                skills_set.add(str(s["name"]).strip().lower())
            elif isinstance(s, str):
                skills_set.add(s.strip().lower())
    elif isinstance(skills_raw, set):
        skills_set = {str(s).strip().lower() for s in skills_raw if s}

    # 1. DETECT: Keyword Stuffing
    profile_summary = str(candidate.get("profile", {}).get("summary", "") or candidate.get("summary", "")).lower()
    all_text_context = profile_summary + " " + " ".join(
        [str(job.get("description", "")).lower() for job in history if isinstance(job, dict)]
    )
    
    # Track repetitions of distinct vocabulary phrases
    unique_tokens = set(all_text_context.split())
    for token in unique_tokens:
        if len(token) > 3 and all_text_context.count(token) >= KEYWORD_STUFFING_MAX_COUNT:
            # Skip common linguistic functional components or structural syntax markers
            if token in {"with", "from", "that", "this", "development", "software", "engineer", "management"}:
                continue
            validation_score -= PENALTY_KEYWORD_STUFFING
            break

    # 2. DETECT: Overlapping Employment Dates & Impossible Timelines
    parsed_intervals: list[tuple[date, date]] = []
    has_impossible_timeline = False
    
    for job in history:
        if not isinstance(job, dict):
            continue
        start_dt = _parse_date(job.get("start_date"))
        end_dt = _parse_date(job.get("end_date")) or TODAY

        if start_dt and end_dt:
            if start_dt > end_dt or start_dt > TODAY:
                has_impossible_timeline = True
            else:
                parsed_intervals.append((start_dt, end_dt))

    # Audit biological impossible limits if age parameters are visible
    dob = _parse_date(candidate.get("dob") or candidate.get("date_of_birth"))
    if dob and (dob.year < MIN_ALLOWED_BIRTH_YEAR or dob >= TODAY):
        has_impossible_timeline = True

    if has_impossible_timeline:
        validation_score -= PENALTY_IMPOSSIBLE_TIMELINE

    # Vectorized sorting check to surface chronological date overlap anomalies
    if len(parsed_intervals) > 1:
        # Sort intervals primarily based on ascending starting timelines
        parsed_intervals.sort(key=lambda x: x[0])
        for idx in range(len(parsed_intervals) - 1):
            # Check if current job entry starts prior to the previous job concluding
            if parsed_intervals[idx + 1][0] < parsed_intervals[idx][1]:
                # Dynamic buffer allowance for minor cross-over or contract handover periods (e.g., 15 days)
                if (parsed_intervals[idx][1] - parsed_intervals[idx + 1][0]).days > 15:
                    validation_score -= PENALTY_OVERLAPPING_DATES
                    break

    # 3. DETECT: Duplicate Experience Block Identification
    unique_roles_signature: set[tuple[str, str]] = set()
    for job in history:
        if not isinstance(job, dict):
            continue
        company = str(job.get("company", "")).strip().lower()
        title = str(job.get("title", "")).strip().lower()
        if company and title:
            signature = (company, title)
            if signature in unique_roles_signature:
                validation_score -= PENALTY_DUPLICATE_EXP
                break
            unique_roles_signature.add(signature)

    # 4. DETECT: Fake / Fabricated Years of Experience Inflation
    stated_years = _coerce_float(candidate.get("years_experience") or candidate.get("total_experience"), 0.0)
    calculated_months = 0.0
    for start_dt, end_dt in parsed_intervals:
        calculated_months += (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
    
    calculated_years = calculated_months / 12.0
    # Flag profiles where declared experience exceeds computed timeline by over 2.5 years
    if stated_years > 0.0 and (stated_years - calculated_years) > 2.5:
        validation_score -= PENALTY_FAKE_YEARS

    # 5. DETECT: Inconsistent / Contradictory Skills Mapping
    # Flag instances declaring expert ML domains while missing foundational prerequisites
    if any(x in skills_set for x in {"deep learning", "transformers", "llm", "rag"}):
        if not (skills_set & {"python", "pytorch", "tensorflow", "scikit-learn", "machine learning", "ml"}):
            validation_score -= PENALTY_INCONSISTENT_SKILLS

    return max(0.0, min(validation_score, 1.0))


def calculate_validation_scores(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Processes top candidates to return a contiguous vector of validation scores.

    Optimized to support high-performance scaling utilizing pre-allocated
    numpy memory layouts.

    Args:
        candidates: Sequence of dynamic candidate profile objects.

    Returns:
        np.ndarray: A 1D numpy array containing calculated float64 validation scores.
    """
    num_candidates = len(candidates)
    if num_candidates == 0:
        return np.empty(0, dtype=np.float64)

    logger.info("Initializing integrity verification pipeline for %d profiles.", num_candidates)
    scores = np.zeros(num_candidates, dtype=np.float64)

    for idx, candidate in enumerate(candidates):
        scores[idx] = verify_single_candidate(candidate)

    return scores


def main() -> None:
    """Lightweight module smoke test to verify component evaluation thresholds."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Executing verification pipeline validation test run.")

    # High risk suspicious profile testing combined penalties
    fraudulent_candidate = {
        "candidate_id": "CAND_FRAUD",
        "years_experience": 15.0,  # Overstated inflating delta
        "summary": "AI AI AI AI AI AI AI AI AI systems engineer generation framework",  # Keyword stuffed
        "skills": ["llm", "transformers"],  # Contradictory pattern missing Python/ML foundational stack
        "career_history": [
            {
                "company": "Enterprise Inc",
                "title": "Staff Engineer",
                "start_date": "2024-01-01",
                "end_date": "2026-01-01",
                "description": "Building scaling platforms"
            },
            {
                "company": "Enterprise Inc",
                "title": "Staff Engineer",  # Identical copy duplicate experience signature
                "start_date": "2024-06-01",  # Overlapping timeline sequence anomaly
                "end_date": "2025-12-01",
                "description": "Building scaling platforms"
            }
        ]
    }

    compliant_candidate = {
        "candidate_id": "CAND_VALID",
        "years_experience": 4.0,
        "summary": "Experienced Full Stack engineer specialized in enterprise platform components.",
        "skills": ["python", "pytorch", "llm", "postgresql"],
        "career_history": [
            {
                "company": "Stripe",
                "title": "Software Engineer II",
                "start_date": "2022-06-01",
                "end_date": "2026-06-01"
            }
        ]
    }

    mock_pool = [fraudulent_candidate, compliant_candidate]

    try:
        scores = calculate_validation_scores(mock_pool)
        
        print("\n--- Integrity Check Results ---")
        print(f"Suspicious Profile Score: {scores[0]:.4f}")
        print(f"Compliant Profile Score : {scores[1]:.4f}")
        print("--------------------------------\n")

        assert scores[0] < 0.5, "Error: Verification logic failed to penalize suspicious attributes."
        assert np.isclose(scores[1], 1.0), "Error: Compliant profiles should score near perfect parity bounds."
        logger.info("Validation infrastructure verification run passed execution targets.")
        
    except Exception as e:
        logger.exception("Validation routine crashed during testing execution: %s", e)


if __name__ == "__main__":
    main()