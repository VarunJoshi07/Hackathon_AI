"""Module for computing a deterministic career quality metric for candidate profiles.

This module scores a candidate's professional trajectory by analyzing specific quality 
signals (career progression, average tenure, promotions, top-tier/production company 
experience, leadership roles, and stable employment) while applying granular deductions 
for negative signals (job hopping, repeated internships, and inconsistent progression).

The resulting scores are mapped and returned strictly bounded between 0.0 and 1.0.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

# Constants for evaluation thresholds
MIN_STABLE_TENURE_YEARS: Final[float] = 2.0
MAX_JOB_HOPPING_TENURE_YEARS: Final[float] = 1.0
INTERNSHIP_MAX_MONTHS: Final[float] = 8.0

# Pre-defined list of recognizable tier-1/production-scale tech organizations
PRODUCTION_COMPANIES: Final[frozenset[str]] = frozenset({
    "google", "amazon", "apple", "netflix", "meta", "microsoft", "uber", 
    "airbnb", "stripe", "salesforce", "atlassian", "nvidia"
})

# Executive and organizational leadership keywords
LEADERSHIP_TERMS: Final[frozenset[str]] = frozenset({
    "lead", "principal", "staff", "manager", "director", "head", "vp", 
    "chief", "cto", "architect", "founder", "co-founder"
})


def _parse_job_history(employment_history: Any) -> list[dict[str, Any]]:
    """Normalizes and extracts a structured job history from raw inputs.
    
    Adopts a parsing schema designed to safely ingest single mappings or 
    sequences, mirroring behavior found in experience_score.py[cite: 3].
    """
    if isinstance(employment_history, Mapping):
        return [dict(employment_history)]
    if isinstance(employment_history, Sequence) and not isinstance(employment_history, str):
        return [dict(job) for job in employment_history if isinstance(job, Mapping)]
    return []


def _coerce_float(value: Any, default: float = 0.0) -> float:
    """Safely converts dynamic values to float types without throwing exceptions."""
    if value is None or isinstance(value, bool):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _analyze_single_career_profile(candidate: Mapping[str, Any]) -> float:
    """Analyzes a single candidate's history to build a unified career quality score."""
    history = _parse_job_history(candidate.get("employment_history"))
    
    # Extract structural summaries or calculate fallbacks
    total_years = _coerce_float(candidate.get("years_experience"), 0.0)
    num_jobs = len(history)
    
    # 1. Base Metrics Initializations
    tenures: list[float] = []
    promotions = 0
    production_exp = 0
    leadership_exp = 0
    internships = 0
    progression_drops = 0
    
    # Track company transitions to compute promotions and track progression
    # Assumes chronological descending order (Newest to Oldest)
    prev_company: str | None = None
    prev_level_idx: int = 99  # Smaller indicates higher tier or seniority
    
    levels_map = {"junior": 3, "mid": 2, "senior": 1, "lead": 0, "principal": 0, "staff": 0}

    for job in history:
        # Resolve individual tenure
        # Handles raw years if pre-calculated, or defaults to a standardized baseline
        job_years = _coerce_float(job.get("years_experience"), 0.0)
        if job_years <= 0.0 and num_jobs > 0:
            job_years = total_years / num_jobs if total_years > 0 else 1.0
        tenures.append(job_years)
        
        # Match company profiles
        company = str(job.get("company", "")).strip().lower()
        if company in PRODUCTION_COMPANIES:
            production_exp += 1
            
        # Match inside-company internal promotions
        if prev_company and company == prev_company:
            promotions += 1
        prev_company = company
        
        # Evaluate title mappings for leadership and hierarchical progression
        title = str(job.get("title", "")).strip().lower()
        if any(term in title for term in LEADERSHIP_TERMS):
            leadership_exp += 1
            
        # Detect repeated internships
        if "intern" in title or "internship" in title or (job_years * 12 <= INTERNSHIP_MAX_MONTHS and "intern" in title):
            internships += 1
            
        # Inconsistent progression tracking (demotions or erratic transitions)
        current_level_idx = 2  # Default to Mid level index
        for lvl_keyword, idx_val in levels_map.items():
            if lvl_keyword in title:
                current_level_idx = idx_val
                break
        
        if current_level_idx > prev_level_idx:
            progression_drops += 1
        prev_level_idx = current_level_idx

    # Calculate Average Tenure safely
    avg_tenure = np.mean(tenures) if tenures else total_years
    
    # 2. Score Component Aggregations
    quality_score = 0.5  # Start from a balanced mid-point benchmark
    
    # Positive Accelerators
    if avg_tenure >= MIN_STABLE_TENURE_YEARS:
        quality_score += 0.15
    if promotions > 0:
        quality_score += 0.10
    if production_exp > 0:
        quality_score += 0.15
    if leadership_exp > 0:
        quality_score += 0.10
    if total_years > 3.0 and progression_drops == 0:
        quality_score += 0.10  # Healthy historical progression stability
        
    # Negative Deductions / Penalties
    if avg_tenure < MAX_JOB_HOPPING_TENURE_YEARS and num_jobs >= 3:
        quality_score -= 0.25  # Job hopping penalty
    if internships >= 3:
        quality_score -= 0.15  # Repeated internships penalty
    if progression_drops >= 2:
        quality_score -= 0.15  # Inconsistent progression penalty

    return max(0.0, min(1.0, quality_score))


def calculate_career_quality_scores(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Calculates professional quality and structural scores for candidates.

    Args:
        candidates: A sequence of mapping objects representing candidate parameters.

    Returns:
        A 1D numpy.ndarray containing float64 values normalized between 0.0 and 1.0.
    """
    num_candidates = len(candidates)
    if num_candidates == 0:
        return np.empty(0, dtype=np.float64)

    logger.info("Evaluating career quality components for %d candidates...", num_candidates)
    
    # pre-allocate space to handle large pools up to 100,000 rows efficiently
    scores = np.zeros(num_candidates, dtype=np.float64)
    
    for idx, candidate in enumerate(candidates):
        scores[idx] = _analyze_single_career_profile(candidate)
        
    return scores


def main() -> None:
    """Lightweight functional smoke test to verify profile quality scoring execution."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger.info("Executing career_quality validation runner.")

    # High quality engineering profile
    candidate_strong = {
        "candidate_id": "CAND_PROGRESSIVE",
        "years_experience": 6.5,
        "employment_history": [
            {"company": "Google", "title": "Lead Backend Infrastructure Engineer", "years_experience": 2.5},
            {"company": "Google", "title": "Senior Software Engineer", "years_experience": 2.0},
            {"company": "Stripe", "title": "Software Engineer", "years_experience": 2.0}
        ]
    }

    # Unstable / High-churn hopping profile
    candidate_unstable = {
        "candidate_id": "CAND_HOPPER",
        "years_experience": 2.4,
        "employment_history": [
            {"company": "Startup A", "title": "Software Intern", "years_experience": 0.4},
            {"company": "Startup B", "title": "Junior Developer", "years_experience": 0.5},
            {"company": "Startup C", "title": "Software Engineer", "years_experience": 0.6},
            {"company": "Startup D", "title": "Developer", "years_experience": 0.5},
            {"company": "Startup E", "title": "Intern", "years_experience": 0.4}
        ]
    }

    pool = [candidate_strong, candidate_unstable]
    
    try:
        scores = calculate_career_quality_scores(pool)
        
        print("\n--- Smoke Test Score Aggregations ---")
        print(f"Strong Profile Score  : {scores[0]:.4f}")
        print(f"Unstable Profile Score: {scores[1]:.4f}")
        print("--------------------------------------\n")
        
        assert scores[0] > scores[1], "Validation error: High tier profile should score above erratic ones."
        assert 0.0 <= scores[0] <= 1.0 and 0.0 <= scores[1] <= 1.0, "Clamping constraints broken."
        logger.info("Career quality module smoke-check succeeded.")
        
    except Exception as e:
        logger.exception("Validation execution failure: %s", e)


if __name__ == "__main__":
    main()