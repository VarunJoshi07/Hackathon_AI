"""Module for computing a deterministic career quality metric for candidate profiles.

This module scores a candidate's professional trajectory by analyzing specific quality 
signals from the available candidate schema (years of experience, current title seniority,
company tier, skill maturity, assessment scores, and profile signals) while handling 
missing values or string-encoded JSON inputs gracefully.

The resulting scores are mapped and returned strictly bounded between 0.0 and 1.0.
"""

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

logger = logging.getLogger(__name__)

# Pre-defined list of recognizable tier-1/production-scale tech organizations
PRODUCTION_COMPANIES: Final[frozenset[str]] = frozenset({
    "google", "amazon", "apple", "netflix", "meta", "microsoft", "uber", 
    "airbnb", "stripe", "salesforce", "adobe", "atlassian", "databricks", 
    "openai", "oracle", "ibm", "intel", "cisco"
})

# Seniority title keywords and their normalized weight mappings
TITLE_KEYWORDS: Final[dict[str, float]] = {
    "intern": 0.1,
    "trainee": 0.1,
    "junior": 0.3,
    "engineer": 0.5,
    "developer": 0.5,
    "analyst": 0.5,
    "associate": 0.5,
    "senior": 0.7,
    "lead": 0.85,
    "architect": 0.85,
    "principal": 0.95,
    "staff": 0.95,
    "director": 1.0,
    "head": 1.0,
    "vp": 1.0,
    "chief": 1.0,
    "cto": 1.0
}

# Scoring Weights Configurations (Sum up to 1.0)
EXPERIENCE_WEIGHT: Final[float] = 0.25
TITLE_WEIGHT: Final[float] = 0.20
COMPANY_WEIGHT: Final[float] = 0.15
SKILL_QUALITY_WEIGHT: Final[float] = 0.20
ASSESSMENT_WEIGHT: Final[float] = 0.10
PROFILE_WEIGHT: Final[float] = 0.10


def _safe_parse_json(field_value: Any) -> Any:
    """Safely converts a potential JSON string into a structured dictionary or list."""
    if isinstance(field_value, (str, bytes)):
        try:
            return json.loads(field_value)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to decode JSON string field, defaulting to empty structure.")
            return None
    return field_value


def _calculate_single_career_quality(candidate: Mapping[str, Any]) -> float:
    """Computes the blended career quality score for a single candidate profile."""
    
    # 1. Years of Experience Score Component
    try:
        years_exp = float(candidate.get("years_experience", 0.0))
    except (ValueError, TypeError):
        years_exp = 0.0

    if years_exp < 1.0:
        exp_score = 0.10
    elif years_exp <= 2.0:
        exp_score = 0.25
    elif years_exp <= 4.0:
        exp_score = 0.45
    elif years_exp <= 6.0:
        exp_score = 0.65
    elif years_exp <= 10.0:
        exp_score = 0.85
    else:
        exp_score = 1.00

    # 2. Current Title Seniority Score Component
    current_title = str(candidate.get("current_title", "")).lower()
    title_score = 0.5  # Neutral base score if no keywords match
    matched_scores = [score for kw, score in TITLE_KEYWORDS.items() if kw in current_title]
    if matched_scores:
        title_score = max(matched_scores)

    # 3. Company Quality Component
    current_company = str(candidate.get("current_company", "")).strip().lower()
    if current_company in PRODUCTION_COMPANIES:
        company_score = 1.0
    else:
        company_score = 0.5  # Neutral value for unknown or smaller companies

    # 4. Skills Maturity Component
    skills_data = _safe_parse_json(candidate.get("skills"))
    if not isinstance(skills_data, list):
        skills_data = []

    # Gather explicitly parsed skills or fallback to candidate's stated count
    try:
        num_skills = int(candidate.get("num_skills", len(skills_data)))
    except (ValueError, TypeError):
        num_skills = len(skills_data)

    sub_skills_count = min(num_skills / 20.0, 1.0)
    
    durations = []
    endorsements = []
    proficiencies = []

    for s in skills_data:
        if isinstance(s, dict):
            # Duration parsing
            try:
                durations.append(float(s.get("duration_months", 0.0)))
            except (ValueError, TypeError):
                pass
            
            # Endorsement parsing
            try:
                endorsements.append(float(s.get("endorsements", 0.0)))
            except (ValueError, TypeError):
                pass
            
            # Proficiency parsing
            prof = str(s.get("proficiency", "")).lower()
            if "advanced" in prof:
                proficiencies.append(1.0)
            elif "intermediate" in prof:
                proficiencies.append(0.7)
            elif "beginner" in prof:
                proficiencies.append(0.3)

    sub_duration = min(np.mean(durations) / 48.0, 1.0) if durations else 0.5
    sub_endorsement = min(np.mean(endorsements) / 50.0, 1.0) if endorsements else 0.0
    sub_proficiency = np.mean(proficiencies) if proficiencies else 0.5

    skill_quality_score = (sub_skills_count + sub_duration + sub_endorsement + sub_proficiency) / 4.0

    # Parse redrob_signals object for the remaining blocks
    signals = _safe_parse_json(candidate.get("redrob_signals"))
    if not isinstance(signals, dict):
        signals = {}

    # 5. Skill Assessment Scores Component
    assessment_scores = signals.get("skill_assessment_scores")
    if isinstance(assessment_scores, dict) and assessment_scores:
        valid_assessments = []
        for val in assessment_scores.values():
            try:
                valid_assessments.append(float(val))
            except (ValueError, TypeError):
                pass
        assessment_score = (sum(valid_assessments) / (100.0 * len(valid_assessments))) if valid_assessments else 0.5
    else:
        assessment_score = 0.5  # Neutral fallback score

    # 6. Profile Quality Components
    try:
        p_completeness = float(signals.get("profile_completeness_score", 0.0)) / 100.0
    except (ValueError, TypeError):
        p_completeness = 0.0

    try:
        github_act = float(signals.get("github_activity_score", 0.0))
        p_github = min(max(github_act, 0.0) / 10.0, 1.0)
    except (ValueError, TypeError):
        p_github = 0.0

    try:
        endorsements_rec = float(signals.get("endorsements_received", 0.0))
        p_endorsements = min(endorsements_rec / 100.0, 1.0)
    except (ValueError, TypeError):
        p_endorsements = 0.0

    try:
        conn_count = float(signals.get("connection_count", 0.0))
        p_connections = min(conn_count / 500.0, 1.0)
    except (ValueError, TypeError):
        p_connections = 0.0

    p_email = 1.0 if bool(signals.get("verified_email", False)) else 0.0
    p_phone = 1.0 if bool(signals.get("verified_phone", False)) else 0.0

    profile_quality_score = (p_completeness + p_github + p_endorsements + p_connections + p_email + p_phone) / 6.0

    # 7. Weighted Blend and Clamping
    final_score = (
        (exp_score * EXPERIENCE_WEIGHT)
        + (title_score * TITLE_WEIGHT)
        + (company_score * COMPANY_WEIGHT)
        + (skill_quality_score * SKILL_QUALITY_WEIGHT)
        + (assessment_score * ASSESSMENT_WEIGHT)
        + (profile_quality_score * PROFILE_WEIGHT)
    )

    return float(np.clip(final_score, 0.0, 1.0))


def calculate_career_quality_scores(candidates: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Calculates deterministic career quality metrics for a pool of candidates.

    Args:
        candidates: A sequence of candidate record dictionaries conforming to the active schema.

    Returns:
        A NumPy array of float64 scores mapped strictly within the [0.0, 1.0] envelope.
    """
    if not candidates:
        return np.empty(0, dtype=np.float64)

    scores = [_calculate_single_career_quality(candidate) for candidate in candidates]
    return np.array(scores, dtype=np.float64)


if __name__ == "__main__":
    # Lightweight smoke test to verify schema adaptations and fallback boundaries
    logging.basicConfig(level=logging.INFO)
    logger.info("Executing updated career quality module smoke test.")

    mock_candidate_strong = {
        "candidate_id": "CAND_STRONG_01",
        "years_experience": 6.5,
        "current_title": "Senior Staff Engineer",
        "current_company": "Stripe",
        "num_skills": 12,
        "skills": [
            {"name": "Python", "duration_months": 60, "endorsements": 45, "proficiency": "advanced"}
        ],
        "redrob_signals": {
            "profile_completeness_score": 95.0,
            "connection_count": 600,
            "endorsements_received": 120,
            "github_activity_score": 8.5,
            "verified_email": True,
            "verified_phone": True,
            "skill_assessment_scores": {"Python": 92, "AWS": 85}
        }
    }

    mock_candidate_minimal = {
        "candidate_id": "CAND_MINIMAL_02",
        "years_experience": "0.5",
        "current_title": "Intern",
        "current_company": "Unknown Startup",
        "skills": "malformed_json_test_string",
        "redrob_signals": "{}"
    }

    pool = [mock_candidate_strong, mock_candidate_minimal]
    results = calculate_career_quality_scores(pool)

    print("\n--- Smoke Test Results ---")
    print(f"Strong Candidate Score: {results[0]:.4f}")
    print(f"Minimal Candidate Score: {results[1]:.4f}")
    print("--------------------------\n")

    assert results[0] > results[1], "Pipeline error: Strong candidate should outscore minimal candidate."
    assert all(0.0 <= s <= 1.0 for s in results), "Pipeline error: Scores out of bounding range [0.0, 1.0]."
    logger.info("Smoke test passed successfully.")