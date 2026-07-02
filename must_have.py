"""Module for extracting mandatory skills from a job description and scoring candidates.

This module provides high-performance, deterministic scoring of candidates based on
their possession of mandatory ("must-have") skills extracted from a job description, 
while explicitly ignoring preferred or optional skills.

Scoring Rules:
    - 1.0: Candidate possesses every single extracted mandatory skill.
    - 0.5: Candidate is missing exactly one mandatory skill.
    - 0.0: Candidate is missing two or more mandatory skills.
"""

import logging
import re
from collections.abc import Iterable, Sequence
import numpy as np

logger = logging.getLogger(__name__)

# Core section headers to identify boundaries
REQUIRED_HEADERS = re.compile(
    r"\b(must\s*have|requirements|required\s*skills|mandatory|essential)\b", 
    re.IGNORECASE
)
PREFERRED_HEADERS = re.compile(
    r"\b(preferred|nice\s*to\s*have|pluses|wishes|optional|desirable)\b", 
    re.IGNORECASE
)

# Pattern to split text into distinct list items or phrases
SKILL_SPLIT_PATTERN = re.compile(r"[•\-*–,;\n\t]+")


def extract_mandatory_skills(jd_text: str) -> set[str]:
    """Extracts mandatory skills from a job description text using rule-based parsing.

    Isolates sections containing requirement keywords and filters out sections 
    associated with preferred or optional assets.

    Args:
        jd_text: The full text of the job description.

    Returns:
        A set of normalized (lowercased, stripped) mandatory skill strings.
    """
    if not jd_text or not jd_text.strip():
        logger.warning("Empty job description text provided to skill extractor.")
        return set()

    lines = jd_text.splitlines()
    mandatory_tokens: set[str] = set()
    
    in_required_section = False

    for line in lines:
        cleaned_line = line.strip()
        if not cleaned_line:
            continue

        # Check section transitions
        if REQUIRED_HEADERS.search(cleaned_line):
            in_required_section = True
            continue
        if PREFERRED_HEADERS.search(cleaned_line):
            in_required_section = False
            continue

        # If inside a requirements block, parse out individual phrases/tokens
        if in_required_section:
            # If a line looks like a header for a completely different section, break out
            if cleaned_line.endswith(":") and not REQUIRED_HEADERS.search(cleaned_line):
                in_required_section = False
                continue

            parts = SKILL_SPLIT_PATTERN.split(cleaned_line)
            for part in parts:
                skill = part.strip().lower()
                # Filter out generic stop words or short filler text
                if skill and len(skill) > 1 and not skill.startswith(("building", "working", "experience")):
                    mandatory_tokens.add(skill)

    logger.info("Extracted %d unique mandatory skills from JD.", len(mandatory_tokens))
    return mandatory_tokens


def calculate_must_have_scores(
    jd_text: str, 
    candidate_skills: Sequence[Iterable[str]]
) -> np.ndarray:
    """Calculates deterministic must-have skill fulfillment scores for all candidates.

    Args:
        jd_text: The full text of the job description.
        candidate_skills: A sequence containing collections of skills for each candidate.

    Returns:
        A 1D numpy.ndarray of float64 scores matching the input sequence order.
    """
    num_candidates = len(candidate_skills)
    if num_candidates == 0:
        return np.empty(0, dtype=np.float64)

    mandatory_skills = extract_mandatory_skills(jd_text)

    # Edge case: If no mandatory skills are found, default everyone to full marks (1.0)
    if not mandatory_skills:
        logger.warning("No mandatory skills extracted from JD. Defaulting all scores to 1.0.")
        return np.ones(num_candidates, dtype=np.float64)

    scores = np.zeros(num_candidates, dtype=np.float64)

    # Process sequentially with local cache sets to optimize for 100,000+ records
    for idx, candidate_skill_list in enumerate(candidate_skills):
        cand_set = {str(s).strip().lower() for s in candidate_skill_list if s}
        
        # Calculate how many mandatory skills are completely missing
        missing_count = sum(1 for skill in mandatory_skills if skill not in cand_set)

        if missing_count == 0:
            scores[idx] = 1.0
        elif missing_count == 1:
            scores[idx] = 0.5
        else:
            scores[idx] = 0.0

    return scores


def main() -> None:
    """Lightweight smoke test for checking functionality and score thresholds."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("Starting must_have.py smoke test.")

    # Sample Job Description text with explicit structure
    mock_jd = """
    Job Title: Backend Infrastructure Engineer
    
    Requirements:
    - Python
    - AWS
    - PostgreSQL
    
    Preferred Skills:
    - Docker
    - Kubernetes
    """

    # Mock candidate pool simulating different degrees of qualification
    mock_candidates = [
        ["Python", "AWS", "PostgreSQL", "Docker"],  # Has everything -> 1.0
        ["python", "postgresql"],                  # Missing exactly one (AWS) -> 0.5
        ["AWS"],                                    # Missing multiple (Python, PostgreSQL) -> 0.0
        ["Docker", "Kubernetes"]                    # Has preferred only (Missing all mandatory) -> 0.0
    ]

    try:
        scores = calculate_must_have_scores(mock_jd, mock_candidates)
        
        print("\n--- Smoke Test Results ---")
        for i, score in enumerate(scores):
            print(f"Candidate {i}: Score = {score:.1f}")
        print("--------------------------\n")

        # Invariants and Assertions
        assert np.isclose(scores[0], 1.0), "Error: Full match should score 1.0"
        assert np.isclose(scores[1], 0.5), "Error: Missing one skill should score 0.5"
        assert np.isclose(scores[2], 0.0), "Error: Missing multiple skills should score 0.0"
        assert np.isclose(scores[3], 0.0), "Error: Preferred skills should not overwrite mandatory matches"
        
        logger.info("Smoke test passed successfully!")
    except Exception as e:
        logger.exception("Smoke test failed: %s", e)


if __name__ == "__main__":
    main()