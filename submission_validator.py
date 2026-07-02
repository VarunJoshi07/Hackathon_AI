"""Module for structural, data integrity, and compliance validation of ranked candidate arrays.

This module acts as the final audit tier before exporting results to submission.csv,
ensuring conformity to strict ordering rules, field ranges, and missing/duplicate metrics.
"""

from __future__ import annotations

import math
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

logger = logging.getLogger(__name__)

# Constants
MIN_SCORE_BOUND: Final[float] = 0.0
MAX_SCORE_BOUND: Final[float] = 1.0


@dataclass(frozen=True)
class ValidationReport:
    """Dataclass holding the metrics and findings of the submission validation pipeline process."""
    is_valid: bool
    total_candidates: int
    failed_candidates: int
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _validate_numeric_field(value: Any, field_name: str, candidate_id: str, errors: list[str]) -> float | None:
    """Helper to check if a numeric component is finite and bounded correctly between 0 and 1."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"Candidate '{candidate_id}' field '{field_name}' must be numeric.")
        return None
    val_float = float(value)
    if math.isnan(val_float) or math.isinf(val_float):
        errors.append(f"Candidate '{candidate_id}' field '{field_name}' contains non-finite value: {val_float}.")
        return None
    if not (MIN_SCORE_BOUND <= val_float <= MAX_SCORE_BOUND):
        errors.append(f"Candidate '{candidate_id}' field '{field_name}' value {val_float} out of range [0, 1].")
        return None
    return val_float


def validate_submission(candidates: Sequence[Mapping[str, Any]]) -> ValidationReport:
    """Performs full integrity, data sanity, and deterministic sorting order checks over candidates."""
    errors: list[str] = []
    warnings: list[str] = []
    total_count = len(candidates)

    if total_count == 0:
        errors.append("Candidate list is empty.")
        return ValidationReport(is_valid=False, total_candidates=0, failed_candidates=0, warnings=warnings, errors=errors)

    seen_ids: set[str] = set()
    seen_emails: set[str] = set()
    seen_phones: set[str] = set()
    seen_resumes: set[str] = set()
    
    previous_score = float("inf")
    previous_id = ""
    previous_rank = 0
    failed_candidates_count = 0

    optional_fields = ("email", "phone", "skills", "experience", "education", "metadata")

    for idx, cand in enumerate(candidates):
        if cand is None or not isinstance(cand, dict) or not cand:
            errors.append(f"Candidate record at position {idx} is empty, None, or invalid type.")
            failed_candidates_count += 1
            continue

        candidate_id = cand.get("candidate_id")
        if not candidate_id or not isinstance(candidate_id, str):
            errors.append(f"Candidate record at position {idx} missing or has a non-string 'candidate_id'.")
            failed_candidates_count += 1
            continue

        candidate_id = candidate_id.strip()
        if not candidate_id:
            errors.append(f"Candidate record at position {idx} has an empty 'candidate_id'.")
            failed_candidates_count += 1
            continue

        # Unique validation
        if candidate_id in seen_ids:
            errors.append(f"Duplicate candidate_id detected: '{candidate_id}'.")
            failed_candidates_count += 1
        seen_ids.add(candidate_id)

        # Rank validations
        rank_val = cand.get("rank")
        if rank_val is None or isinstance(rank_val, bool) or not isinstance(rank_val, int):
            errors.append(f"Candidate '{candidate_id}' has missing or non-integer 'rank'.")
        else:
            if rank_val != previous_rank + 1:
                errors.append(f"Non-consecutive rank found for candidate '{candidate_id}': expected {previous_rank + 1}, got {rank_val}.")
            previous_rank = rank_val

        # Score validations
        final_score = cand.get("final_score")
        valid_score = _validate_numeric_field(final_score, "final_score", candidate_id, errors)

        # Sort verification checks
        if valid_score is not None:
            if valid_score > previous_score:
                errors.append(f"Sorting violation at candidate '{candidate_id}': score is higher than preceding record.")
            elif math.isclose(valid_score, previous_score) and previous_id:
                if candidate_id < previous_id:
                    errors.append(f"Sorting tie-breaker violation at candidate '{candidate_id}': equal scores require ascending alphanumeric candidate_id ordering.")
            previous_score = valid_score
            previous_id = candidate_id

        # Disqualification checks
        if bool(cand.get("disqualified", False)):
            errors.append(f"Disqualified candidate '{candidate_id}' is included in the final active ranked pool.")

        # Reasoning explanation check
        reason = cand.get("ranking_reason")
        if reason is None or not isinstance(reason, str) or not reason.strip():
            errors.append(f"Candidate '{candidate_id}' missing or has empty text in 'ranking_reason'.")

        # Duplicate metadata checks (Warnings only)
        for field_key, seen_set in [("email", seen_emails), ("phone", seen_phones), ("resume_path", seen_resumes)]:
            field_val = cand.get(field_key)
            if field_val and isinstance(field_val, str):
                f_val_stripped = field_val.strip()
                if f_val_stripped:
                    if f_val_stripped in seen_set:
                        warnings.append(f"Duplicate component metadata literal '{f_val_stripped}' found for key '{field_key}' at candidate '{candidate_id}'.")
                    seen_set.add(f_val_stripped)

        # Missing field checks (Warnings only)
        for field_key in optional_fields:
            if field_key not in cand or cand[field_key] is None:
                warnings.append(f"Optional field '{field_key}' is missing or null for candidate '{candidate_id}'.")

        # Inner mathematical components validations
        for comp_score in ("cross_encoder_score", "must_have_score", "career_quality_score", "behavior_score", "honeypot_penalty"):
            if comp_score in cand:
                _validate_numeric_field(cand[comp_score], comp_score, candidate_id, errors)

    # Output structural dimension consistency
    if previous_rank != total_count:
        errors.append(f"Output count mismatch: Total elements count {total_count} does not match highest achieved rank {previous_rank}.")

    is_valid = len(errors) == 0
    return ValidationReport(
        is_valid=is_valid,
        total_candidates=total_count,
        failed_candidates=failed_candidates_count,
        warnings=warnings,
        errors=errors
    )


def raise_if_invalid(candidates: Sequence[Mapping[str, Any]]) -> None:
    """Invokes validation logic and directly raises descriptive ValueError containing errors if found."""
    report = validate_submission(candidates)
    if not report.is_valid:
        aggregated_errors = "\n".join(f"- {err}" for err in report.errors)
        raise ValueError(f"Submission Validation Failed with {len(report.errors)} violations:\n{aggregated_errors}")


def summarize_validation(report: ValidationReport) -> str:
    """Generates a clean string summary representation of a completed audit validation report process."""
    status_str = "PASSED" if report.is_valid else "FAILED"
    return (
        f"Validation Summary\n"
        f"Candidates Checked: {report.total_candidates}\n"
        f"Errors: {len(report.errors)}\n"
        f"Warnings: {len(report.warnings)}\n"
        f"Status: {status_str}"
    )


def main() -> None:
    """Main execution point providing local unit smoke validations across success and error bounds."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Executing local verification checks on submission_validator.")

    valid_mock = [
        {
            "candidate_id": "CAND_0000001",
            "rank": 1,
            "final_score": 0.95,
            "ranking_reason": "Top tier candidate matches criteria.",
            "email": "c1@test.com",
            "skills": ["Python"]
        },
        {
            "candidate_id": "CAND_0000002",
            "rank": 2,
            "final_score": 0.90,
            "ranking_reason": "Strong profile with consistent background.",
            "email": "c2@test.com",
        }
    ]

    report = validate_submission(valid_mock)
    print(summarize_validation(report))
    assert report.is_valid, "Valid mock batch scenario failed validation unexpectedly."

    invalid_mock = [
        {
            "candidate_id": "CAND_BAD_ORDER",
            "rank": 2,  # Rank starts out of order sequence
            "final_score": 0.40,
            "ranking_reason": "",  # Empty reasoning string error
            "disqualified": True,  # Disqualified record present in active list error
        },
        {
            "candidate_id": "CAND_BAD_ORDER",  # Duplicate ID error
            "rank": 2,
            "final_score": 0.85,  # Score increases relative to historical record error
            "ranking_reason": "Valid reasoning entry",
        }
    ]

    bad_report = validate_submission(invalid_mock)
    print("\n--- Testing Invalid Mock Outputs ---")
    print(summarize_validation(bad_report))
    print(f"Identified Errors Count: {len(bad_report.errors)}")
    for err in bad_report.errors:
        print(f" Detected Error -> {err}")
        
    assert not bad_report.is_valid, "Invalid mock candidate structure went completely undetected."
    logger.info("Validation runner checks evaluated successfully.")


if __name__ == "__main__":
    main()