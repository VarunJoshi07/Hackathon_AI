"""Experience matching utilities for candidate ranking."""

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any


logger = logging.getLogger(__name__)


DEFAULT_NO_REQUIREMENT_SCORE = 0.8
SENIORITY_BONUS = 0.05
SENIORITY_TERMS = frozenset({"senior", "lead", "principal", "staff"})
EXPERIENCE_PATTERN = re.compile(
    r"""
    (?:
        \bminimum\s+|\bat\s+least\s+|\bmin(?:imum)?\.?\s+
    )?
    (?P<years>\d{1,2})
    \s*\+?
    \s*
    (?:
        years?|yrs?
    )
    \b
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
DATE_PATTERNS = (
    re.compile(r"\b(?P<year>\d{4})-(?P<month>\d{1,2})(?:-\d{1,2})?\b"),
    re.compile(r"\b(?P<month>\d{1,2})/(?P<year>\d{4})\b"),
)
MONTH_LOOKUP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
MONTH_YEAR_PATTERN = re.compile(
    r"\b(?P<month>[a-zA-Z]{3,9})\s+(?P<year>\d{4})\b",
    flags=re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(?P<year>19\d{2}|20\d{2})\b")


@dataclass(frozen=True)
class EmploymentPeriod:
    """Normalized employment period for experience estimation."""

    start: date | None
    end: date | None


def extract_required_experience(jd_text: str | None) -> int | None:
    """Extract required years of experience from a job description."""
    if jd_text is None:
        return None

    if not isinstance(jd_text, str):
        msg = f"Expected jd_text to be str or None, got {type(jd_text).__name__}"
        logger.error(msg)
        raise TypeError(msg)

    matches = [int(match.group("years")) for match in EXPERIENCE_PATTERN.finditer(jd_text)]
    if not matches:
        logger.debug("No required experience found in job description")
        return None

    required_years = min(matches)
    logger.debug("Extracted required experience: %d years", required_years)
    return required_years


def get_candidate_experience(candidate: Mapping[str, Any]) -> float:
    """Return candidate experience in years."""
    if not isinstance(candidate, Mapping):
        msg = f"Expected candidate to be a mapping, got {type(candidate).__name__}"
        logger.error(msg)
        raise TypeError(msg)

    years_experience = _coerce_float(candidate.get("years_experience"))
    if years_experience is not None:
        return max(0.0, years_experience)

    estimated_years = _estimate_experience_from_history(
        candidate.get("employment_history"),
    )
    logger.debug("Estimated candidate experience from history: %.2f", estimated_years)
    return estimated_years


def calculate_experience_score(candidate: Mapping[str, Any], jd_text: str | None) -> float:
    """Calculate a deterministic experience match score."""
    required_experience = extract_required_experience(jd_text)
    if required_experience is None:
        return DEFAULT_NO_REQUIREMENT_SCORE

    candidate_experience = get_candidate_experience(candidate)
    if required_experience <= 0:
        base_score = 1.0
    elif candidate_experience >= required_experience:
        base_score = 1.0
    else:
        base_score = candidate_experience / required_experience

    if _has_matching_seniority(candidate, jd_text):
        base_score += SENIORITY_BONUS

    return _clamp(base_score)


def _estimate_experience_from_history(employment_history: Any) -> float:
    """Estimate total years of experience from employment history."""
    periods = _employment_periods(employment_history)
    total_months = 0

    for period in periods:
        if period.start is None or period.end is None:
            continue
        total_months += _months_between(period.start, period.end)

    return round(total_months / 12, 2)


def _employment_periods(employment_history: Any) -> list[EmploymentPeriod]:
    """Extract normalized employment periods from supported history shapes."""
    jobs = _employment_jobs(employment_history)
    periods: list[EmploymentPeriod] = []

    for job in jobs:
        start = _first_parsed_date(job, ("start_date", "start", "from"))
        end = _first_parsed_date(job, ("end_date", "end", "to"))
        if end is None and _is_current_job(job):
            end = date.today()
        periods.append(EmploymentPeriod(start=start, end=end))

    return periods


def _employment_jobs(employment_history: Any) -> list[Mapping[str, Any]]:
    """Return mapping-like employment entries from a history value."""
    if isinstance(employment_history, Mapping):
        return [employment_history]

    if isinstance(employment_history, Sequence) and not isinstance(employment_history, str):
        return [job for job in employment_history if isinstance(job, Mapping)]

    return []


def _first_parsed_date(job: Mapping[str, Any], field_names: tuple[str, ...]) -> date | None:
    """Parse the first available date-like field from a job record."""
    for field_name in field_names:
        if field_name not in job:
            continue
        parsed_date = _parse_date(job[field_name])
        if parsed_date is not None:
            return parsed_date
    return None


def _parse_date(value: Any) -> date | None:
    """Parse common employment date formats into a month-level date."""
    if value is None:
        return None

    if isinstance(value, date):
        return value

    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if not text or text in {"present", "current", "now", "ongoing"}:
        return None

    for pattern in DATE_PATTERNS:
        match = pattern.search(text)
        if match:
            return _safe_date(
                year=int(match.group("year")),
                month=int(match.group("month")),
            )

    month_year_match = MONTH_YEAR_PATTERN.search(text)
    if month_year_match:
        month = MONTH_LOOKUP.get(month_year_match.group("month").lower())
        if month is not None:
            return _safe_date(
                year=int(month_year_match.group("year")),
                month=month,
            )

    year_match = YEAR_PATTERN.search(text)
    if year_match:
        return _safe_date(year=int(year_match.group("year")), month=1)

    return None


def _safe_date(year: int, month: int) -> date | None:
    """Create a date when the year and month are valid."""
    try:
        return date(year, month, 1)
    except ValueError:
        logger.debug("Invalid date parts year=%s month=%s", year, month)
        return None


def _is_current_job(job: Mapping[str, Any]) -> bool:
    """Infer whether an employment entry represents a current role."""
    for field_name in ("is_current", "current"):
        value = job.get(field_name)
        if isinstance(value, bool):
            return value

    for field_name in ("end_date", "end", "to"):
        value = job.get(field_name)
        if isinstance(value, str) and value.strip().lower() in {
            "present",
            "current",
            "now",
            "ongoing",
        }:
            return True

    return False


def _months_between(start: date, end: date) -> int:
    """Return whole calendar months between two dates."""
    if end < start:
        return 0
    return (end.year - start.year) * 12 + end.month - start.month


def _has_matching_seniority(candidate: Mapping[str, Any], jd_text: str | None) -> bool:
    """Return whether JD and candidate title share seniority terms."""
    if not jd_text:
        return False

    jd_terms = _seniority_terms_in_text(jd_text)
    if not jd_terms:
        return False

    current_title = candidate.get("current_title")
    if not isinstance(current_title, str):
        return False

    candidate_terms = _seniority_terms_in_text(current_title)
    return bool(jd_terms.intersection(candidate_terms))


def _seniority_terms_in_text(text: str) -> set[str]:
    """Return seniority terms found in text."""
    normalized_text = text.lower()
    return {
        term
        for term in SENIORITY_TERMS
        if re.search(rf"\b{re.escape(term)}\b", normalized_text)
    }


def _coerce_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return float(value)

    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        try:
            return float(stripped_value)
        except ValueError:
            logger.debug("Unable to parse years_experience=%r", value)
            return None

    return None


def _clamp(score: float) -> float:
    """Clamp a score to the inclusive range [0.0, 1.0]."""
    return max(0.0, min(1.0, score))


def main() -> None:
    """Provide a minimal CLI smoke-check entry point."""
    logger.info("experience_score module is ready")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()