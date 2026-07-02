"""Build credibility metadata for candidate records without assigning penalties."""

import logging
import re
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.offline.build_skill_vocab import iter_candidates, normalize_skill
from src.offline.feature_extractor import extract_features


logger = logging.getLogger(__name__)


DEFAULT_DATA_PATH = Path("data") / "candidates.jsonl.gz"
DEFAULT_OUTPUT_PATH = Path("artifacts") / "penalty_metadata.parquet"
CURRENT_DATE = date.today()
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


@dataclass(frozen=True)
class EmploymentPeriod:
    """Normalized employment period used for timeline diagnostics."""

    start: date | None
    end: date | None
    is_current: bool


def _iter_skill_values(skills: Any) -> Iterable[str]:
    """Yield raw skill strings from common skills field shapes."""
    if skills is None:
        return

    if isinstance(skills, str):
        if skills.strip():
            yield skills
        return

    if isinstance(skills, Mapping):
        for value in skills.values():
            yield from _iter_skill_values(value)
        return

    if isinstance(skills, Sequence):
        for item in skills:
            if isinstance(item, Mapping):
                for field_name in ("name", "skill", "skill_name", "label"):
                    value = item.get(field_name)
                    if isinstance(value, str) and value.strip():
                        yield value
                        break
                else:
                    yield from _iter_skill_values(list(item.values()))
            else:
                yield from _iter_skill_values(item)


def _skill_stats(skills: Any) -> tuple[int, float]:
    """Compute number of skills and duplicate skill ratio."""
    normalized_skills = [
        normalize_skill(skill).casefold()
        for skill in _iter_skill_values(skills)
        if normalize_skill(skill)
    ]
    num_skills = len(normalized_skills)
    if num_skills == 0:
        return 0, 0.0

    unique_skills = len(set(normalized_skills))
    duplicate_ratio = (num_skills - unique_skills) / num_skills
    return num_skills, duplicate_ratio


def _parse_date(value: Any) -> date | None:
    """Parse common employment date formats into a date."""
    if value is None:
        return None

    if isinstance(value, date):
        return value

    if not isinstance(value, str):
        return None

    text = value.strip().lower()
    if not text or text in {"present", "current", "now", "ongoing"}:
        return None

    iso_match = re.search(r"\b(\d{4})-(\d{1,2})(?:-\d{1,2})?\b", text)
    if iso_match:
        return _safe_date(int(iso_match.group(1)), int(iso_match.group(2)))

    month_year_match = re.search(
        r"\b([a-zA-Z]{3,9})\s+(\d{4})\b|\b(\d{1,2})/(\d{4})\b",
        text,
    )
    if month_year_match:
        if month_year_match.group(1) and month_year_match.group(2):
            month = MONTH_LOOKUP.get(month_year_match.group(1).lower())
            year = int(month_year_match.group(2))
            return _safe_date(year, month) if month else None
        month = int(month_year_match.group(3))
        year = int(month_year_match.group(4))
        return _safe_date(year, month)

    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    if year_match:
        return _safe_date(int(year_match.group(1)), 1)

    return None


def _safe_date(year: int, month: int | None) -> date | None:
    """Create a date if year and month are valid."""
    if month is None:
        return None
    try:
        return date(year, month, 1)
    except ValueError:
        logger.debug("Invalid date parts year=%s month=%s", year, month)
        return None


def _is_current_job(job: Mapping[str, Any]) -> bool:
    """Infer whether a job record is current."""
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


def _employment_jobs(employment_history: Any) -> list[Mapping[str, Any]]:
    """Return dictionary-like employment records."""
    if isinstance(employment_history, Mapping):
        return [employment_history]
    if isinstance(employment_history, Sequence) and not isinstance(employment_history, str):
        return [job for job in employment_history if isinstance(job, Mapping)]
    return []


def _employment_periods(employment_history: Any) -> list[EmploymentPeriod]:
    """Extract normalized employment periods from employment history."""
    periods: list[EmploymentPeriod] = []

    for job in _employment_jobs(employment_history):
        start = _first_parsed_date(job, ("start_date", "start", "from"))
        end = _first_parsed_date(job, ("end_date", "end", "to"))
        is_current = _is_current_job(job)
        if is_current and end is None:
            end = CURRENT_DATE
        periods.append(EmploymentPeriod(start=start, end=end, is_current=is_current))

    return periods


def _first_parsed_date(job: Mapping[str, Any], field_names: tuple[str, ...]) -> date | None:
    """Parse the first available date field from a job record."""
    for field_name in field_names:
        if field_name in job:
            parsed_date = _parse_date(job[field_name])
            if parsed_date is not None:
                return parsed_date
    return None


def _has_future_dates(periods: Sequence[EmploymentPeriod]) -> bool:
    """Return whether any employment date is in the future."""
    for period in periods:
        if period.start and period.start > CURRENT_DATE:
            return True
        if period.end and period.end > CURRENT_DATE:
            return True
    return False


def _has_overlapping_employment(periods: Sequence[EmploymentPeriod]) -> bool:
    """Return whether any known employment periods overlap."""
    dated_periods = sorted(
        (period for period in periods if period.start and period.end),
        key=lambda period: period.start or date.min,
    )

    for previous, current in zip(dated_periods, dated_periods[1:], strict=False):
        if previous.end and current.start and current.start < previous.end:
            return True
    return False


def _career_gap_months(periods: Sequence[EmploymentPeriod]) -> int:
    """Return the largest known career gap in months."""
    dated_periods = sorted(
        (period for period in periods if period.start and period.end),
        key=lambda period: period.start or date.min,
    )
    max_gap = 0

    for previous, current in zip(dated_periods, dated_periods[1:], strict=False):
        if previous.end and current.start and current.start > previous.end:
            gap = _months_between(previous.end, current.start)
            max_gap = max(max_gap, gap)

    return max_gap


def _months_between(start: date, end: date) -> int:
    """Return whole calendar months between two dates."""
    return max(0, (end.year - start.year) * 12 + end.month - start.month)


def _job_hops(periods: Sequence[EmploymentPeriod]) -> int:
    """Count short known employment periods under one year."""
    hops = 0
    for period in periods:
        if period.start and period.end and _months_between(period.start, period.end) < 12:
            hops += 1
    return hops


def _experience_vs_job_history(years_experience: Any, periods: Sequence[EmploymentPeriod]) -> bool:
    """Flag large mismatch between stated experience and dated job history."""
    stated_years = _coerce_float(years_experience)
    if stated_years is None:
        return False

    history_months = 0
    for period in periods:
        if period.start and period.end:
            history_months += _months_between(period.start, period.end)

    if history_months == 0:
        return False

    history_years = history_months / 12
    return abs(stated_years - history_years) > 3


def _coerce_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _coerce_text(value: Any) -> str:
    """Convert scalar values to strings while preserving missing values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _keyword_stuffing_score(text: str) -> float:
    """Estimate repeated keyword concentration in the resume text."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]{1,}", text.lower())
    if not words:
        return 0.0

    counts = Counter(words)
    repeated_terms = sum(count for count in counts.values() if count >= 5)
    return repeated_terms / len(words)


def penalty_metadata_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compute credibility metadata for one candidate."""
    features = extract_features(candidate)
    skills = features.get("skills")
    semantic_document = features.get("semantic_document")
    semantic_text = semantic_document if isinstance(semantic_document, str) else ""
    employment_history = features.get("employment_history")
    periods = _employment_periods(employment_history)

    num_skills, duplicate_skill_ratio = _skill_stats(skills)
    empty_resume = not semantic_text.strip()
    text_length = len(semantic_text)
    future_employment_dates = _has_future_dates(periods)
    overlapping_employment = _has_overlapping_employment(periods)
    career_gap = _career_gap_months(periods)
    job_hops = _job_hops(periods)
    experience_vs_job_history = _experience_vs_job_history(
        features.get("years_experience"),
        periods,
    )
    timeline_flag = (
        future_employment_dates
        or overlapping_employment
        or career_gap >= 24
        or experience_vs_job_history
    )
    keyword_stuffing_score = _keyword_stuffing_score(semantic_text)
    credibility_flag = (
        empty_resume
        or timeline_flag
        or duplicate_skill_ratio >= 0.5
        or keyword_stuffing_score >= 0.25
    )
    suspicious_profile_flag = credibility_flag or (num_skills >= 100 and text_length < 500)

    return {
        "candidate_id": _coerce_text(features.get("candidate_id")),
        "num_skills": num_skills,
        "duplicate_skill_ratio": duplicate_skill_ratio,
        "empty_resume": empty_resume,
        "text_length": text_length,
        "future_employment_dates": future_employment_dates,
        "overlapping_employment": overlapping_employment,
        "career_gap": career_gap,
        "job_hops": job_hops,
        "experience_vs_job_history": experience_vs_job_history,
        "timeline_flag": timeline_flag,
        "credibility_flag": credibility_flag,
        "keyword_stuffing_score": keyword_stuffing_score,
        "suspicious_profile_flag": suspicious_profile_flag,
    }


def build_penalty_metadata(candidate_path: Path) -> pd.DataFrame:
    """Build credibility metadata for all candidates."""
    rows: list[dict[str, Any]] = []

    for candidate in tqdm(iter_candidates(candidate_path), desc="Building metadata"):
        rows.append(penalty_metadata_row(candidate))

    dataframe = pd.DataFrame.from_records(rows)
    return optimize_dataframe(dataframe)


def optimize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply compact dtypes to metadata columns."""
    optimized = dataframe.copy()

    integer_columns = ("num_skills", "text_length", "career_gap", "job_hops")
    for column in integer_columns:
        optimized[column] = pd.to_numeric(
            optimized[column],
            errors="coerce",
            downcast="unsigned",
        )

    float_columns = ("duplicate_skill_ratio", "keyword_stuffing_score")
    for column in float_columns:
        optimized[column] = pd.to_numeric(
            optimized[column],
            errors="coerce",
            downcast="float",
        )

    boolean_columns = (
        "empty_resume",
        "future_employment_dates",
        "overlapping_employment",
        "experience_vs_job_history",
        "timeline_flag",
        "credibility_flag",
        "suspicious_profile_flag",
    )
    for column in boolean_columns:
        optimized[column] = optimized[column].astype("bool")

    optimized["candidate_id"] = optimized["candidate_id"].astype("string")
    return optimized


def save_penalty_metadata(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save credibility metadata to a Parquet artifact."""
    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataframe.to_parquet(artifact_path, index=False)
    except ImportError as exc:
        msg = "Saving Parquet requires pyarrow or fastparquet to be installed"
        logger.error(msg)
        raise RuntimeError(msg) from exc
    except OSError:
        logger.exception("Failed to save penalty metadata to %s", artifact_path)
        raise

    logger.info("Saved penalty metadata to %s", artifact_path)


def main(
    candidate_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Build and save credibility metadata."""
    start_time = time.perf_counter()
    dataframe = build_penalty_metadata(candidate_path)
    save_penalty_metadata(dataframe, output_path)
    elapsed_seconds = time.perf_counter() - start_time

    print(f"Rows: {len(dataframe)}")
    print(f"Saved: {output_path}")
    print(f"Time taken: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()