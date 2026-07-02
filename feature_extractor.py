"""Extract standardized candidate features from raw candidate records."""

import logging
from collections.abc import Mapping, Sequence
from typing import Any

logger = logging.getLogger(__name__)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return value as a mapping when possible."""
    if isinstance(value, Mapping):
        return value
    return None


def _as_sequence_of_mappings(value: Any) -> list[Mapping[str, Any]]:
    """Return mapping elements from a sequence."""
    if isinstance(value, str) or not isinstance(value, Sequence):
        return []

    return [item for item in value if isinstance(item, Mapping)]


def _is_current_role(job: Mapping[str, Any]) -> bool:
    """Determine whether an employment entry represents the current role."""
    if job.get("is_current") is True:
        return True

    end_date = job.get("end_date")
    if end_date is None:
        return True

    if isinstance(end_date, str):
        return end_date.strip().lower() in {
            "",
            "present",
            "current",
            "ongoing",
        }

    return False


def _current_job(career_history: Any) -> Mapping[str, Any] | None:
    """Return the current employment record."""
    jobs = _as_sequence_of_mappings(career_history)

    if not jobs:
        return None

    for job in jobs:
        if _is_current_role(job):
            return job

    return jobs[0]


def _skill_names(skills: Any) -> list[str]:
    """Extract skill names."""
    names: list[str] = []

    for skill in _as_sequence_of_mappings(skills):
        name = skill.get("name")
        if isinstance(name, str):
            names.append(name)

    return names


def _career_text(career_history: Any) -> str:
    """Flatten career history into semantic text."""
    parts: list[str] = []

    for job in _as_sequence_of_mappings(career_history):
        for field in (
            "title",
            "company",
            "industry",
            "description",
        ):
            value = job.get(field)
            if isinstance(value, str):
                parts.append(value)

    return " ".join(parts)


def _education_text(education: Any) -> str:
    """Flatten education into semantic text."""
    parts: list[str] = []

    for edu in _as_sequence_of_mappings(education):
        for field in (
            "institution",
            "degree",
            "field_of_study",
        ):
            value = edu.get(field)
            if isinstance(value, str):
                parts.append(value)

    return " ".join(parts)


def build_semantic_document(candidate: Mapping[str, Any]) -> str:
    """Create one semantic document for embeddings and TF-IDF."""

    profile = candidate.get("profile", {})
    jobs = candidate.get("career_history", [])
    education = candidate.get("education", [])
    skills = candidate.get("skills", [])

    parts = []

    # -----------------------------
    # Profile
    # -----------------------------
    if profile.get("current_title"):
        parts.append(profile["current_title"])

    if profile.get("summary"):
        parts.append(profile["summary"])

    # -----------------------------
    # Skills
    # -----------------------------
    if skills:
        parts.append("Skills")

        for skill in skills:
            if isinstance(skill, dict):
                name = skill.get("name")
            else:
                name = str(skill)

            if name:
                parts.append(name)

    # -----------------------------
    # Experience
    # -----------------------------
    if jobs:
        parts.append("Experience")

        for job in jobs:
            if job.get("title"):
                parts.append(job["title"])

            if job.get("industry"):
                parts.append(job["industry"])

            if job.get("description"):
                parts.append(job["description"])

    # -----------------------------
    # Education
    # -----------------------------
    if education:
        parts.append("Education")

        for edu in education:
            degree = edu.get("degree")
            field = edu.get("field_of_study")

            if degree and field:
                parts.append(f"{degree} {field}")
            elif degree:
                parts.append(degree)
            elif field:
                parts.append(field)

    return "\n".join(parts)

def extract_features(candidate: Mapping[str, Any]) -> dict[str, Any]:

    profile = candidate.get("profile", {})

    return {
        "candidate_id": candidate.get("candidate_id"),
        "semantic_document": build_semantic_document(candidate),
        "skills": candidate.get("skills"),
        "years_experience": profile.get("years_of_experience"),
        "current_title": profile.get("current_title"),
        "current_company": profile.get("current_company"),
        "education": candidate.get("education"),
        "location": profile.get("location"),
        "employment_history": candidate.get("career_history"),
        "redrob_signals": candidate.get("redrob_signals"),
    }


def main() -> None:
    """Module smoke test."""
    logging.basicConfig(level=logging.INFO)
    logger.info("feature_extractor.py ready.")


if __name__ == "__main__":
    main()