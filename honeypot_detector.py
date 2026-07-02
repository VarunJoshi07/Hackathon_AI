"""Module for detecting profile fraud, ATS keyword stuffing, honeypots, and AI-generated resumes.

This module provides high-performance compliance auditing to protect the discovery
pipeline against profile gaming, structural fraud, template-inflation, and inserted
honeypot traps. It evaluates profiles deterministically and yields explicit metrics bounded
between 0.0 and 1.0, independent of active text embeddings or vector rank indices.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Final, List, Optional, Set

import numpy as np

logger = logging.getLogger(__name__)

# System date anchor for timeline evaluation bounds
TODAY: Final[date] = date(2026, 6, 30)

# Honeypots / Traps defined by recruiters to catch malicious scrapers or unqualified bots
HONEYPOT_KEYWORDS: Final[Set[str]] = {
    "hyper-converged multi-modal quantum tokenization",
    "cobol-based blockchain llm engine",
    "bft-consensus automatic speech distribution architecture"
}

# AI Resume patterns and buzzword configurations
AI_BUZZWORDS: Final[Set[str]] = {
    "leveraged", "synergized", "spearheaded", "orchestrated", "revolutionized",
    "testament to", "fostered innovation", "dynamic professional", "passionate engineer"
}


@dataclass
class HoneypotResult:
    """Dataclass holding fraud evaluation outputs for a specific candidate."""
    disqualified: bool
    penalty_score: float
    reasons: List[str] = field(default_factory=list)


def _parse_date(s: Optional[str]) -> Optional[date]:
    """Safely converts dynamic timeline values into robust datetime objects."""
    if not s or isinstance(s, bool):
        return None
    try:
        return date.fromisoformat(str(s)[:10])
    except (ValueError, TypeError):
        return None


def _clean_string(text: Any) -> str:
    """Normalizes unstructured text inputs for down-stream string inspections."""
    if not text or isinstance(text, bool):
        return ""
    return str(text).strip().lower()


def detect_honeypots_and_fraud(candidate: Dict[str, Any]) -> HoneypotResult:
    """Audits candidate metadata profiles to detect anomalies, honeypots, and structural fraud.

    Args:
        candidate: Mapping structure containing candidate profile history.

    Returns:
        HoneypotResult: Encapsulated compliance and penalty flags.
    """
    disqualified = False
    penalty_score = 0.0
    reasons: List[str] = []

    # Safe layout extractions
    profile = candidate.get("profile", {}) or {}
    history = candidate.get("career_history", []) or []
    skills_raw = candidate.get("skills", []) or []
    signals = candidate.get("redrob_signals", {}) or {}

    # Gather clean collection of skills
    skills_set: Set[str] = set()
    skills_list: List[str] = []
    for s in skills_raw:
        s_name = _clean_string(s.get("name") if isinstance(s, dict) else s)
        if s_name:
            skills_set.add(s_name)
            skills_list.append(s_name)

    # 1. KEYWORD STUFFING DETECTION
    summary = _clean_string(profile.get("summary", ""))
    headline = _clean_string(profile.get("headline", ""))
    descriptions = " ".join([_clean_string(j.get("description", "")) for j in history if isinstance(j, dict)])
    full_text_blob = f"{headline} {summary} {descriptions}"

    # Check for excessive repetition of identical skill items
    if len(skills_list) > len(skills_set) + 5:
        penalty_score += 0.25
        reasons.append("Excessive repetition of identical skills found in core tracking tokens.")

    # Track unrealistic term distribution density
    words = [w for w in re.split(r"\W+", full_text_blob) if len(w) > 3]
    if words:
        for word in set(words):
            if word in {"with", "from", "systems", "software", "development", "engineer"}:
                continue
            if words.count(word) > 25:
                penalty_score += 0.20
                reasons.append(f"Unrealistically high keyword density detected for token '{word}'.")
                break

    # 2. TIMELINE FRAUD DETECTION
    intervals: List[tuple[date, date]] = []
    has_timeline_fraud = False

    for job in history:
        if not isinstance(job, dict):
            continue
        start = _parse_date(job.get("start_date"))
        end = _parse_date(job.get("end_date")) or TODAY

        if start and end:
            if start > end:
                has_timeline_fraud = True
                reasons.append("Negative employment duration found within career intervals.")
            if start > TODAY:
                has_timeline_fraud = True
                reasons.append("Impossible employment date starting after current project evaluation horizon.")
            if start <= end:
                intervals.append((start, end))

    # Audit chronologically overlapping roles (multi-employment timeline tracking)
    if len(intervals) > 1:
        intervals.sort(key=lambda x: x[0])
        overlapping_count = 0
        for i in range(len(intervals) - 1):
            if intervals[i+1][0] < intervals[i][1]:
                overlap_days = (intervals[i][1] - intervals[i+1][0]).days
                if overlap_days > 45:  # Allow standard cross-over or freelance transitions
                    overlapping_count += 1
        if overlapping_count >= 2:
            penalty_score += 0.30
            reasons.append("Highly dense overlapping employment timelines detected across career blocks.")

    # Total experience bounds matching biological age constraints
    dob = _parse_date(profile.get("date_of_birth") or profile.get("dob"))
    years_exp = float(profile.get("years_of_experience") or profile.get("years_experience") or 0.0)
    if dob:
        age_at_evaluation = (TODAY - dob).days / 365.25
        if years_exp > (age_at_evaluation - 16):
            has_timeline_fraud = True
            reasons.append(f"Stated experience ({years_exp} yrs) is biologically impossible for candidate age.")

    if has_timeline_fraud:
        disqualified = True

    # 3. FAKE SKILLS DETECTION
    for skill in skills_set:
        # Flag specialized AI/ML skills that appear in skills list but are nowhere inside context/descriptions
        if skill in {"pytorch", "tensorflow", "transformers", "fine-tuning llms", "rag"}:
            if skill not in full_text_blob:
                penalty_score += 0.15
                reasons.append(f"Declared expertise '{skill}' lacks supporting project or history descriptions.")
                break

    # Contradictory domain declarations mapping check
    if "cobol" in skills_set and any(x in skills_set for x in {"llm", "rag", "transformers"}):
        penalty_score += 0.10
        reasons.append("Contradictory tech stacks listed (Legacy Mainframe paired directly with Generative AI).")

    # 4. DUPLICATE CONTENT DETECTION
    jobs_seen: Set[tuple[str, str]] = set()
    for job in history:
        if not isinstance(job, dict):
            continue
        comp = _clean_string(job.get("company", ""))
        title = _clean_string(job.get("title", ""))
        if comp and title:
            signature = (comp, title)
            if signature in jobs_seen:
                penalty_score += 0.20
                reasons.append(f"Duplicate employment history entry discovered for company/title: {signature}.")
                break
            jobs_seen.add(signature)

    # 5. HONEYPOT / RECRUITER TRAP DETECTION
    for hp in HONEYPOT_KEYWORDS:
        if hp in full_text_blob:
            # Check context surrounding the keyword to see if it is naturally nested or stuffed
            idx = full_text_blob.find(hp)
            surrounding = full_text_blob[max(0, idx - 50): min(len(full_text_blob), idx + len(hp) + 50)]
            # If the keyword exists but lacks typical professional conversational phrasing around it, trigger penalty
            if not any(v in surrounding for v in {"built", "implemented", "scaled", "using", "designed"}):
                penalty_score += 0.50
                reasons.append(f"Honeypot keyword trap detected ('{hp}') without valid surrounding career context.")

    # 6. AI-GENERATED RESUME DETECTION
    buzzword_hits = sum(1 for b in AI_BUZZWORDS if b in full_text_blob)
    if buzzword_hits >= 4:
        penalty_score += 0.10
        reasons.append("Profile contains excessive template AI-generator buzzwords and generic statements.")

    # 7. HARD BUSINESS RULES AND BLACKLIST DISQUALIFIERS
    if signals.get("blacklist_match", False) or candidate.get("blacklist_match", False):
        disqualified = True
        reasons.append("Candidate identified inside corporate internal blacklist registries.")

    if signals.get("missing_mandatory_work_authorization", False) or candidate.get("missing_work_auth", False):
        disqualified = True
        reasons.append("Mandatory regional or international work authorization criteria absent.")

    return HoneypotResult(
        disqualified=disqualified,
        penalty_score=float(np.clip(penalty_score, 0.0, 1.0)),
        reasons=reasons
    )


def main() -> None:
    """Lightweight test executor to verify candidate fraud and trap evaluations."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Initializing fraud and honeypot detection routine validation check.")

    fraudulent_mock = {
        "candidate_id": "CAND_TRAP_01",
        "profile": {
            "summary": "Expert in cobol-based blockchain llm engine platform systems architecture layout setup.",
            "years_of_experience": 25.0,
            "dob": "2010-01-01"  # Biologically impossible for 25 years of experience
        },
        "skills": [{"name": "Pytorch"}, {"name": "Pytorch"}, {"name": "COBOL"}, {"name": "LLM"}],
        "career_history": [
            {"company": "Stripe", "title": "AI Lead", "start_date": "2025-01-01", "end_date": "2024-01-01"}  # Negative timeline
        ]
    }

    try:
        report = detect_honeypots_and_fraud(fraudulent_mock)
        print("\n--- Compliance Evaluation Matrix ---")
        print(f"Is Disqualified: {report.disqualified}")
        print(f"Penalty Score  : {report.penalty_score:.2f}")
        print(f"Flags Triggered: {report.reasons}")
        print("------------------------------------\n")

        assert report.disqualified is True, "Validation failure: Timeline fraud skipped disqualification rules."
        assert report.penalty_score > 0.40, "Validation failure: Target honeypots or skill stuffing unpenalized."
        logger.info("Compliance detector checks passed runtime specification targets successfully.")
    except Exception as e:
        logger.exception("Compliance evaluation routine validation run failed: %s", e)


if __name__ == "__main__":
    main()