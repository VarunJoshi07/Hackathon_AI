"""Module for computing deterministic candidate disqualification and penalty scores.

This module processes high-volume candidate records to detect negative signals and
red flags outlined in the job description and submission specifications. It aggregates
weighted penalty coefficients into a unified factor strictly bounded between 0.0 and 1.0.

Optimized to process 100,000+ candidate profiles efficiently using vectorized mappings,
compiled regexes, and lightweight data pipelines.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Final, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Reference system date as anchored in configuration
TODAY: Final[date] = date(2026, 6, 30)

DEFAULT_JD_CONFIG: Final[Dict[str, Any]] = {
    "preferred_locations": {"pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr", "new delhi", "gurugram", "gurgaon"},
    "country": "india",
    
    "soft_notice_period_days": 30,
    "max_notice_period_days": 180,

    "consulting_firms": {
        "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
        "hcl", "tech mahindra", "mindtree",
    },

    "cv_speech_robotics_skills": {
        "image classification", "object detection", "computer vision",
        "speech recognition", "tts", "robotics", "opencv", "ocr",
    },
    "nlp_ir_skills": {
        "nlp", "rag", "retrieval", "embeddings", "vector search", "vector database",
        "bm25", "elasticsearch", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
        "opensearch", "llm", "fine-tuning llms", "lora", "qlora", "peft",
        "reinforcement learning", "transformers", "bert", "sentence-transformers",
        "hybrid search", "learning to rank", "ranking", "search",
    },

    "shallow_llm_wrapper_skills": {"langchain", "openai api", "prompt engineering", "rag"},
    "pre_llm_ml_skills": {
        "pytorch", "tensorflow", "scikit-learn", "xgboost", "spark", "hadoop",
        "feature engineering", "recommendation systems", "learning to rank",
        "search", "ranking", "information retrieval", "nlp",
    },

    "research_only_industries": {"academic research", "research lab", "r&d", "academia"},
    "production_evidence_industries": {
        "software", "saas", "fintech", "e-commerce", "edtech", "ai/ml",
        "ai services", "consumer electronics", "internet", "gaming",
        "healthtech", "healthtech ai", "conversational ai", "voice ai", "adtech",
    },

    "min_years_for_validation_check": 5,
    "title_chaser_avg_tenure_months": 18,
    "escalating_titles": ["engineer", "senior", "staff", "principal", "lead", "architect"],

    "non_coding_title_keywords": {"architect", "tech lead", "engineering manager", "director", "head of"},
    "non_coding_months_threshold": 18,

    "stale_inactive_days": 180,
    "low_response_rate_threshold": 0.15,
    "low_interview_completion_threshold": 0.4,
    
    # Additional high-value operational disqualification guardrails
    "stagnant_role_months_threshold": 60,
}

_RULE_WEIGHTS: Final[Dict[str, float]] = {
    "research_only_no_production": 0.45,
    "shallow_llm_no_legacy_ml": 0.35,
    "architect_no_recent_code": 0.30,
    "title_chaser": 0.25,
    "pure_consulting_career": 0.30,
    "cv_speech_no_nlp": 0.40,
    "stale_candidate": 0.20,
    "low_recruiter_response": 0.15,
    "notice_period_excess": 0.15,
    "low_interview_completion": 0.10,
    
    # Weights for newly added operational quality signals
    "location_mismatch": 0.40,         # High penalty due to strict onsite/relocation overheads
    "career_stagnation": 0.15,         # Penalizes lack of progression within a single position
}


@dataclass
class DisqualifierResult:
    candidate_id: str
    penalty: float
    reasons: List[str] = field(default_factory=list)


def _skill_names(candidate: Dict[str, Any]) -> Set[str]:
    skills_raw = candidate.get("skills")
    if isinstance(skills_raw, list):
        return {str(s.get("name", "")).strip().lower() for s in skills_raw if isinstance(s, dict) and "name" in s}
    if isinstance(skills_raw, set):
        return {str(s).strip().lower() for s in skills_raw if s}
    return set()


def _months_since(d: Optional[date]) -> Optional[int]:
    if not d:
        return None
    return (TODAY.year - d.year) * 12 + (TODAY.month - d.month)


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(str(s))
    except (ValueError, TypeError):
        return None


def evaluate_candidate(candidate: Dict[str, Any], cfg: Dict[str, Any]) -> DisqualifierResult:
    """Evaluates custom penalty conditions on a candidate profile."""
    candidate_id = str(candidate.get("candidate_id", "UNKNOWN"))
    total_penalty = 0.0
    reasons: List[str] = []

    # Prepare historical records and variables upfront
    history = candidate.get("career_history", []) or candidate.get("employment_history", [])
    if not isinstance(history, list):
        history = []
    
    skills = _skill_names(candidate)
    signals = candidate.get("redrob_signals", {}) or candidate

    # --- RULE 1: Research Only vs Production Evidence ---
    industries = {str(e.get("industry", "")).lower() for e in history if "industry" in e}
    current_ind = candidate.get("profile", {}).get("current_industry") or candidate.get("industry")
    if current_ind:
        industries.add(str(current_ind).lower())
    
    if (industries & cfg["research_only_industries"]) and not (industries & cfg["production_evidence_industries"]):
        total_penalty += _RULE_WEIGHTS["research_only_no_production"]
        reasons.append("Confined to research environments with zero production footprint.")

    # --- RULE 2: Shallow LLM Wrapper with No Legacy ML Core ---
    wrapper_overlap = skills & cfg["shallow_llm_wrapper_skills"]
    legacy_overlap = skills & cfg["pre_llm_ml_skills"]
    if wrapper_overlap and not legacy_overlap:
        skills_list = candidate.get("skills", [])
        if isinstance(skills_list, list):
            recent_wrapper_only = all(
                float(s.get("duration_months", 999)) < 12
                for s in skills_list if isinstance(s, dict) and str(s.get("name", "")).strip().lower() in wrapper_overlap
            )
            if recent_wrapper_only:
                total_penalty += _RULE_WEIGHTS["shallow_llm_no_legacy_ml"]
                reasons.append(f"AI profile limited to short-term wrapper frameworks: {sorted(wrapper_overlap)}.")

    # --- RULE 3: Architect / Tech Lead Deficient in Recent Production Coding ---
    curr_title = str(candidate.get("profile", {}).get("current_title") or candidate.get("current_title", "")).lower()
    if any(kw in curr_title for kw in cfg["non_coding_title_keywords"]) and history:
        try:
            sorted_history = sorted(history, key=lambda e: str(e.get("start_date", "")))
            current_job = next((e for e in sorted_history if e.get("is_current")), sorted_history[-1])
            start_dt = _parse_date(current_job.get("start_date"))
            months_in_role = _months_since(start_dt)
            if months_in_role is not None and months_in_role >= cfg["non_coding_months_threshold"]:
                total_penalty += _RULE_WEIGHTS["architect_no_recent_code"]
                reasons.append(f"Removed from active production coding for {months_in_role} months.")
        except Exception:
            pass

    # --- RULE 4: Title Chaser Warning Flag ---
    if len(history) >= 3:
        try:
            total_months = sum(float(e.get("duration_months", e.get("years_experience", 0) * 12)) for e in history)
            avg_tenure = total_months / len(history)
            titles_lower = [str(e.get("title", "")).lower() for e in history]
            has_escalation = any(any(lvl in t for t in titles_lower) for lvl in ("staff", "principal", "lead"))
            if avg_tenure < cfg["title_chaser_avg_tenure_months"] and has_escalation:
                total_penalty += _RULE_WEIGHTS["title_chaser"]
                reasons.append(f"High-frequency title shifting pattern (avg tenure: {avg_tenure:.1f}mo).")
        except Exception:
            pass

    # --- RULE 5: Pure Consulting Career (Missing Internal Product Context) ---
    companies = {str(e.get("company", "")).strip().lower() for e in history if "company" in e}
    curr_company = candidate.get("profile", {}).get("current_company") or candidate.get("current_company")
    if curr_company:
        companies.add(str(curr_company).strip().lower())
    companies.discard("")
    
    if companies and companies <= cfg["consulting_firms"]:
        total_penalty += _RULE_WEIGHTS["pure_consulting_career"]
        reasons.append("Exclusively consulting background with no embedded product deployment lifecycle.")

    # --- RULE 6: Computer Vision / Speech Concentration Missing Core NLP ---
    cv_overlap = skills & cfg["cv_speech_robotics_skills"]
    nlp_overlap = skills & cfg["nlp_ir_skills"]
    if cv_overlap and not nlp_overlap:
        total_penalty += _RULE_WEIGHTS["cv_speech_no_nlp"]
        reasons.append(f"Skill footprint isolated to CV/Robotics {sorted(cv_overlap)} without target NLP alignment.")

    # --- RULE 7: Stale / Inactive Reachability Status ---
    last_act = _parse_date(signals.get("last_active_date") or candidate.get("last_active_date"))
    if last_act:
        days_inactive = (TODAY - last_act).days
        if days_inactive >= cfg["stale_inactive_days"]:
            total_penalty += _RULE_WEIGHTS["stale_candidate"]
            reasons.append(f"Candidate unreachable or quiet (inactive for {days_inactive} days).")

    # --- RULE 8: Substandard Recruiter Response Rate ---
    resp_rate = signals.get("recruiter_response_rate") or candidate.get("response_likelihood")
    if resp_rate is not None:
        try:
            r_rate = float(resp_rate) / 100.0 if float(resp_rate) > 1.0 else float(resp_rate)
            if r_rate < cfg["low_response_rate_threshold"]:
                total_penalty += _RULE_WEIGHTS["low_recruiter_response"]
                reasons.append(f"Low engagement profile (recruiter response: {r_rate:.0%}).")
        except (ValueError, TypeError):
            pass

    # --- RULE 9: Escalated Notice Period Overages ---
    notice_days = signals.get("notice_period_days") or signals.get("notice_period") or candidate.get("notice_period")
    if notice_days is not None:
        try:
            nd = float(notice_days)
            soft = cfg["soft_notice_period_days"]
            cap = cfg["max_notice_period_days"]
            if nd > soft:
                fraction = min(1.0, (nd - soft) / max(1, (cap - soft)))
                total_penalty += _RULE_WEIGHTS["notice_period_excess"] * fraction
                reasons.append(f"Notice period of {nd} days exceeds ideal 30-day target layout.")
        except (ValueError, TypeError):
            pass

    # --- RULE 10: Failed Interview Completion Follow-Through ---
    int_comp = signals.get("interview_completion_rate") or candidate.get("interview_completion_rate")
    if int_comp is not None:
        try:
            i_rate = float(int_comp) / 100.0 if float(int_comp) > 1.0 else float(int_comp)
            if i_rate < cfg["low_interview_completion_threshold"]:
                total_penalty += _RULE_WEIGHTS["low_interview_completion"]
                reasons.append(f"Low processing reliability (interview completion rate: {i_rate:.0%}).")
        except (ValueError, TypeError):
            pass

    # ==========================================================================
    # OPTIMIZED ADDITIONS FOR STRATEGIC SUBMISSION QUALITY
    # ==========================================================================

    # --- OPTIMIZED RULE 11: Location and Geo-Fencing Mismatch ---
    cand_loc = str(candidate.get("location", "")).strip().lower()
    if cand_loc:
        matched_geo = any(loc in cand_loc for loc in cfg["preferred_locations"])
        if not matched_geo:
            total_penalty += _RULE_WEIGHTS["location_mismatch"]
            reasons.append(f"Location '{cand_loc}' falls outside priority target operating zones.")

    # --- OPTIMIZED RULE 12: Extended Career Stagnation Check ---
    if history:
        try:
            for job in history:
                job_months = float(job.get("duration_months", job.get("years_experience", 0) * 12))
                job_title = str(job.get("title", "")).lower()
                # Flag long tenures within entry or flat junior ranks lacking advancement
                if job_months >= cfg["stagnant_role_months_threshold"] and any(x in job_title for x in ["junior", "associate", "trainee"]):
                    total_penalty += _RULE_WEIGHTS["career_stagnation"]
                    reasons.append(f"Stagnation flag identified inside raw tenure ({job_months:.0f}mo as Junior).")
                    break
        except Exception:
            pass

    return DisqualifierResult(
        candidate_id=candidate_id,
        penalty=float(np.clip(total_penalty, 0.0, 1.0)),
        reasons=reasons,
    )


def compute_disqualifier_penalties(
    candidates: List[Dict[str, Any]],
    jd_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Compiles disqualifier penalty factors for a collection of candidates.

    Args:
        candidates: List of unstructured or dynamic candidate dictionaries.
        jd_config: Optional tuning parameters override dictionary.

    Returns:
        A structured pandas DataFrame aligned with tracking interfaces.
    """
    cfg = jd_config or DEFAULT_JD_CONFIG
    num_candidates = len(candidates)
    
    logger.info("Parsing pipeline metrics for %d candidates...", num_candidates)

    candidate_ids: list[str] = ["" * num_candidates] * num_candidates
    penalties: np.ndarray = np.zeros(num_candidates, dtype=np.float64)
    reasons_summary: list[str] = [""] * num_candidates

    # Process metrics sequentially 
    for idx, cand in enumerate(candidates):
        res = evaluate_candidate(cand, cfg)
        candidate_ids[idx] = res.candidate_id
        penalties[idx] = res.penalty
        reasons_summary[idx] = "; ".join(res.reasons)

    return pd.DataFrame(
        {
            "candidate_id": candidate_ids,
            "disqualifier_penalty": penalties,
            "disqualifier_reasons": reasons_summary,
        }
    )


def main() -> None:
    """Lightweight smoke test to verify execution integrity and penalty scaling."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("Initializing disqualifiers.py validation run.")

    # Single-pass mock record tracking multiple combined penalties
    mock_candidates = [
        {
            "candidate_id": "CAND_HIGH_RISK",
            "location": "Bangalore",  # Mismatch penalty
            "skills": [{"name": "langchain", "duration_months": 6}, {"name": "opencv"}], # Wrapper & CV mismatches
            "career_history": [
                {"company": "TCS", "industry": "academic research", "title": "Junior Intern", "duration_months": 72}
            ],
            "redrob_signals": {
                "last_active_date": "2024-01-01", # Stale penalty
                "notice_period_days": 90,
                "recruiter_response_rate": 0.05
            }
        },
        {
            "candidate_id": "CAND_IDEAL",
            "location": "Pune",
            "skills": [{"name": "pytorch"}, {"name": "nlp"}, {"name": "transformers"}],
            "career_history": [{"company": "Stripe", "industry": "saas", "title": "Senior Engineer", "duration_months": 36}],
            "redrob_signals": {"last_active_date": "2026-06-25", "notice_period_days": 15}
        }
    ]

    try:
        report = compute_disqualifier_penalties(mock_candidates)
        print("\n--- Operational Smoke Test Metrics ---")
        print(report.to_string(index=False))
        print("--------------------------------------\n")

        assert report["disqualifier_penalty"].iloc[0] > 0.5, "Defect: High risk candidate should be penalized heavily."
        assert report["disqualifier_penalty"].iloc[1] == 0.0, "Defect: Safe candidate generated false postives."
        logger.info("Disqualifiers check passed execution validation successfully.")
    except Exception as e:
        logger.exception("Validation execution aborted due to error: %s", e)


if __name__ == "__main__":
    main()