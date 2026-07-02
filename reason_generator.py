"""Module for generating fully dynamic, deterministic, and evidence-based ranking explanations.

This module analyzes candidates against job requirements without using external LLM APIs,
guaranteeing zero hallucinations and complete reproducibility as required by the
Redrob Hackathon guidelines.
"""

from typing import Dict, Any, List


def generate_reasoning(candidate: Dict[str, Any], jd_constraints: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a structured explanation payload for a given candidate and JD constraints.

    Args:
        candidate: A dictionary containing pre-computed metrics and raw candidate fields.
        jd_constraints: A dictionary containing thresholds and mandatory/preferred criteria.

    Returns:
        A dictionary containing structured reasoning, highlights, strengths, and risks.
    """
    candidate_id = candidate.get("candidate_id", "UNKNOWN")
    rank = candidate.get("rank", 1)
    final_score = candidate.get("final_score", 0.0)

    # 1. Gather numerical thresholds into natural descriptor mappings
    similarity = candidate.get("cross_encoder_score", candidate.get("embedding_similarity", 0.0))
    if similarity >= 0.90:
        sim_desc = "Exceptional semantic alignment"
    elif similarity >= 0.80:
        sim_desc = "Strong semantic alignment"
    elif similarity >= 0.65:
        sim_desc = "Good semantic alignment"
    else:
        sim_desc = "Moderate semantic alignment"

    behavior_score = candidate.get("behavior_score", 0.0)
    if behavior_score >= 0.85:
        behavior_desc = "exceptional platform availability"
    elif behavior_score >= 0.70:
        behavior_desc = "strong platform engagement"
    else:
        behavior_desc = "moderate platform engagement"

    career_score = candidate.get("career_score", 0.0)
    if career_score >= 0.85:
        career_desc = "stellar professional trajectory"
    elif career_score >= 0.70:
        career_desc = "stable and progressive professional history"
    else:
        career_desc = "consistent career history"

    # 2. Skill parsing & matching
    matched_skills = candidate.get("matched_skills", [])
    missing_skills = candidate.get("missing_skills", [])
    mandatory_skills = jd_constraints.get("mandatory skills", [])
    preferred_skills = jd_constraints.get("preferred skills", [])

    matched_mandatory = [s for s in matched_skills if s in mandatory_skills]
    missing_mandatory = [s for s in missing_skills if s in mandatory_skills]
    matched_preferred = [s for s in matched_skills if s in preferred_skills]

    # 3. Dynamic Section Building (Prioritized Evidential Ordering)
    highlights: List[str] = []
    strengths: List[str] = []
    risks: List[str] = []
    summary_sentences: List[str] = []

    # --- FACT 1: SEMANTIC ALIGNMENT ---
    summary_sentences.append(f"This candidate demonstrates {sim_desc.lower()} with the target position.")
    strengths.append(sim_desc)
    highlights.append(sim_desc)

    # --- FACT 2: MANDATORY SKILLS ---
    if mandatory_skills:
        if len(missing_mandatory) == 0:
            summary_sentences.append("They satisfy all mandatory technical skill requirements.")
            strengths.append("Matched all mandatory skills")
            highlights.append("Matched all mandatory skills")
        elif len(matched_mandatory) > 0:
            msg = f"Matched {len(matched_mandatory)} of {len(mandatory_skills)} mandatory skills"
            summary_sentences.append(f"They possess solid coverage of core technical domains, matching {len(matched_mandatory)} required skills.")
            strengths.append(msg)
            highlights.append(msg)

    # --- FACT 3: PREFERRED SKILLS ---
    if matched_preferred:
        summary_sentences.append(f"Additionally, they possess preferred experience in {', '.join(matched_preferred[:3])}.")
        strengths.append(f"Matches preferred skill criteria: {', '.join(matched_preferred[:2])}")

    # --- FACT 4: YEARS OF EXPERIENCE ---
    years_exp = candidate.get("years_experience")
    if years_exp is not None:
        highlights.append(f"{years_exp} years experience")
        exp_min = jd_constraints.get("experience range", {}).get("min", 0)
        exp_max = jd_constraints.get("experience range", {}).get("max", 99)
        if exp_min <= years_exp <= exp_max:
            strengths.append(f"Experience level ({years_exp} years) fits requested window")
        elif years_exp < exp_min:
            risks.append(f"Total experience ({years_exp} years) sits below preferred range")
        else:
            strengths.append(f"Highly seasoned background with {years_exp} years of industry tenure")

    # --- FACT 5: PRODUCTION SYSTEM BACKGROUND ---
    prod_bg = candidate.get("production_background")
    if prod_bg is True:
        summary_sentences.append("Their profile documents past ownership of production systems deployment.")
        strengths.append("Strong production background")
        highlights.append("Production system deployment")
    elif prod_bg is False:
        # Avoid saying anything if production data is missing, but if explicit False, capture it if a requirement
        if jd_constraints.get("production requirement"):
            risks.append("Lacks explicit production deployment background")

    # --- FACT 6: CAREER QUALITY & PROFILE PRESTIGE ---
    summary_sentences.append(f"Their background is backed by a {career_desc}.")
    if career_score >= 0.70:
        strengths.append(f"High-quality career progression score")

    # --- FACT 7: BEHAVIORAL SIGNALS & AVAILABILITY ---
    signals = candidate.get("redrob_signals", {})
    response_rate = signals.get("recruiter_response_rate")
    notice_days = signals.get("notice_period_days")
    github_score = signals.get("github_activity_score", -1)

    availability_clauses = []
    if response_rate is not None:
        if response_rate >= 0.75:
            strengths.append("Excellent recruiter response rate")
            availability_clauses.append("exhibit high recruiter responsiveness")
        elif response_rate < 0.20:
            risks.append(f"Low recruiter responsiveness rate ({int(response_rate * 100)}%)")

    if notice_days is not None:
        highlights.append(f"{notice_days}-day notice period")
        if notice_days <= 30:
            strengths.append(f"Short notice period ({notice_days} days)")
            availability_clauses.append(f"possess an immediate or short notice timeline of {notice_days} days")
        elif notice_days > 90:
            risks.append(f"Extended notice period of {notice_days} days")

    if availability_clauses:
        summary_sentences.append(f"Active engagement metrics show they {' and '.join(availability_clauses)}.")

    # --- FACT 8: EVIDENCE-BASED RISKS ---
    for missing_skill in missing_mandatory[:3]:
        risks.append(f"Missing mandatory skill: {missing_skill}")
    
    if github_score == 0 or github_score == -1:
        risks.append("No active open-source or public GitHub activity recorded")

    # 4. Construct Final Response Payload
    reason_text = " ".join(summary_sentences)

    return {
        "candidate_id": candidate_id,
        "rank": int(rank),
        "final_score": float(final_score),
        "reason": reason_text,
        "highlights": highlights,
        "strengths": strengths,
        "risks": risks
    }