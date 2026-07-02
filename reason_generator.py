"""Module for generating quantitative, score-defensive candidate explanations."""

from typing import Any, Dict, Optional


def generate_reasoning(
    candidate: Dict[str, Any],
    jd_constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """
    Generates a deterministic reasoning string backed by quantitative metrics.
    Buckets language strictly by score thresholds to ensure tone alignment,
    defensibility, and natural gradient degradation across ranks.
    """
    # 1. Safely extract core metrics across potential schema variations
    final_score = candidate.get("final_score", 0.0)
    ce_score = candidate.get("cross_encoder_score", final_score)
    emb_score = candidate.get("embedding_similarity", final_score)
    
    # Check common experience key formats safely
    exp = candidate.get("years_experience")
    if exp is None:
        exp = candidate.get("years_of_experience")
    if exp is None:
        exp = candidate.get("profile", {}).get("years_of_experience", "N/A")

    # 2. Bucket qualitative descriptions based on the Cross-Encoder/Final Score
    if ce_score >= 0.90:
        verdict = "an outstanding fit"
        overall_assessment = "exceptional semantic match with minimal onboarding risk, making them highly recommended for immediate interview"
    elif ce_score >= 0.85:
        verdict = "an excellent fit"
        overall_assessment = "substantial alignment with technical prerequisites and a strong recommendation for technical review"
    elif ce_score >= 0.80:
        verdict = "a strong fit"
        overall_assessment = "dependable competence across core tech stacks and a clear recommendation for further technical assessment"
    elif ce_score >= 0.75:
        verdict = "a good fit"
        overall_assessment = "a solid technical foundation with minor gaps compared to higher-ranked profiles"
    elif ce_score >= 0.70:
        verdict = "a moderate fit"
        overall_assessment = "partial coverage of core technologies with some gaps that warrant deeper validation during interviews"
    elif ce_score >= 0.65:
        verdict = "a partial fit"
        overall_assessment = "transferable engineering experience, though additional validation is required to assess key proficiencies"
    else:
        verdict = "a borderline fit"
        overall_assessment = "limited direct alignment with required technologies, necessitating further evaluation if capacity allows"

    # 3. Format metrics cleanly for template injection
    try:
        ce_str = f"{float(ce_score):.2f}"
    except (ValueError, TypeError):
        ce_str = "N/A"

    try:
        emb_str = f"{float(emb_score):.2f}"
    except (ValueError, TypeError):
        emb_str = "N/A"

    if isinstance(exp, (int, float)):
        exp_str = f"brings {float(exp):.1f} years of relevant experience"
    else:
        exp_str = "demonstrates transferable background experience"

    # 4. Construct the realistic, defensible ATS reasoning text
    reason_text = (
        f"Ranked based on {verdict} with the job description (Cross-Encoder Score: {ce_str}). "
        f"The candidate also achieved an embedding similarity of {emb_str} and {exp_str}, "
        f"indicating {overall_assessment}."
    )

    return {"reason": reason_text}
