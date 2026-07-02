"""Module for generating quantitative, score-defensive candidate explanations."""

import json
from typing import Any, Dict, List, Optional


def _clean_skills(skills_input: Any) -> List[str]:
    """Safely parses cross-schema skills fields into a clean list of strings."""
    if not skills_input:
        return []
    if isinstance(skills_input, str):
        if skills_input.startswith("["):
            try:
                skills_input = json.loads(skills_input)
            except Exception:
                pass
        else:
            return [s.strip() for s in skills_input.split(",") if s.strip()]
    if isinstance(skills_input, list):
        cleaned = []
        for s in skills_input:
            if isinstance(s, dict):
                name = s.get("name") or s.get("skill")
                if name:
                    cleaned.append(str(name))
            elif isinstance(s, str):
                cleaned.append(s)
        return cleaned
    return []


def generate_reasoning(
    candidate: Dict[str, Any],
    jd_constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Generates a deterministic reasoning string backed by quantitative metrics.
    Ensures tone alignment, defensibility, and natural gradient degradation 
    across ranks without using any external APIs or models.
    """
    # 1. Safely extract core metrics across potential schema variations
    ce_val = candidate.get("cross_encoder_score")
    if ce_val is None:
        ce_val = candidate.get("final_score")
        
    emb_val = candidate.get("embedding_similarity")
    career_val = candidate.get("career_score")
    
    yoe_val = candidate.get("years_experience")
    if yoe_val is None:
        yoe_val = candidate.get("years_of_experience")
    if yoe_val is None:
        yoe_val = candidate.get("profile", {}).get("years_of_experience")

    production = bool(candidate.get("production_background", False))

    # 2. Format quantitative metric snippet using defined priorities
    metric_str = ""
    if ce_val is not None:
        metric_str = f"Cross-Encoder {float(ce_val):.4f}"
    elif emb_val is not None:
        metric_str = f"Embedding Similarity {float(emb_val):.4f}"
    elif career_val is not None:
        metric_str = f"Career Score {float(career_val):.4f}"

    exp_str = ""
    if yoe_val is not None:
        exp_str = f"{float(yoe_val):.1f} years of engineering experience"
    else:
        exp_str = "relevant professional experience"

    # 3. Handle skills filters safely
    m_skills = _clean_skills(candidate.get("matched_skills"))
    miss_skills = _clean_skills(candidate.get("missing_skills"))
    
    if jd_constraints and "mandatory_skills" in jd_constraints:
        jd_mandatory = _clean_skills(jd_constraints.get("mandatory_skills"))
        if jd_mandatory:
            m_skills = [s for s in m_skills if s.lower() in [j.lower() for j in jd_mandatory]]
            miss_skills = [s for s in miss_skills if s.lower() in [j.lower() for j in jd_mandatory]]

    # 4. Resolve rank bucket and select deterministic variation index
    rank_val = candidate.get("rank")
    try:
        rank = int(rank_val) if rank_val is not None else 50
    except (TypeError, ValueError):
        rank = 50

    cand_id_str = str(candidate.get("candidate_id", ""))
    variant_idx = sum(ord(c) for c in cand_id_str) if cand_id_str else rank
    v = variant_idx % 3

    sentences = []

    # 5. Build rank-aware narrative components
    if rank <= 10:
        # Rank 1-10: Confident tone, strengths focused, minor concerns
        openings = [
            f"Strong semantic alignment with the job requirements ({metric_str}), {exp_str}, and solid core capabilities justify this top-tier position.",
            f"Ranks highly because of exceptional semantic alignment ({metric_str}) combined with {exp_str}.",
            f"Earned a top position due to high technical alignment ({metric_str}) and a proven record spanning {exp_str}."
        ]
        sentences.append(openings[v])
        
        if production:
            sentences.append("This background includes verified production-scale engineering experience.")
        if m_skills:
            sentences.append(f"Demonstrated {', '.join(m_skills[:2])} alignment reinforces this standing.")
            
        concerns = [
            "Only minor gaps appear relative to the highest-ranked profiles.",
            "Minor coverage limitations represent the only notable trade-off.",
            "Subtle exposure limitations in secondary areas are the only minor concerns."
        ]
        sentences.append(concerns[v])

    elif rank <= 30:
        # Rank 11-30: Positive tone, one strength, one trade-off
        openings = [
            f"Placed in the upper tier after demonstrating robust capabilities ({metric_str}) across {exp_str}.",
            f"Shows solid evidence of core engineering fundamentals ({metric_str}) along with {exp_str}.",
            f"Good technical alignment with the required technologies ({metric_str}) supports this placement."
        ]
        sentences.append(openings[v])
        
        if production:
            sentences.append("Profile shows evidence of production-scale engineering experience.")
        if m_skills:
            sentences.append(f"Core competencies include verified {', '.join(m_skills[:2])} work.")
            
        trade_offs = [
            "However, some core requirements appear less consistently demonstrated than higher-ranked candidates.",
            "Though some technological requirements show less depth than top-tier profiles.",
            "Although certain specialized domain skills are less pronounced compared to the highest-ranked peers."
        ]
        sentences.append(trade_offs[v])
        if miss_skills:
            sentences.append(f"Some exposure gaps remain around {miss_skills[0]}.")

    elif rank <= 60:
        # Rank 31-60: Balanced tone, strengths, noticeable gaps
        openings = [
            f"Demonstrates solid engineering experience and moderate semantic similarity ({metric_str}) with the required role.",
            f"Demonstrates moderate alignment with the core job requirements ({metric_str}) over {exp_str}.",
            f"Shows a balanced technical baseline across key dimensions ({metric_str}) backed by {exp_str}."
        ]
        sentences.append(openings[v])
        
        if production:
            sentences.append("Includes relevant production-scale engineering experience.")
        if m_skills:
            sentences.append(f"Background features some {', '.join(m_skills[:2])} exposure.")
            
        gaps = [
            "It shows fewer signals across mandatory requirements compared with higher-ranked candidates.",
            "Noticeable gaps remain around core requirements, limiting upward mobility in the rank list.",
            "Clear exposure limitations in core areas differentiate this profile from higher-tier candidates."
        ]
        sentences.append(gaps[v])
        if miss_skills:
            sentences.append(f"For instance, a specific exposure gap remains around {miss_skills[0]}.")

    else:
        # Rank 61-100: Conservative tone, entry rationale, limitations
        openings = [
            f"Included in the shortlist because of reasonable overall similarity ({metric_str}) and relevant engineering experience.",
            f"Included near the end of the shortlist based on acceptable experience ({exp_str}) and baseline metrics ({metric_str}).",
            f"Positioned in the lower tier due to weaker semantic alignment with the JD ({metric_str}), though overall background remains valid."
        ]
        sentences.append(openings[v])
        
        if production:
            sentences.append("Features some production-scale engineering experience.")
            
        limitations = [
            "Weaker job description coverage and lower similarity metrics limit overall confidence relative to higher-ranked profiles.",
            "Additional technical validation would be required due to significant gaps compared to higher-ranked profiles.",
            "Substantial coverage gaps relative to the role requirements necessitate deeper evaluation before further processing."
        ]
        sentences.append(limitations[v])
        if miss_skills:
            sentences.append(f"Gaps include a lack of explicit coverage for {miss_skills[0]}.")

    # 6. Assembly and polish clamp cleanup
    reason_text = " ".join([s.strip() for s in sentences if s.strip()])
    
    # Clean up any potential double period artifacts safely
    reason_text = reason_text.replace("..", ".").strip()

    return {"reason": reason_text}
