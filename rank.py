import logging
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.first_pass.credibility import calculate_credibility
from src.first_pass.experience_score import calculate_experience_score
from src.first_pass.score_fusion import get_top_candidates, rank_candidates
from src.first_pass.signals import calculate_hiring_likelihood
from src.first_pass.similarity import (
    calculate_embedding_similarity,
    calculate_tfidf_similarity,
)
from src.first_pass.skills import calculate_skill_overlap
from src.first_pass.weights import TOP_K_FIRST_PASS

logger = logging.getLogger(__name__)


def run_first_pass_ranking(
    candidates: Sequence[Mapping[str, Any]],
    jd_text: str,
    jd_embedding: np.ndarray,
    required_skills: list[str],
) -> list[dict[str, Any]]:
    """
    Execute the complete first-pass ranking pipeline.
    """

    logger.info("Running first-pass ranking for %d candidates", len(candidates))

    candidate_df = pd.DataFrame(candidates)

    # ---------- Vector scores ----------
    embedding_scores = calculate_embedding_similarity(jd_embedding)

    tfidf_scores = calculate_tfidf_similarity(jd_text)

    skill_scores = calculate_skill_overlap(
        required_skills,
        candidate_df["skills"],
    )

    hiring_scores = calculate_hiring_likelihood(candidate_df)

    credibility_scores = calculate_credibility(candidate_df)

    # ---------- Experience (scalar function) ----------
    experience_scores = np.array(
        [
            calculate_experience_score(candidate, jd_text)
            for candidate in candidates
        ]
    )

    # ---------- Assemble ----------
    feature_rows: list[dict[str, Any]] = []

    for i, candidate in enumerate(candidates):

        feature_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "embedding_score": float(embedding_scores[i]),
                "tfidf_score": float(tfidf_scores[i]),
                "skill_score": float(skill_scores[i]),
                "experience_score": float(experience_scores[i]),
                "hiring_score": float(hiring_scores[i]),
                "credibility_score": float(credibility_scores[i]),
                "penalty_score": 0.0,
            }
        )

    ranked = rank_candidates(feature_rows)

    return get_top_candidates(
        ranked,
        top_k=TOP_K_FIRST_PASS,
    )


def main() -> None:
    def main():
     logger.info("PIPELINE STARTED")
    logging.basicConfig(level=logging.INFO)
    logger.info("rank.py ready")


if __name__ == "__main__":
    main()