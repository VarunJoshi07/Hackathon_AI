"""Pipeline module for orchestrating the complete second-pass candidate reranking stage.

This module acts strictly as a thin orchestrator, passing data between dedicated
scoring modules, handling exceptions per candidate stage, and returning the top 100 sorted profiles.
"""

import logging
from typing import Any

# Imports from second-pass modules
try:
    from src.second_pass.cross_encoder import calculate_cross_encoder_scores
except ImportError:
    from cross_encoder import calculate_cross_encoder_scores

try:
    from src.second_pass.must_have import calculate_must_have_scores, extract_mandatory_skills
except ImportError:
    from must_have import calculate_must_have_scores, extract_mandatory_skills

try:
    from src.second_pass.career_quality import calculate_career_quality_scores
except ImportError:
    from career_quality import calculate_career_quality_scores

try:
    from src.second_pass.behavioural_score import calculate_behavioral_scores
except ImportError:
    from behavioural_score import calculate_behavioral_scores

try:
    from src.second_pass.disqualifiers import compute_disqualifier_penalties
except ImportError:
    from disqualifiers import compute_disqualifier_penalties

try:
    from src.second_pass.second_pass_score_fusion import fuse_second_pass_scores
except ImportError:
    from second_pass_score_fusion import fuse_second_pass_scores


class SecondPassPipeline:
    """Orchestrates second-pass filtering and scoring."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

    def run(self, jd_text: str, top_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Runs the second-pass candidate scoring and ranking."""
        print("=" * 60)
        print("SECOND PASS RUN CALLED")
        print("=" * 60)
        self.logger.info("Starting Second Pass Reranking Pipeline...")

        # STAGE 1: Candidate Validation
        valid_candidates = []
        print("Received", len(top_candidates), "candidates")
        for candidate in top_candidates:
            if candidate and isinstance(candidate, dict) and "candidate_id" in candidate:
                valid_candidates.append(candidate)

        self.logger.info(
            "After validation: %d candidates",
            len(valid_candidates),
        )
        print("DEBUG: valid_candidates =", len(valid_candidates))

        # STAGE 2: Cross-Encoder Scoring
        try:
            candidate_docs = [candidate.get("semantic_document", "") for candidate in valid_candidates]
            print("About to run CrossEncoder")
            print(len(candidate_docs))
            print(candidate_docs[:2])
            ce_scores = calculate_cross_encoder_scores(jd_text, candidate_docs)
            print(ce_scores[:5])
            print("CrossEncoder finished")
            print("Cross Encoder scores:")
            print(ce_scores[:10])
            for idx, candidate in enumerate(valid_candidates):
                candidate["cross_encoder_score"] = float(ce_scores[idx]) if idx < len(ce_scores) else 0.0
                print(valid_candidates[0]["cross_encoder_score"])
        except Exception as e:
            
             self.logger.exception(e)
             raise
        self.logger.exception("Exception occurred during Cross-Encoder evaluation.")
        for candidate in valid_candidates:
             candidate.setdefault("cross_encoder_score", 0.0)
        

        # STAGE 3: Must-Have Skills Scoring
        try:
            candidate_skills = [candidate.get("skills", []) for candidate in valid_candidates]
            mh_scores = calculate_must_have_scores(jd_text, candidate_skills)
            print("Mandatory skills:")
            print(candidate_skills)
            for idx, candidate in enumerate(valid_candidates):
                candidate["must_have_score"] = float(mh_scores[idx]) if idx < len(mh_scores) else 0.0
        except Exception:
            self.logger.exception("Exception occurred during Must-Have evaluation.")
            for candidate in valid_candidates:
                candidate.setdefault("must_have_score", 0.0)

        # STAGE 4: Career Quality Scoring
        try:
            cq_scores = calculate_career_quality_scores(valid_candidates)
            for idx, candidate in enumerate(valid_candidates):
                candidate["career_quality_score"] = float(cq_scores[idx]) if idx < len(cq_scores) else 0.0
        except Exception:
            self.logger.exception("Exception occurred during Career Quality evaluation.")
            for candidate in valid_candidates:
                candidate.setdefault("career_quality_score", 0.0)

        # STAGE 5: Behavioral Scoring
        try:
            b_scores = calculate_behavioral_scores(valid_candidates)
            for idx, candidate in enumerate(valid_candidates):
                candidate["behavioral_score"] = float(b_scores[idx]) if idx < len(b_scores) else 0.0
        except Exception:
            self.logger.exception("Exception occurred during Behavioral evaluation.")
            for candidate in valid_candidates:
                candidate.setdefault("behavioral_score", 0.0)

        # STAGE 6: Disqualifier Penalties
        try:
            penalties_df = compute_disqualifier_penalties(valid_candidates)
            for idx, candidate in enumerate(valid_candidates):
                penalty_val = (
                    float(penalties_df.iloc[idx]["disqualifier_penalty"])
                    if idx < len(penalties_df)
                    else 0.0
                )
                candidate["penalty"] = penalty_val
                candidate["disqualifier_penalty"] = penalty_val
        except Exception:
            self.logger.exception("Exception occurred during Disqualifier Penalties evaluation.")
            for candidate in valid_candidates:
                candidate.setdefault("penalty", 0.0)
                candidate.setdefault("disqualifier_penalty", 0.0)

        # STAGE 7: Second-pass Score Fusion
        self.logger.info("Running Second-pass Score Fusion matrix stage...")

        try:
            self.logger.info(
                "Before score fusion: %d candidates",
                len(valid_candidates),
            )
            print("DEBUG: Before fusion =", len(valid_candidates))

            fused_candidates = fuse_second_pass_scores(valid_candidates)

            self.logger.info(
                "After score fusion: %d candidates",
                len(fused_candidates),
            )
            print("DEBUG: After fusion =", len(fused_candidates))

        except Exception:
            self.logger.exception("Critical exception inside score fusion module. Executing basic rank sort.")
            for candidate in valid_candidates:
                candidate.setdefault("final_score", candidate.get("cross_encoder_score", 0.0))
            fused_candidates = sorted(
                valid_candidates,
                key=lambda x: x.get("final_score", 0.0),
                reverse=True,
            )

        # STAGE 8: Truncate to Top 100 Highest Scoring candidates
        self.logger.info("Second-pass processing complete. Returning Top-100 results.")
        
        self.logger.info(
            "Returning %d candidates",
            len(fused_candidates[:100]),
        )
        print("DEBUG: Returning =", len(fused_candidates[:100]))

        return fused_candidates[:100]