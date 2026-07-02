"""Module for orchestrating the first-pass candidate ranking pipeline."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import json

from src.first_pass.embed_jd import memory_service
from src.first_pass import credibility, experience_score, signals, similarity, skills, score_fusion
from src.first_pass.weights import TOP_K_FIRST_PASS

logger = logging.getLogger(__name__)


class FirstPassPipeline:
    """Orchestrates the first-pass candidate filtering and scoring pipeline."""
    @staticmethod
    def _safe_json_load(value: Any, default: Any):
        """Safely parse JSON strings."""
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return default
        return value if value is not None else default

    @staticmethod
    def _build_semantic_document(
        candidate: Dict[str, Any],
        parsed_skills: List[Dict[str, Any]],
        parsed_signals: Dict[str, Any],
    ) -> str:
        """Build a natural-language representation of a candidate."""

        skill_names = [
            skill.get("name", "")
            for skill in parsed_skills
            if isinstance(skill, dict)
        ]

        return f"""
Current Title: {candidate.get("current_title", "")}

Current Company: {candidate.get("current_company", "")}

Years of Experience: {candidate.get("years_experience", 0)}

Location: {candidate.get("location", "")}

Skills:
{", ".join(skill_names)}

Open to Work: {parsed_signals.get("open_to_work_flag", False)}

Preferred Work Mode: {parsed_signals.get("preferred_work_mode", "")}

Notice Period:
{parsed_signals.get("notice_period_days", "")} days

Recruiter Response Rate:
{parsed_signals.get("recruiter_response_rate", "")}
""".strip()
    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        """Initializes the pipeline and loads offline data artifacts once."""
        logger.info("Initializing FirstPassPipeline and loading artifacts...")
        self.artifacts_path = Path(artifacts_dir)

        # Load data tables once during initialization
        self.feature_table = pd.read_parquet(self.artifacts_path / "feature_table.parquet")
        self.penalty_metadata = pd.read_parquet(self.artifacts_path / "penalty_metadata.parquet")
        
        # Merge penalty metadata ahead of time for efficient lookup
        if "candidate_id" in self.feature_table.columns and "candidate_id" in self.penalty_metadata.columns:
            self.merged_df = pd.merge(self.feature_table, self.penalty_metadata, on="candidate_id", how="left")
        else:
            self.merged_df = self.feature_table
            
        if "penalty_score" not in self.merged_df.columns:
            self.merged_df["penalty_score"] = 0.0

        logger.info("FirstPassPipeline initialized successfully with %d candidates.", len(self.merged_df))

    def run(self, jd_text: str) -> List[Dict[str, Any]]:
        """Executes the complete first-pass ranking flow for the given job description text.

        Args:
            jd_text: The raw text of the job description.

        Returns:
            A list containing up to the top 1000 ranked candidate dictionaries.
        """
        if not jd_text or not jd_text.strip():
            logger.warning("Received empty job description text. Returning empty candidate list.")
            return []

        logger.info("Starting first-pass pipeline execution.")

        # 1. Generate the JD embedding using embed_jd service
        memory_service.update_jd_memory(jd_text)
        jd_embedding = memory_service.stored_jd_embedding

        if jd_embedding is None:
            logger.error("Failed to generate or retrieve job description embedding.")
            return []

        # 2 & 3. Compute bulk vector similarities
        embedding_scores = similarity.calculate_embedding_similarity(jd_embedding)
        tfidf_scores = similarity.calculate_tfidf_similarity(jd_text)

        # Bulk compute DataFrame-driven signals to optimize execution speed
        hiring_scores = signals.calculate_hiring_likelihood(self.merged_df)
        credibility_scores = credibility.calculate_credibility(self.merged_df)

        # Placeholder for required skills since extraction happens in other layers
        required_skills: List[str] = []
        candidate_skills_list = self.merged_df["skills"].tolist()
        skill_scores = skills.calculate_skill_overlap(required_skills, candidate_skills_list)

        # 4. Iterate over candidates to gather scores and compute individual metrics
        fusion_input_rows = []
        from typing import cast

        candidates_records = cast(
    List[Dict[str, Any]],
    self.merged_df.to_dict(orient="records")
)

        for idx, candidate in enumerate(candidates_records):
            cand_id = candidate.get("candidate_id")
            
            # Compute scalar experience score
            exp_score = experience_score.calculate_experience_score(candidate, jd_text)

            # Map to fusion-compatible structure
            fusion_input_rows.append({
                "candidate_id": cand_id,
                "embedding_score": float(embedding_scores[idx]),
                "tfidf_score": float(tfidf_scores[idx]),
                "skill_score": float(skill_scores[idx]),
                "experience_score": float(exp_score),
                "hiring_score": float(hiring_scores[idx]),
                "credibility_score": float(credibility_scores[idx]),
                "penalty_score": float(candidate.get("penalty_score", 0.0)),
                "_meta_index": idx  # Store trace link back to original profile metadata
            })

        # 5 & 6. Fuse all signals and sort candidates deterministically
        ranked_fusion_results = score_fusion.rank_candidates(fusion_input_rows)

        # 7. Truncate to top K first-pass candidates
        top_fused_candidates = score_fusion.get_top_candidates(ranked_fusion_results, top_k=TOP_K_FIRST_PASS)

        # Format output structures ensuring all downstream fields are strictly preserved
        final_output = []

        for ranked_cand in top_fused_candidates:
            orig_idx = ranked_cand["_meta_index"]
            orig_record = candidates_records[orig_idx]

            parsed_skills = self._safe_json_load(
                orig_record.get("skills", "[]"),
                [],
            )

            parsed_signals = self._safe_json_load(
                orig_record.get("redrob_signals", "{}"),
                {},
            )

            semantic_document = self._build_semantic_document(
                orig_record,
                parsed_skills,
                parsed_signals,
            )

            final_output.append(
                {
                    "candidate_id": ranked_cand["candidate_id"],
                    "first_pass_score": ranked_cand["final_score"],
                    "embedding_similarity": ranked_cand["embedding_score"],
                    "tfidf_similarity": ranked_cand["tfidf_score"],
                    "experience_score": ranked_cand["experience_score"],
                    "skill_score": ranked_cand["skill_score"],
                    "behavior_score": ranked_cand["hiring_score"],
                    "years_experience": orig_record.get("years_experience"),
                    "skills": parsed_skills,
                    "employment_history": orig_record.get(
                        "employment_history",
                        [],
                    ),
                    "redrob_signals": parsed_signals,
                    "semantic_document": semantic_document,
                }
            )
            logger.info(
            "Pipeline run complete. Returning top %d candidates.",
            len(final_output),
        )

        return final_output