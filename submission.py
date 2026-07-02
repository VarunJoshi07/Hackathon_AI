"""Main orchestration module for the Intelligent Candidate Discovery & Ranking system."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

import pandas as pd

from src.first_pass.pipeline_1 import FirstPassPipeline
from src.second_pass.pipeline_2 import SecondPassPipeline
from src.final_review.reason_generator import generate_reasoning
from src.second_pass.must_have import extract_mandatory_skills

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


class RankingOrchestrator:
    """Runs the complete candidate ranking workflow."""

    def __init__(self) -> None:
        print("Creating FirstPass...", flush=True)
        self.first_pass = FirstPassPipeline()

        print("Creating SecondPass...", flush=True)
        self.second_pass = SecondPassPipeline()

        print("Done creating pipelines.", flush=True)

    def load_jd(self, jd_input: str | Path) -> str:
        """Loads a job description from a file or raw string."""

        path = Path(jd_input)

        if path.exists():
            logger.info("Loading JD from %s", path)
            return path.read_text(encoding="utf-8").strip()

        return str(jd_input).strip()

    def execute(self, jd_input: str | Path) -> None:
        print("Checkpoint 1")
        jd_text = self.load_jd(jd_input)

        print("Checkpoint 2")
        first_pass_results = self.first_pass.run(jd_text)

        print("Checkpoint 3")
        second_pass_results = self.second_pass.run(
        jd_text=jd_text,
        top_candidates=first_pass_results,
)

        print("Checkpoint 4")
        start = time.perf_counter()

        jd_text = self.load_jd(jd_input)

        logger.info("Running First Pass...")
        first_pass_results = self.first_pass.run(jd_text)

        logger.info(
            "First Pass completed with %d candidates.",
            len(first_pass_results),
        )

        logger.info("Running Second Pass...")
        second_pass_results = self.second_pass.run(
            jd_text=jd_text,
            top_candidates=first_pass_results,
        )

        logger.info(
            "Second Pass completed with %d candidates.",
            len(second_pass_results),
        )

        mandatory_skills = extract_mandatory_skills(jd_text)

        jd_constraints = {
            "mandatory_skills": list(mandatory_skills),
            "preferred_skills": [],
            "preferred_locations": [],
        }

        for rank, candidate in enumerate(second_pass_results, start=1):

            candidate["rank"] = rank

            reason = generate_reasoning(
                candidate,
                jd_constraints,
            )

            candidate["Reason"] = reason.get("reason", "")

        final_candidates = second_pass_results[:100]

        rows = []

        for candidate in final_candidates:

            rows.append(
                {
                    "Rank": candidate["rank"],
                    "Candidate_ID": candidate["candidate_id"],
                    "Final_Score": candidate["final_score"],
                    "Reason": candidate.get("Reason", ""),
                    "Matched_Skills": json.dumps(
                        candidate.get("matched_skills", [])
                    ),
                    "Missing_Skills": json.dumps(
                        candidate.get("missing_skills", [])
                    ),
                    "Years_Experience": candidate.get(
                        "years_experience"
                    ),
                    "Embedding_Similarity": candidate.get(
                        "embedding_similarity", 0.0
                    ),
                    "Cross_Encoder_Score": candidate.get(
                        "cross_encoder_score", 0.0
                    ),
                    "Production_Background": candidate.get(
                        "production_background", False
                    ),
                    "Behavior_Score": candidate.get(
                        "behavior_score", 0.0
                    ),
                    "Career_Score": candidate.get(
                        "career_score", 0.0
                    ),
                }
            )

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        submission_path = output_dir / "Submission.csv"

        pd.DataFrame(rows).to_csv(
            submission_path,
            index=False,
        )

        runtime = time.perf_counter() - start

        metadata = {
            "runtime_seconds": round(runtime, 2),
            "first_pass_candidates": len(first_pass_results),
            "second_pass_candidates": len(second_pass_results),
            "submission_candidates": len(final_candidates),
            "generated_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        with open(
            output_dir / "run_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=4)

        print("=" * 60)
        print("Candidate Ranking Complete")
        print("=" * 60)
        print(f"Candidates after First Pass : {len(first_pass_results)}")
        print(f"Candidates after Second Pass: {len(second_pass_results)}")
        print(f"Submission Candidates       : {len(final_candidates)}")
        print(f"Submission File             : {submission_path}")
        print(f"Total Runtime               : {runtime:.2f} seconds")
        print("=" * 60)


if __name__ == "__main__":

    print("MAIN STARTED", flush=True)

    orchestrator = RankingOrchestrator()

    print("ORCHESTRATOR CREATED", flush=True)

    jd_file = Path("artifacts/job_description.txt")

    print("JD EXISTS:", jd_file.exists(), flush=True)

    if jd_file.exists():
        orchestrator.execute(jd_file)
    else:
        orchestrator.execute(
            "Senior AI Engineer with Python, LLMs, AWS, Docker and 5+ years experience."
        )