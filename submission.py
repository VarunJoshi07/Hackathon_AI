"""Main orchestration module for the Intelligent Candidate Discovery & Ranking system."""

from datetime import datetime, timezone
import logging
from pathlib import Path
import time

import pandas as pd

from src.first_pass.pipeline_1 import FirstPassPipeline
from src.second_pass.pipeline_2 import SecondPassPipeline
from src.final_review.reason_generator import generate_reasoning

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

        for rank, candidate in enumerate(second_pass_results, start=1):
            candidate["rank"] = rank

            reason = generate_reasoning(candidate, {})
            candidate["Reason"] = reason.get("reason", "")

        final_candidates = second_pass_results[:100]

        rows = []
        for candidate in final_candidates:
            rows.append({
                "Rank": candidate["rank"],
                "Candidate_ID": candidate["candidate_id"],
                "Final_Score": round(candidate["final_score"], 3),
                "Reason": candidate.get("Reason", "")
            })

        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)

        submission_path = output_dir / "Submission.csv"

        # header=False prevents columns text from appearing in the first row
        pd.DataFrame(rows).to_csv(
            submission_path,
            index=False,
            header=False,
        )

        runtime = time.perf_counter() - start

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
    orchestrator = RankingOrchestrator()

    jd_file = Path("artifacts/job_description.txt")

    if jd_file.exists():
        orchestrator.execute(jd_file)
    else:
        orchestrator.execute(
            "Senior AI Engineer with Python, LLMs, AWS, Docker and 5+ years experience."
        )
