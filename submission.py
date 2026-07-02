"""Main orchestration module for the Intelligent Candidate Discovery & Ranking system."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Union

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

    def load_jd(self, jd_input: Union[str, Path]) -> str:
        path = Path(jd_input)

        if path.exists():
            logger.info("Loading JD from %s", path)
            return path.read_text(encoding="utf-8").strip()

        return str(jd_input).strip()

    def run(self, jd_path: Union[str, Path], output_dir: Union[str, Path]) -> None:
        start = time.perf_counter()

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        submission_path = output_dir / "team_xyz.csv"

        jd_text = self.load_jd(jd_path)

        logger.info("Running First Pass Pipeline...")
        first_pass_results = self.first_pass.run(jd_text)

        logger.info("Running Second Pass Pipeline...")
        second_pass_results = self.second_pass.run(
            jd_text=jd_text,
            top_candidates=first_pass_results,
        )

        final_candidates = sorted(
            second_pass_results,
            key=lambda x: (
                -float(x.get("final_score", 0.0)),
                -float(x.get("cross_encoder_score", x.get("final_score", 0.0))),
                -float(x.get("embedding_similarity", x.get("final_score", 0.0))),
                str(x.get("candidate_id", ""))
            )
        )

        if len(final_candidates) < 100:
            logger.warning(
                "Only %d candidates available.",
                len(final_candidates)
            )

        rows = []
        for i, candidate in enumerate(final_candidates[:100], start=1):
            candidate_id = candidate.get("candidate_id")
            score = candidate.get("final_score", 0.0)

            # Robust safe-get evaluation covering potential profile schema variants
            experience = (
                candidate.get("years_experience") 
                or candidate.get("years_of_experience") 
                or candidate.get("profile", {}).get("years_of_experience", 0)
            )

            reasoning_dict = generate_reasoning(candidate)
            reasoning = reasoning_dict.get("reason", "")

            rows.append({
                "candidate_id": candidate_id,
                "rank": i,
                "score": f"{score:.4f}",
                "reasoning": reasoning
            })

        df = pd.DataFrame(rows)
        df.to_csv(
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

    orchestrator = RankingOrchestrator()

    jd_file = Path("artifacts/job_description.txt")
    output_directory = Path("output")
    output_directory.mkdir(parents=True, exist_ok=True)

    orchestrator.run(
        jd_path=jd_file,
        output_dir=output_directory,
    )
