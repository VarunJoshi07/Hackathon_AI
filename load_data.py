"""Load candidate records from a JSON Lines file."""


import json
import logging
import sys
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def load_candidates(path: Path) -> list[dict[str, Any]]:
    """Load candidate records from a `.jsonl.gz` file.

    Each non-empty line must contain one JSON object. Empty lines are skipped.

    Args:
        path: Path to the gzipped JSON Lines candidate file.

    Returns:
        A list of candidate dictionaries.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If a non-empty line is invalid JSON or not a JSON object.
        OSError: If the gzip file cannot be opened or read.
    """
    candidate_path = Path(path)
    logger.info("Loading candidates from %s", candidate_path)

    if not candidate_path.exists():
        msg = f"Candidate file not found: {candidate_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    candidates: list[dict[str, Any]] = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                try:
                    record = json.loads(stripped_line)
                except json.JSONDecodeError as exc:
                    msg = (
                        f"Invalid JSON in {candidate_path} at line "
                        f"{line_number}: {exc.msg}"
                    )
                    logger.error(msg)
                    raise ValueError(msg) from exc

                if not isinstance(record, dict):
                    msg = (
                        f"Invalid record in {candidate_path} at line "
                        f"{line_number}: expected a JSON object"
                    )
                    logger.error(msg)
                    raise ValueError(msg)

                candidates.append(record)
    except OSError:
        logger.exception("Failed to read gzip file: %s", candidate_path)
        raise

    logger.info("Loaded %d candidates", len(candidates))
    return candidates


def main(path: Path = Path("data") / "candidates.jsonl") -> None:
    """Load candidates and print the number of records read."""
    candidates = load_candidates(path)
    print(f"Loaded {len(candidates)} candidates")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data") / "candidates.jsonl"
    main(input_path)