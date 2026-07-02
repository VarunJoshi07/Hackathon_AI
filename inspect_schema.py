"""Inspect the first candidate record to understand the input schema."""

import gzip
import json
import logging
import pprint
import sys
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


FIELDS_TO_PRINT: tuple[str, ...] = (
    "candidate_id",
    "semantic_document",
    "skills",
    "experience",
    "employment_history",
    "education",
    "redrob_signals",
    "location",
)


def load_first_candidate(path: Path) -> dict[str, Any]:
    """Load the first non-empty JSON object from a gzipped JSON Lines file.

    Args:
        path: Path to the gzipped JSON Lines candidate file.

    Returns:
        The first candidate record.

    Raises:
        FileNotFoundError: If the candidate file does not exist.
        ValueError: If the first non-empty line is invalid or no record exists.
        OSError: If the gzip file cannot be opened or read.
    """
    candidate_path = Path(path)
    logger.info("Inspecting first candidate from %s", candidate_path)

    if not candidate_path.exists():
        msg = f"Candidate file not found: {candidate_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        with gzip.open(candidate_path, mode="rt", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
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

                return record
    except OSError:
        logger.exception("Failed to read gzip file: %s", candidate_path)
        raise

    msg = f"No candidate records found in {candidate_path}"
    logger.error(msg)
    raise ValueError(msg)


def print_candidate_schema(candidate: dict[str, Any]) -> None:
    """Print selected fields and top-level keys from a candidate record."""
    pretty_printer = pprint.PrettyPrinter(indent=2, sort_dicts=False, width=100)

    print("candidate_id")
    pretty_printer.pprint(candidate.get("candidate_id"))
    print()

    print("all top-level keys")
    pretty_printer.pprint(list(candidate.keys()))
    print()

    for field_name in FIELDS_TO_PRINT[1:]:
        print(field_name)
        pretty_printer.pprint(candidate.get(field_name))
        print()


def main(path: Path = Path("data") / "candidates.jsonl.gz") -> None:
    """Load and print schema information for the first candidate."""
    candidate = load_first_candidate(path)
    print_candidate_schema(candidate)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data") / "candidates.jsonl.gz"
    main(input_path)