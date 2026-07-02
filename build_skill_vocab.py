"""Build a sorted skill vocabulary artifact from candidate records."""

import gzip
import json
import logging
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.offline.feature_extractor import extract_features


logger = logging.getLogger(__name__)


DEFAULT_DATA_PATH = Path("data") / "candidates.jsonl.gz"
DEFAULT_ARTIFACT_PATH = Path("artifacts") / "skill_vocab.json"


def iter_candidates(path: Path) -> Iterable[dict[str, Any]]:
    """Yield candidate records from a gzipped JSON Lines file.

    Args:
        path: Path to the candidate `.jsonl.gz` file.

    Yields:
        Candidate dictionaries.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If a non-empty line is invalid JSON or not an object.
        OSError: If the gzip file cannot be opened or read.
    """
    candidate_path = Path(path)
    logger.info("Reading candidates from %s", candidate_path)

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

                yield record
    except OSError:
        logger.exception("Failed to read gzip file: %s", candidate_path)
        raise


def normalize_skill(skill: str) -> str:
    """Normalize skill capitalization and spacing."""
    normalized = " ".join(skill.strip().split())
    if not normalized:
        return ""

    known_spellings = {
        "c++": "C++",
        "c#": "C#",
        "llm": "LLM",
        "rag": "RAG",
        "api": "API",
        "fastapi": "FastAPI",
        "pytorch": "PyTorch",
        "tensorflow": "TensorFlow",
        "langchain": "LangChain",
    }
    lower_normalized = normalized.lower()
    if lower_normalized in known_spellings:
        return known_spellings[lower_normalized]

    return normalized.title()


def extract_skill_names(skills: Any) -> set[str]:
    """Extract normalized skill names from a raw skills field."""
    skill_names: set[str] = set()

    if skills is None:
        return skill_names

    if isinstance(skills, str):
        normalized = normalize_skill(skills)
        if normalized:
            skill_names.add(normalized)
        return skill_names

    if isinstance(skills, Mapping):
        for value in skills.values():
            skill_names.update(extract_skill_names(value))
        return skill_names

    if isinstance(skills, Iterable):
        for item in skills:
            if isinstance(item, Mapping):
                for field_name in ("name", "skill", "skill_name", "label"):
                    value = item.get(field_name)
                    if isinstance(value, str):
                        normalized = normalize_skill(value)
                        if normalized:
                            skill_names.add(normalized)
                        break
                else:
                    skill_names.update(extract_skill_names(item.values()))
            else:
                skill_names.update(extract_skill_names(item))

    return skill_names


def build_skill_vocab(candidate_path: Path) -> tuple[list[str], int]:
    """Build a sorted unique skill vocabulary from all candidates."""
    unique_skills: set[str] = set()
    total_candidates = 0

    for candidate in tqdm(iter_candidates(candidate_path), desc="Building skill vocab"):
        features = extract_features(candidate)
        unique_skills.update(extract_skill_names(features.get("skills")))
        total_candidates += 1

    return sorted(unique_skills, key=str.casefold), total_candidates


def save_skill_vocab(skill_vocab: list[str], output_path: Path) -> None:
    """Save the skill vocabulary as formatted JSON."""
    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        artifact_path.write_text(
            json.dumps(skill_vocab, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError:
        logger.exception("Failed to write skill vocabulary to %s", artifact_path)
        raise

    logger.info("Saved skill vocabulary to %s", artifact_path)


def main(
    candidate_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_ARTIFACT_PATH,
) -> None:
    """Build and save the skill vocabulary artifact."""
    start_time = time.perf_counter()
    skill_vocab, total_candidates = build_skill_vocab(candidate_path)
    save_skill_vocab(skill_vocab, output_path)
    elapsed_seconds = time.perf_counter() - start_time

    print(f"Total candidates processed: {total_candidates}")
    print(f"Unique skills: {len(skill_vocab)}")
    print(f"Time taken: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()