"""Build a compact offline feature table from candidate records."""

import json
import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.offline.build_skill_vocab import iter_candidates
from src.offline.feature_extractor import extract_features


logger = logging.getLogger(__name__)


DEFAULT_DATA_PATH = Path("data") / "candidates.jsonl.gz"
DEFAULT_OUTPUT_PATH = Path("artifacts") / "feature_table.parquet"
FEATURE_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "years_experience",
    "skills",
    "num_skills",
    "current_title",
    "current_company",
    "location",
    "text_length",
    "redrob_signals",
)


def _json_dumps(value: Any) -> str:
    """Serialize nested values as stable compact JSON strings."""
    if value is None:
        return ""

    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except TypeError:
        logger.warning(
            "Value of type %s is not JSON serializable; converting to string",
            type(value).__name__,
        )
        return str(value)


def _count_skills(skills: Any) -> int:
    """Count skills without expanding nested data into additional columns."""
    if skills is None:
        return 0

    if isinstance(skills, str):
        return 1 if skills.strip() else 0

    if isinstance(skills, Mapping):
        return sum(_count_skills(value) for value in skills.values())

    if isinstance(skills, Sequence):
        return len(skills)

    return 0


def _coerce_years_experience(value: Any) -> float | None:
    """Convert years of experience to a compact numeric value when possible."""
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int | float):
        return float(value)

    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        try:
            return float(stripped_value)
        except ValueError:
            logger.debug("Could not parse years_experience=%r as float", value)
            return None

    return None


def _coerce_text(value: Any) -> str:
    """Convert scalar display fields to strings while preserving missing values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def feature_row(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Build one feature-table row from a raw candidate record."""
    features = extract_features(candidate)
    semantic_document = features.get("semantic_document")
    semantic_text = semantic_document if isinstance(semantic_document, str) else ""
    skills = features.get("skills")

    return {
        "candidate_id": _coerce_text(features.get("candidate_id")),
        "years_experience": _coerce_years_experience(features.get("years_experience")),
        "skills": _json_dumps(skills),
        "num_skills": _count_skills(skills),
        "current_title": _coerce_text(features.get("current_title")),
        "current_company": _coerce_text(features.get("current_company")),
        "location": _json_dumps(features.get("location")),
        "text_length": len(semantic_text),
        "redrob_signals": _json_dumps(features.get("redrob_signals")),
    }


def build_feature_table(candidate_path: Path) -> pd.DataFrame:
    """Create a pandas DataFrame containing standardized candidate features."""
    rows: list[dict[str, Any]] = []

    for candidate in tqdm(iter_candidates(candidate_path), desc="Building feature table"):
        rows.append(feature_row(candidate))

    dataframe = pd.DataFrame.from_records(rows, columns=FEATURE_COLUMNS)
    return optimize_dataframe(dataframe)


def optimize_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply memory-conscious dtypes to the feature table."""
    optimized = dataframe.copy()

    optimized["years_experience"] = pd.to_numeric(
        optimized["years_experience"],
        errors="coerce",
        downcast="float",
    )
    optimized["num_skills"] = pd.to_numeric(
        optimized["num_skills"],
        errors="coerce",
        downcast="unsigned",
    )
    optimized["text_length"] = pd.to_numeric(
        optimized["text_length"],
        errors="coerce",
        downcast="unsigned",
    )

    string_columns = (
        "candidate_id",
        "skills",
        "current_title",
        "current_company",
        "location",
        "redrob_signals",
    )
    for column in string_columns:
        optimized[column] = optimized[column].astype("string")

    return optimized


def save_feature_table(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save the feature table as a Parquet artifact."""
    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        dataframe.to_parquet(artifact_path, index=False)
    except ImportError as exc:
        msg = "Saving Parquet requires pyarrow or fastparquet to be installed"
        logger.error(msg)
        raise RuntimeError(msg) from exc
    except OSError:
        logger.exception("Failed to save feature table to %s", artifact_path)
        raise

    logger.info("Saved feature table to %s", artifact_path)


def main(
    candidate_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Build and save the offline feature table artifact."""
    start_time = time.perf_counter()
    dataframe = build_feature_table(candidate_path)
    save_feature_table(dataframe, output_path)
    elapsed_seconds = time.perf_counter() - start_time

    print(f"Rows: {len(dataframe)}")
    print(f"Columns: {len(dataframe.columns)}")
    print(f"Saved: {output_path}")
    print(f"Time taken: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()