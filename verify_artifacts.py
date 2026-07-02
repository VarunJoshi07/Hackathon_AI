"""Verify offline preprocessing artifacts before ranking."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy import sparse
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer


logger = logging.getLogger(__name__)


DEFAULT_ARTIFACTS_DIR = Path("artifacts")
TFIDF_VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
TFIDF_MATRIX_FILENAME = "tfidf_matrix.npz"
FEATURE_TABLE_FILENAME = "feature_table.parquet"
PENALTY_METADATA_FILENAME = "penalty_metadata.parquet"
SKILL_VOCAB_FILENAME = "skill_vocab.json"


@dataclass(frozen=True)
class ArtifactSummary:
    """Summary of verified artifact dimensions."""

    tfidf_matrix_shape: tuple[int, int]
    tfidf_vocabulary_size: int
    feature_table_shape: tuple[int, int]
    penalty_metadata_shape: tuple[int, int]
    skill_vocab_size: int
    candidate_count: int


def _artifact_path(artifacts_dir: Path, filename: str) -> Path:
    """Return a resolved artifact path."""
    return Path(artifacts_dir) / filename


def _require_file(path: Path) -> None:
    """Raise FileNotFoundError if an artifact is missing."""
    if not path.exists():
        msg = f"Required artifact not found: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if not path.is_file():
        msg = f"Artifact path is not a file: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)


def load_tfidf_vectorizer(path: Path) -> TfidfVectorizer:
    """Load and validate the TF-IDF vectorizer artifact."""
    _require_file(path)
    try:
        vectorizer = joblib.load(path)
    except Exception:
        logger.exception("Failed to load TF-IDF vectorizer from %s", path)
        raise

    if not isinstance(vectorizer, TfidfVectorizer):
        msg = (
            f"Invalid TF-IDF vectorizer artifact at {path}: "
            f"expected TfidfVectorizer, got {type(vectorizer).__name__}"
        )
        logger.error(msg)
        raise TypeError(msg)

    return vectorizer


def load_tfidf_matrix(path: Path) -> spmatrix:
    """Load and validate the TF-IDF sparse matrix artifact."""
    _require_file(path)
    try:
        matrix = sparse.load_npz(path)
    except Exception:
        logger.exception("Failed to load TF-IDF matrix from %s", path)
        raise

    if matrix.ndim != 2:
        msg = f"Invalid TF-IDF matrix at {path}: expected 2 dimensions"
        logger.error(msg)
        raise ValueError(msg)

    return matrix


def load_parquet_table(path: Path) -> pd.DataFrame:
    """Load a Parquet table artifact."""
    _require_file(path)
    try:
        return pd.read_parquet(path)
    except ImportError as exc:
        msg = "Reading Parquet requires pyarrow or fastparquet to be installed"
        logger.error(msg)
        raise RuntimeError(msg) from exc
    except Exception:
        logger.exception("Failed to load Parquet artifact from %s", path)
        raise


def load_skill_vocab(path: Path) -> list[str]:
    """Load and validate the skill vocabulary artifact."""
    _require_file(path)
    try:
        raw_vocab: Any = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON in skill vocabulary artifact: {path}"
        logger.error(msg)
        raise ValueError(msg) from exc
    except OSError:
        logger.exception("Failed to read skill vocabulary from %s", path)
        raise

    if not isinstance(raw_vocab, list) or not all(
        isinstance(item, str) for item in raw_vocab
    ):
        msg = f"Invalid skill vocabulary at {path}: expected list[str]"
        logger.error(msg)
        raise TypeError(msg)

    return raw_vocab


def _require_columns(
    dataframe: pd.DataFrame,
    required_columns: set[str],
    artifact_name: str,
) -> None:
    """Validate that a DataFrame contains required columns."""
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        msg = (
            f"{artifact_name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )
        logger.error(msg)
        raise ValueError(msg)


def _candidate_count(dataframe: pd.DataFrame, artifact_name: str) -> int:
    """Return candidate count after validating candidate_id exists."""
    _require_columns(dataframe, {"candidate_id"}, artifact_name)
    return len(dataframe)


def verify_row_counts(
    matrix: spmatrix,
    feature_table: pd.DataFrame,
    penalty_metadata: pd.DataFrame,
) -> int:
    """Verify that candidate row counts match across all candidate artifacts."""
    matrix_rows = matrix.shape[0]
    feature_rows = _candidate_count(feature_table, "feature_table.parquet")
    penalty_rows = _candidate_count(penalty_metadata, "penalty_metadata.parquet")

    row_counts = {
        "tfidf_matrix.npz": matrix_rows,
        "feature_table.parquet": feature_rows,
        "penalty_metadata.parquet": penalty_rows,
    }

    if len(set(row_counts.values())) != 1:
        msg = f"Candidate row counts do not match: {row_counts}"
        logger.error(msg)
        raise ValueError(msg)

    return matrix_rows


def verify_shapes(
    vectorizer: TfidfVectorizer,
    matrix: spmatrix,
    feature_table: pd.DataFrame,
    penalty_metadata: pd.DataFrame,
    skill_vocab: list[str],
) -> ArtifactSummary:
    """Verify artifact shapes and return a summary."""
    if matrix.shape[1] != len(vectorizer.get_feature_names_out()):
        msg = (
            "TF-IDF matrix column count does not match vectorizer "
            f"vocabulary size: matrix={matrix.shape[1]}, "
            f"vectorizer={len(vectorizer.get_feature_names_out())}"
        )
        logger.error(msg)
        raise ValueError(msg)

    candidate_count = verify_row_counts(matrix, feature_table, penalty_metadata)

    if len(skill_vocab) != len(set(skill_vocab)):
        msg = "Skill vocabulary contains duplicate entries"
        logger.error(msg)
        raise ValueError(msg)

    return ArtifactSummary(
        tfidf_matrix_shape=matrix.shape,
        tfidf_vocabulary_size=len(vectorizer.get_feature_names_out()),
        feature_table_shape=feature_table.shape,
        penalty_metadata_shape=penalty_metadata.shape,
        skill_vocab_size=len(skill_vocab),
        candidate_count=candidate_count,
    )


def verify_artifacts(artifacts_dir: Path) -> ArtifactSummary:
    """Load and verify all required offline artifacts."""
    artifact_dir = Path(artifacts_dir)
    logger.info("Verifying artifacts in %s", artifact_dir)

    vectorizer = load_tfidf_vectorizer(
        _artifact_path(artifact_dir, TFIDF_VECTORIZER_FILENAME)
    )
    matrix = load_tfidf_matrix(_artifact_path(artifact_dir, TFIDF_MATRIX_FILENAME))
    feature_table = load_parquet_table(_artifact_path(artifact_dir, FEATURE_TABLE_FILENAME))
    penalty_metadata = load_parquet_table(
        _artifact_path(artifact_dir, PENALTY_METADATA_FILENAME)
    )
    skill_vocab = load_skill_vocab(_artifact_path(artifact_dir, SKILL_VOCAB_FILENAME))

    return verify_shapes(
        vectorizer=vectorizer,
        matrix=matrix,
        feature_table=feature_table,
        penalty_metadata=penalty_metadata,
        skill_vocab=skill_vocab,
    )


def print_summary(summary: ArtifactSummary) -> None:
    """Print artifact verification summary for CLI use."""
    print("Artifact verification summary")
    print(f"Candidate count: {summary.candidate_count}")
    print(f"TF-IDF matrix shape: {summary.tfidf_matrix_shape}")
    print(f"TF-IDF vocabulary size: {summary.tfidf_vocabulary_size}")
    print(f"Feature table shape: {summary.feature_table_shape}")
    print(f"Penalty metadata shape: {summary.penalty_metadata_shape}")
    print(f"Skill vocabulary size: {summary.skill_vocab_size}")
    print("Status: OK")


def main(artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR) -> None:
    """Verify offline artifacts and print a summary."""
    summary = verify_artifacts(artifacts_dir)
    print_summary(summary)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()