"""Build offline TF-IDF artifacts from cleaned candidate text from a .jsonl.gz file."""

import gzip
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator, cast

import joblib
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# Adjust these imports according to your exact pipeline architecture
from src.offline.build_skill_vocab import iter_candidates
from src.offline.feature_extractor import extract_features
from src.offline.text_cleaner import clean_text

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path("data") / "candidates.jsonl.gz"
DEFAULT_ARTIFACTS_DIR = Path("artifacts")
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"
MATRIX_FILENAME = "tfidf_matrix.npz"


def collect_clean_documents(candidate_path: Path) -> tuple[list[str], int]:
    """
    Read every candidate, build the semantic document,
    clean it and return all cleaned documents.
    """

    documents: list[str] = []
    total_candidates = 0

    logger.info("Streaming candidates from %s", candidate_path)

    for candidate in tqdm(
        iter_candidates(candidate_path),
        desc="Building documents",
        unit="candidate",
    ):
        features = extract_features(candidate)

        semantic_document = _to_text(
            features.get("semantic_document")
        )

        cleaned = clean_text(semantic_document)

        if cleaned.strip():
            documents.append(cleaned)

        total_candidates += 1

    logger.info(
        "Collected %d non-empty documents from %d candidates",
        len(documents),
        total_candidates,
    )

    return documents, total_candidates
   


def _to_text(value: Any) -> str:
    """Convert a semantic document value to text conservatively."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value

    logger.warning(
        "Expected semantic_document to be str or None, got %s; using empty text",
        type(value).__name__,
    )
    return ""


def create_vectorizer() -> TfidfVectorizer:
    """Create the configured TF-IDF vectorizer."""
    return TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )


def build_tfidf_matrix(
    documents: list[str],
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit a TF-IDF vectorizer and transform all documents exactly once."""
    if not documents:
        msg = "Cannot build TF-IDF artifacts from an empty document collection"
        logger.error(msg)
        raise ValueError(msg)

    vectorizer = create_vectorizer()
    logger.info("Fitting TF-IDF vectorizer on %d documents", len(documents))

    with tqdm(total=1, desc="Fitting TF-IDF") as progress_bar:
        # Fixed: Removed duplicate fit_transform call to optimize performance for large datasets
        matrix = vectorizer.fit_transform(documents)
        progress_bar.update(1)
        matrix = cast(csr_matrix, matrix)

    logger.info(
        "Built TF-IDF matrix with shape=%s and nnz=%d",
        matrix.shape,
        matrix.nnz,
    )

    return vectorizer, matrix


def save_artifacts(
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    artifacts_dir: Path,
) -> tuple[Path, Path]:
    """Save the TF-IDF vectorizer and sparse matrix artifacts."""
    output_dir = Path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorizer_path = output_dir / VECTORIZER_FILENAME
    matrix_path = output_dir / MATRIX_FILENAME

    try:
        joblib.dump(vectorizer, vectorizer_path)
        sparse.save_npz(matrix_path, matrix)
    except OSError:
        logger.exception("Failed to save TF-IDF artifacts to %s", output_dir)
        raise

    logger.info("Saved TF-IDF vectorizer to %s", vectorizer_path)
    logger.info("Saved TF-IDF matrix to %s", matrix_path)
    return vectorizer_path, matrix_path


def main(
    candidate_path: Path = DEFAULT_DATA_PATH,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
) -> None:

    start = time.perf_counter()

    documents, total_candidates = collect_clean_documents(candidate_path)

    vectorizer, matrix = build_tfidf_matrix(documents)

    vectorizer_path, matrix_path = save_artifacts(
        vectorizer,
        matrix,
        artifacts_dir,
    )

    elapsed = time.perf_counter() - start

    logger.info("Finished successfully.")
    logger.info("Candidates processed : %d", total_candidates)
    logger.info("Documents indexed    : %d", len(documents))
    logger.info("Matrix shape         : %s", matrix.shape)
    logger.info("Vectorizer           : %s", vectorizer_path)
    logger.info("Matrix               : %s", matrix_path)
    logger.info("Completed in %.2f seconds", elapsed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()