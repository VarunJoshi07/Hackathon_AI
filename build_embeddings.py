"""Build offline sentence embeddings for candidate semantic documents."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from build_skill_vocab import iter_candidates
from feature_extractor import extract_features
from offline.text_cleaner import clean_text


logger = logging.getLogger(__name__)


DEFAULT_DATA_PATH = Path("data") / "candidates.jsonl.gz"
DEFAULT_OUTPUT_PATH = Path("artifacts") / "candidate_embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 256


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


def collect_clean_documents(candidate_path: Path) -> tuple[list[str], int]:
    """Collect cleaned semantic documents from all candidates.

    Args:
        candidate_path: Path to the gzipped candidate JSON Lines file.

    Returns:
        A tuple containing cleaned documents and total candidates processed.
    """
    documents: list[str] = []
    total_candidates = 0

    for candidate in tqdm(iter_candidates(candidate_path), desc="Cleaning documents"):
        features = extract_features(candidate)
        documents.append(clean_text(_to_text(features.get("semantic_document"))))
        total_candidates += 1

    return documents, total_candidates


def load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load the sentence-transformers embedding model."""
    logger.info("Loading sentence-transformers model: %s", model_name)
    try:
        return SentenceTransformer(model_name)
    except Exception:
        logger.exception("Failed to load sentence-transformers model: %s", model_name)
        raise


def encode_documents(
    documents: list[str],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> NDArray[np.float32]:
    """Generate embeddings for cleaned semantic documents using batch encoding."""
    if not documents:
        msg = "Cannot build embeddings from an empty document collection"
        logger.error(msg)
        raise ValueError(msg)

    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Encoding %d documents with batch_size=%d", len(documents), batch_size)
    embeddings = model.encode(
        documents,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if not isinstance(embeddings, np.ndarray):
        msg = f"Expected numpy.ndarray embeddings, got {type(embeddings).__name__}"
        logger.error(msg)
        raise TypeError(msg)

    return embeddings.astype(np.float32, copy=False)


def save_embeddings(embeddings: NDArray[np.float32], output_path: Path) -> None:
    """Save candidate embeddings as a NumPy `.npy` artifact."""
    artifact_path = Path(output_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        np.save(artifact_path, embeddings)
    except OSError:
        logger.exception("Failed to save embeddings to %s", artifact_path)
        raise

    logger.info("Saved embeddings to %s", artifact_path)


def main(
    candidate_path: Path = DEFAULT_DATA_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Build and save offline candidate embedding artifacts."""
    start_time = time.perf_counter()

    documents, total_candidates = collect_clean_documents(candidate_path)
    model = load_embedding_model(model_name)
    embeddings = encode_documents(documents, model, batch_size=batch_size)
    save_embeddings(embeddings, output_path)

    elapsed_seconds = time.perf_counter() - start_time
    print(f"Total candidates processed: {total_candidates}")
    print(f"Embedding matrix shape: {embeddings.shape}")
    print(f"Saved: {output_path}")
    print(f"Time taken: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()