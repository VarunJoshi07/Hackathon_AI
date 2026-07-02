"""Module for reranking candidate documents against a job description using a Cross-Encoder.

This module leverages the sentence-transformers CrossEncoder to compute high-quality
relevance scores between a Job Description (JD) text and a list of candidate documents.
It is optimized to act as a second-pass ranker for the top candidates from `rank.py`.

Requirements:
    - sentence-transformers
    - torch
    - numpy
"""

import logging
from typing import Final, Optional
import threading

import numpy as np
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# Constants
MODEL_NAME: Final[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BATCH_SIZE: Final[int] = 32

# Singleton Model and Thread Safety Primitives
MODEL: Optional[CrossEncoder] = None
_MODEL_LOCK: Final[threading.Lock] = threading.Lock()


def _get_device() -> torch.device:
    """Determine the best available hardware accelerator."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cross_encoder() -> CrossEncoder:
    """Retrieves or instantiates the cross-encoder singleton instance safely across threads."""
    global MODEL

    if MODEL is None:
        with _MODEL_LOCK:
            if MODEL is None:
                device = _get_device()
                logger.info(f"Loading CrossEncoder model '{MODEL_NAME}' onto device: {device}")
                MODEL = CrossEncoder(
                    MODEL_NAME,
                    device=str(device)
                )
    return MODEL


def _min_max_scale(scores: np.ndarray) -> np.ndarray:
    """Normalize raw model scores to a safe 0-1 range.

    Handles edge cases where all scores are identical to avoid division by zero.
    """
    min_val = np.min(scores)
    max_val = np.max(scores)

    if np.isclose(min_val, max_val):
        logger.warning("All raw scores are identical; returning uniform normalization values.")
        return np.ones_like(scores, dtype=np.float32)

    return (scores - min_val) / (max_val - min_val)


def calculate_cross_encoder_scores(
    jd_text: str,
    candidate_documents: list[str],
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Computes normalized semantic similarity scores for a list of candidate documents.

    Pairs each document with the job description and performs batched inference using
    the underlying CrossEncoder model.

    Args:
        jd_text: Raw string containing the job description requirements.
        candidate_documents: A list of text representations of candidates.
        batch_size: Number of pairs to process simultaneously during inference.

    Returns:
        A 1D numpy array containing a normalized score (0.0 to 1.0) for each candidate.
    """
    if not candidate_documents:
        logger.warning("Received an empty list of candidate documents. Returning empty array.")
        return np.array([], dtype=np.float32)

    logger.info(f"Preparing cross-encoder inference for {len(candidate_documents)} records.")
    
    # Retrieve the persistent singleton model instance
    model = get_cross_encoder()

    # Formulate inputs matching the semantic cross-encoding layout signature
    pairs = [[jd_text, doc] for doc in candidate_documents]

    # Perform highly optimized batched model predictions
    raw_scores = model.predict(
    pairs,
    batch_size=batch_size,
    show_progress_bar=False,
    convert_to_numpy=True,
)  # type: ignore[arg-type]

    logger.info("Inference completed successfully. Applying Min-Max scaling.")
    return _min_max_scale(raw_scores).astype(np.float32)


def main() -> None:
    """Lightweight smoke test to verify functionality and environment setup."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("Starting cross_encoder.py smoke test.")

    # Mock inputs
    mock_jd = "Looking for a Senior Python Software Engineer with expertise in AWS and PyTorch."
    mock_candidates = [
        "Junior frontend developer specializing in React and CSS layouts.",
        "Senior Python Engineer with 8 years of experience building scalable backends on AWS.",
        "Data Scientist proficient in Python, PyTorch, and machine learning models.",
    ]

    try:
        scores = calculate_cross_encoder_scores(mock_jd, mock_candidates)
        print("\n--- Smoke Test Results ---")
        for i, score in enumerate(scores):
            print(f"Candidate {i}: Normalized Score = {score:.4f}")
        print("--------------------------\n")

        assert len(scores) == len(mock_candidates), "Output shape mismatch!"
        assert (
            scores[1] > scores[0]
        ), "Sanity check failed: Senior Python dev should score higher than Frontend dev."
        logger.info("Smoke test completed successfully!")

    except Exception as e:
        logger.exception("An exception occurred during smoke test validation.")


if __name__ == "__main__":
    main()