"""Conservative text cleaning utilities for candidate ranking features."""

import logging
import re
from typing import Final


logger = logging.getLogger(__name__)


TECHNICAL_TERMS: Final[tuple[str, ...]] = (
    "C++",
    "C#",
    "PyTorch",
    "TensorFlow",
    "LangChain",
    "LLM",
    "RAG",
    "FastAPI",
)

_PLACEHOLDER_PREFIX: Final[str] = "__TECH_TERM_"
_WHITESPACE_PATTERN: Final[re.Pattern[str]] = re.compile(r"\s+")
_REPEATED_PUNCTUATION_PATTERN: Final[re.Pattern[str]] = re.compile(r"([!?.,;:])\1+")


def _protect_technical_terms(text: str) -> tuple[str, dict[str, str]]:
    """Replace protected technical terms with placeholders."""
    protected_text = text
    placeholders: dict[str, str] = {}

    for index, term in enumerate(TECHNICAL_TERMS):
        placeholder = f"{_PLACEHOLDER_PREFIX}{index}__"
        placeholders[placeholder.lower()] = term
        protected_text = re.sub(
            re.escape(term),
            placeholder,
            protected_text,
            flags=re.IGNORECASE,
        )

    return protected_text, placeholders


def _restore_technical_terms(text: str, placeholders: dict[str, str]) -> str:
    """Restore protected technical terms after text normalization."""
    restored_text = text
    for placeholder, term in placeholders.items():
        restored_text = restored_text.replace(placeholder, term)
    return restored_text


def clean_text(text: str | None) -> str:
    """Return conservatively cleaned text.

    The cleaner lowercases general text, replaces newlines with spaces,
    normalizes whitespace, removes repeated punctuation, and preserves
    important technical terms.

    Args:
        text: Input text to clean. ``None`` is treated as empty text.

    Returns:
        Cleaned text.

    Examples:
        >>> clean_text("Senior PyTorch\\nEngineer!!!  FastAPI")
        'senior PyTorch engineer! FastAPI'
        >>> clean_text("C++ / C# / LLM / RAG")
        'C++ / C# / LLM / RAG'
        >>> clean_text("TensorFlow,,,   LangChain??")
        'TensorFlow, LangChain?'
        >>> clean_text(None)
        ''
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        msg = f"Expected text to be str or None, got {type(text).__name__}"
        logger.error(msg)
        raise TypeError(msg)

    protected_text, placeholders = _protect_technical_terms(text)
    cleaned_text = protected_text.replace("\r", " ").replace("\n", " ")
    cleaned_text = cleaned_text.lower()
    cleaned_text = _REPEATED_PUNCTUATION_PATTERN.sub(r"\1", cleaned_text)
    cleaned_text = _WHITESPACE_PATTERN.sub(" ", cleaned_text).strip()

    return _restore_technical_terms(cleaned_text, placeholders)


def main() -> None:
    """Run a small smoke check for the text cleaner."""
    sample_text = "Senior PyTorch\nEngineer!!!  FastAPI"
    print(clean_text(sample_text))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()