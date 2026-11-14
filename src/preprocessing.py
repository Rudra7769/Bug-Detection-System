"""
Preprocessing Pipeline for Bug Report Duplicate Detection & Severity Prediction
Features:
- HTML tag and URL removal
- Stack trace / log preservation without losing key method/class identifiers
- Technical token and programming term normalization
- Contraction expansion and whitespace cleanup
- Severity label normalization
- Title + description combination
"""

import re
import html
import logging
from typing import Optional, List, Union
import pandas as pd
import numpy as np

from .config import (
    MIN_TEXT_LENGTH,
    MAX_TEXT_LENGTH,
    CANONICAL_SEVERITIES
)

logger = logging.getLogger("Preprocessing")

# Regex patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')

# Normalize stack traces while keeping error class and source file
# E.g., 'at org.eclipse.ui.internal.Workbench.run(Workbench.java:1234)' -> 'STACK_TRACE org.eclipse.ui.internal.Workbench.run'
STACK_TRACE_PATTERN = re.compile(r'^\s*at\s+([a-zA-Z0-9_$.]+)\s*\([^)]*\)', re.MULTILINE)
EXCEPTION_HEADER_PATTERN = re.compile(r'([a-zA-Z0-9_.]+(?:Exception|Error)):')

# Contraction mapping
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am"
}


def expand_contractions(text: str) -> str:
    """Expands common English contractions."""
    for cont, expanded in CONTRACTIONS.items():
        text = text.replace(cont, expanded)
    return text


def clean_text(
    text: Union[str, float, None],
    preserve_technical_tokens: bool = True,
    lowercase: bool = True
) -> str:
    """
    Cleans raw bug report text (title or description).
    Carefully preserves technical entities:
      - Class names (e.g. NullPointerException, ClassCastException)
      - File names & extensions (.java, .xml, .c, .py, .h)
      - Method names and namespaces
      - Error codes
    """
    if pd.isna(text) or text is None:
        return ""

    text = str(text)

    # 1. Unescape HTML entities (&amp;, &lt;, etc.)
    text = html.unescape(text)

    # 2. Remove HTML tags
    text = HTML_TAG_PATTERN.sub(" ", text)

    # 3. Replace URLs and email addresses with semantic tokens
    text = URL_PATTERN.sub(" URL_TOKEN ", text)
    text = EMAIL_PATTERN.sub(" EMAIL_TOKEN ", text)

    # 4. Handle stack traces gracefully:
    # Instead of deleting lines, standardize: preserve package/method, remove line numbers
    text = STACK_TRACE_PATTERN.sub(r' at \1 ', text)

    # 5. Expand contractions
    text = expand_contractions(text)

    # 6. Normalize punctuation while preserving technical tokens (dots in classnames/extensions, underscores)
    if preserve_technical_tokens:
        # Keep letters, numbers, dots (for classpaths/filenames), underscores, hyphens, and whitespace
        text = re.sub(r'[^\w\s\.\-_]', ' ', text)
        # Avoid orphan dots
        text = re.sub(r'\s+\.\s+', ' ', text)
    else:
        text = re.sub(r'[^\w\s]', ' ', text)

    # 7. Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()

    if lowercase:
        text = text.lower()

    return text


def normalize_severity(severity_str: Union[str, float, None]) -> str:
    """
    Normalizes severity labels to Bugzilla canonical standards:
    ['trivial', 'minor', 'normal', 'major', 'critical', 'blocker']
    """
    if pd.isna(severity_str) or not severity_str:
        return "normal"

    s = str(severity_str).strip().lower()

    mapping = {
        "blocker": "blocker",
        "crit": "critical",
        "critical": "critical",
        "maj": "major",
        "major": "major",
        "norm": "normal",
        "normal": "normal",
        "min": "minor",
        "minor": "minor",
        "triv": "trivial",
        "trivial": "trivial",
        "enhancement": "trivial",
        "p1": "blocker",
        "p2": "critical",
        "p3": "normal",
        "p4": "minor",
        "p5": "trivial"
    }

    return mapping.get(s, "normal")


def preprocess_dataframe(
    df: pd.DataFrame,
    min_length: int = MIN_TEXT_LENGTH,
    max_length: int = MAX_TEXT_LENGTH
) -> pd.DataFrame:
    """
    Executes end-to-end preprocessing pipeline on bug dataframe:
    - Fills missing titles/descriptions
    - Cleans text fields
    - Generates combined 'clean_text' representation (Title + Description)
    - Normalizes severity
    - Removes empty or excessively long reports
    - Drops exact duplicate texts
    - Preserves bug_id, duplicate_id, component, created_time
    """
    logger.info(f"Starting preprocessing on {len(df):,} bug reports...")
    cleaned_df = df.copy()

    # Ensure required columns exist
    for col in ["title", "description", "severity"]:
        if col not in cleaned_df.columns:
            raise ValueError(f"Required column '{col}' missing from input DataFrame.")

    # Fill NA text
    cleaned_df["title"] = cleaned_df["title"].fillna("")
    cleaned_df["description"] = cleaned_df["description"].fillna("")

    # Clean text
    logger.info("Cleaning titles and descriptions...")
    cleaned_df["clean_title"] = cleaned_df["title"].apply(lambda t: clean_text(t, preserve_technical_tokens=True))
    cleaned_df["clean_description"] = cleaned_df["description"].apply(lambda d: clean_text(d, preserve_technical_tokens=True))

    # Combine Title and Description
    # Weighting: Title carries dense summary information, so it precedes description
    cleaned_df["clean_text"] = cleaned_df.apply(
        lambda row: f"{row['clean_title']} . {row['clean_description']}".strip()
        if row["clean_title"] and row["clean_description"]
        else (row["clean_title"] or row["clean_description"]),
        axis=1
    )

    # Normalize severity
    cleaned_df["severity"] = cleaned_df["severity"].apply(normalize_severity)

    # Filter out empty or too short text
    initial_count = len(cleaned_df)
    cleaned_df["text_len"] = cleaned_df["clean_text"].str.len()
    cleaned_df = cleaned_df[cleaned_df["text_len"] >= min_length].copy()
    logger.info(f"Filtered {initial_count - len(cleaned_df):,} reports with text length < {min_length}")

    # Truncate extremely long reports to prevent OOM / GPU limits
    cleaned_df["clean_text"] = cleaned_df["clean_text"].str.slice(0, max_length)

    # Remove exact duplicate text reports (preserving primary bug ID)
    prev_len = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset=["clean_text"], keep="first").copy()
    logger.info(f"Removed {prev_len - len(cleaned_df):,} exact text duplicate records.")

    # Set binary duplicate indicator
    cleaned_df["has_duplicate"] = cleaned_df["duplicate_id"].notna().astype(int)

    # Clean up auxiliary columns
    cleaned_df = cleaned_df.drop(columns=["text_len"])

    logger.info(f"Preprocessing completed. Final dataset size: {len(cleaned_df):,} records.")
    return cleaned_df
