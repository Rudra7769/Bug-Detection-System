"""
Bug-Detection-System Configuration Module
Centralized, configurable settings with zero hardcoded absolute paths.
"""

import os
from pathlib import Path

# Base Paths (Relative to project root)
# Resolves the directory where this file resides (src/) and steps up to the project root
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

# Raw Data Files
ECLIPSE_PLATFORM_RAW = RAW_DATA_DIR / "eclipse_platform.csv"
ECLIPSE_SEVERITY_RAW = RAW_DATA_DIR / "eclipse_severity.csv"

# Processed Data Files
CLEANED_DATA_CSV = PROCESSED_DATA_DIR / "cleaned_bug_reports.csv"
TRAIN_DATA_CSV = PROCESSED_DATA_DIR / "train_bug_reports.csv"
VAL_DATA_CSV = PROCESSED_DATA_DIR / "val_bug_reports.csv"
TEST_DATA_CSV = PROCESSED_DATA_DIR / "test_bug_reports.csv"
SAMPLE_EVAL_CSV = SAMPLE_DATA_DIR / "sample_test_bugs.csv"

# Models and Reports
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Model Artifact Paths
SEVERITY_MODEL_PKL = MODELS_DIR / "severity_classifier.joblib"
TFIDF_VECTORIZER_PKL = MODELS_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PKL = MODELS_DIR / "label_encoder.joblib"
EMBEDDINGS_NPY = MODELS_DIR / "corpus_embeddings.npy"
CORPUS_METADATA_PKL = MODELS_DIR / "corpus_metadata.joblib"

# Pipeline Settings
RANDOM_SEED = 42
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K_DUPLICATES = 5

# Text Cleaning Parameters
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 20000

# Canonical Severity Classes (Bugzilla Standard)
CANONICAL_SEVERITIES = ["trivial", "minor", "normal", "major", "critical", "blocker"]
