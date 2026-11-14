"""
Data Loader Module for Bug-Detection-System
Handles downloading, ingesting, and consolidating real-world Bugzilla bug reports.
"""

import os
import io
import zipfile
import logging
import urllib.request
from typing import Optional, Tuple
import pandas as pd
from pathlib import Path

from .config import (
    RAW_DATA_DIR,
    ECLIPSE_PLATFORM_RAW,
    ECLIPSE_SEVERITY_RAW,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DataLoader")

BUGHUB_ECLIPSE_URL = "https://raw.githubusercontent.com/logpai/bughub/master/EclipsePlatform/eclipse_platform.zip"
BUGZILLA_SEVERITY_URL = "https://huggingface.co/datasets/AliArshad/Bugzilla_Eclipse_Bug_Reports_Dataset/raw/main/filtered_data.xlsx%20-%20Sheet1.csv"


def download_raw_datasets(
    raw_dir: Optional[Path] = None,
    force_download: bool = False
) -> Tuple[Path, Path]:
    """
    Downloads raw Eclipse Bugzilla datasets from verified public sources if not already present.
    """
    dest_dir = raw_dir or RAW_DATA_DIR
    os.makedirs(dest_dir, exist_ok=True)

    platform_path = dest_dir / "eclipse_platform.csv"
    severity_path = dest_dir / "eclipse_severity.csv"

    # 1. Download Bughub Eclipse Platform Archive (contains duplicate relationships, priority, full text)
    if force_download or not platform_path.exists():
        logger.info(f"Downloading Bughub EclipsePlatform dataset from {BUGHUB_ECLIPSE_URL}...")
        try:
            req = urllib.request.Request(BUGHUB_ECLIPSE_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                zf = zipfile.ZipFile(io.BytesIO(resp.read()))
                zf.extract("eclipse_platform.csv", dest_dir)
            logger.info(f"Successfully extracted {platform_path} ({os.path.getsize(platform_path)} bytes)")
        except Exception as e:
            logger.error(f"Failed to download Eclipse Platform dataset: {e}")
            raise
    else:
        logger.info(f"Dataset already exists: {platform_path}")

    # 2. Download Curated Bugzilla Eclipse Severity Dataset
    if force_download or not severity_path.exists():
        logger.info(f"Downloading Curated Bugzilla Severity dataset from {BUGZILLA_SEVERITY_URL}...")
        try:
            req = urllib.request.Request(BUGZILLA_SEVERITY_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                content = resp.read()
                with open(severity_path, "wb") as f:
                    f.write(content)
            logger.info(f"Successfully saved {severity_path} ({os.path.getsize(severity_path)} bytes)")
        except Exception as e:
            logger.error(f"Failed to download Bugzilla Severity dataset: {e}")
            raise
    else:
        logger.info(f"Dataset already exists: {severity_path}")

    return platform_path, severity_path


def load_and_consolidate_raw(
    platform_path: Optional[Path] = None,
    severity_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Loads raw CSVs, merges verified Bugzilla severity labels with full textual reports and duplicate relations.
    Preserves:
      - Bug ID
      - Title / Summary
      - Description
      - Severity (Direct Bugzilla label where available, mapped Priority fallback where noted)
      - Priority
      - Duplicated_issue (Original duplicate relationship)
      - Component
      - Created_time
      - Status / Resolution
    """
    p_path = platform_path or ECLIPSE_PLATFORM_RAW
    s_path = severity_path or ECLIPSE_SEVERITY_RAW

    if not p_path.exists() or not s_path.exists():
        logger.info("Raw files missing, initiating download...")
        download_raw_datasets()

    logger.info(f"Reading platform reports from {p_path}...")
    df_plat = pd.read_csv(p_path, low_memory=False)
    logger.info(f"Loaded {len(df_plat):,} raw Eclipse platform reports.")

    logger.info(f"Reading severity labels from {s_path}...")
    df_sev = pd.read_csv(s_path, low_memory=False)
    logger.info(f"Loaded {len(df_sev):,} raw severity records.")

    # Standardize column naming
    df_plat = df_plat.rename(columns={
        "Issue_id": "bug_id",
        "Title": "title",
        "Description": "description",
        "Duplicated_issue": "duplicate_id",
        "Priority": "priority",
        "Component": "component",
        "Status": "status",
        "Resolution": "resolution",
        "Created_time": "created_time"
    })

    df_sev = df_sev.rename(columns={
        "Bug ID": "bug_id",
        "Severity Label": "severity_direct"
    })

    # Join verified Bugzilla severity labels
    sev_dict = dict(zip(df_sev["bug_id"], df_sev["severity_direct"].astype(str).str.lower().str.strip()))
    df_plat["severity_raw"] = df_plat["bug_id"].map(sev_dict)

    # Establish Bugzilla standard mapping for Priority when direct label is missing
    priority_to_severity = {
        "P1": "blocker",
        "P2": "critical",
        "P3": "normal",
        "P4": "minor",
        "P5": "trivial"
    }

    df_plat["severity_source"] = df_plat["severity_raw"].apply(
        lambda x: "bugzilla_verified" if pd.notna(x) else "bugzilla_priority_derived"
    )
    df_plat["severity"] = df_plat["severity_raw"].fillna(df_plat["priority"].map(priority_to_severity))

    # Standardize duplicate_id
    def clean_dup_id(val):
        if pd.isna(val) or val == "" or str(val).lower() == "nan" or str(val).lower() == "null":
            return None
        # Handle float representation (e.g. 28126.0 -> "28126")
        try:
            return str(int(float(val)))
        except (ValueError, TypeError):
            val_str = str(val).strip()
            # If multiple separated by ';', take primary or joined string
            parts = [str(int(float(p))) for p in val_str.split(";") if p.strip()]
            return ";".join(parts) if parts else None

    df_plat["duplicate_id"] = df_plat["duplicate_id"].apply(clean_dup_id)

    logger.info(f"Consolidated DataFrame shape: {df_plat.shape}")
    logger.info(f"Verified Severity records: {(df_plat['severity_source'] == 'bugzilla_verified').sum():,}")
    logger.info(f"Duplicate reports count: {df_plat['duplicate_id'].notna().sum():,}")

    return df_plat
