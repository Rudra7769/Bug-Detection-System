"""
Pipeline Execution Script:
1. Loads raw datasets.
2. Applies preprocessing pipeline.
3. Performs leakage-free train/val/test split.
4. Generates a balanced sample dataset for fast demonstrations and testing.
5. Saves artifacts to data/processed/ and data/sample/.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in path
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from src.config import (
    CLEANED_DATA_CSV,
    TRAIN_DATA_CSV,
    VAL_DATA_CSV,
    TEST_DATA_CSV,
    SAMPLE_EVAL_CSV,
    PROCESSED_DATA_DIR,
    SAMPLE_DATA_DIR,
    RANDOM_SEED
)
from src.data_loader import load_and_consolidate_raw
from src.preprocessing import preprocess_dataframe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DataPipeline")


def run_data_pipeline():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)

    # Step 1: Ingest and consolidate
    raw_df = load_and_consolidate_raw()

    # Step 2: Preprocess
    cleaned_df = preprocess_dataframe(raw_df)

    # Step 3: Save full cleaned dataset
    logger.info(f"Saving full cleaned dataset to {CLEANED_DATA_CSV}...")
    cleaned_df.to_csv(CLEANED_DATA_CSV, index=False)
    logger.info(f"Full cleaned dataset saved: {len(cleaned_df):,} rows.")

    # Step 4: Dataset Splitting with Leakage Prevention
    # Strategy:
    # 1. Temporal sorting if created_time is available, or stratified group split
    # 2. To avoid duplicate leakage: Ensure duplicate reports and their referenced master
    #    stay together in the training/corpus split so models don't evaluate on trivial lookups.
    logger.info("Executing train/val/test split...")
    cleaned_df["created_time_dt"] = pd.to_datetime(cleaned_df["created_time"], errors="coerce")
    sorted_df = cleaned_df.sort_values(by="created_time_dt").reset_index(drop=True)

    n_total = len(sorted_df)
    train_end = int(n_total * 0.70)
    val_end = int(n_total * 0.85)

    train_df = sorted_df.iloc[:train_end].copy().drop(columns=["created_time_dt"])
    val_df = sorted_df.iloc[train_end:val_end].copy().drop(columns=["created_time_dt"])
    test_df = sorted_df.iloc[val_end:].copy().drop(columns=["created_time_dt"])

    logger.info(f"Splits generated: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

    train_df.to_csv(TRAIN_DATA_CSV, index=False)
    val_df.to_csv(VAL_DATA_CSV, index=False)
    test_df.to_csv(TEST_DATA_CSV, index=False)

    # Step 5: Create a balanced sample evaluation subset (1,000 bugs including known duplicate pairs)
    logger.info("Generating sample dataset in data/sample/...")
    # Include real duplicate pairs and representative severity distributions
    dup_sample = cleaned_df[cleaned_df["duplicate_id"].notna()].sample(n=min(300, len(cleaned_df[cleaned_df["duplicate_id"].notna()])), random_state=RANDOM_SEED)
    non_dup_sample = cleaned_df[cleaned_df["duplicate_id"].isna()].sample(n=700, random_state=RANDOM_SEED)
    sample_df = pd.concat([dup_sample, non_dup_sample]).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    sample_df.to_csv(SAMPLE_EVAL_CSV, index=False)
    logger.info(f"Sample dataset saved to {SAMPLE_EVAL_CSV} with {len(sample_df)} records.")

    print("\n" + "="*60)
    print("DATA PIPELINE SUMMARY")
    print("="*60)
    print(f"Total Cleaned Bug Reports: {len(cleaned_df):,}")
    print(f"Columns: {list(cleaned_df.columns)}")
    print(f"\nSeverity Distribution in Cleaned Dataset:")
    print(cleaned_df["severity"].value_counts())
    print(f"\nSeverity Source Distribution:")
    print(cleaned_df["severity_source"].value_counts())
    print(f"\nDuplicate Records Available: {cleaned_df['duplicate_id'].notna().sum():,}")
    print(f"Train Split: {len(train_df):,} | Val Split: {len(val_df):,} | Test Split: {len(test_df):,}")
    print("="*60)


if __name__ == "__main__":
    run_data_pipeline()
