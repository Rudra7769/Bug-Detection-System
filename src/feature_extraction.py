"""
Feature Extraction Module for Bug-Detection-System
Provides:
- Scikit-learn TfidfVectorizer pipeline with unigram/bigram sublinear scaling, fitted strictly on training data.
- Sentence-BERT embedding extraction using sentence-transformers/all-MiniLM-L6-v2 (384-dimensional).
- Automatic GPU / CPU hardware detection.
- Resumable batch generation with checkpointing in models/embeddings_cache/.
- Alignment guarantees between bug IDs, row indices, and embeddings.
"""

import os
import glob
import logging
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from .config import (
    DEFAULT_EMBEDDING_MODEL,
    MODELS_DIR,
    TFIDF_VECTORIZER_PKL,
    RANDOM_SEED
)

logger = logging.getLogger("FeatureExtraction")


class TFIDFFeatureExtractor:
    """
    TF-IDF Vectorizer wrapper with persistence and strict train-only fitting.
    Features:
      - Unigram and bigram extraction (ngram_range=(1, 2))
      - Sublinear TF scaling (1 + log(tf)) to reduce impact of high-frequency words
      - Configurable max_features, min_df, and max_df
      - Technical-token friendly tokenization regex
      - Preservation of sparse matrices for memory efficiency
    """
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 3,
        max_df: float = 0.85,
        sublinear_tf: bool = True
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf

        # Token pattern keeps identifiers, dotted packages, underscores, hyphens
        token_pattern = r'(?u)\b\w[\w\.\-_]+\b'

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            stop_words="english",
            token_pattern=token_pattern
        )
        self.is_fitted = False

    def fit(self, texts: Union[pd.Series, List[str]]):
        """Fits vectorizer strictly on training data."""
        logger.info(f"Fitting TF-IDF Vectorizer on {len(texts):,} training documents...")
        self.vectorizer.fit(texts)
        self.is_fitted = True
        logger.info(f"TF-IDF fitted with vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        return self

    def transform(self, texts: Union[pd.Series, List[str]]):
        """Transforms documents to sparse TF-IDF matrix."""
        if not self.is_fitted:
            raise ValueError("TF-IDF vectorizer has not been fitted yet! Call fit() on training data first.")
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: Union[pd.Series, List[str]]):
        """Fits strictly on training documents and transforms."""
        self.fit(texts)
        return self.transform(texts)

    def save(self, output_path: Optional[Path] = None):
        """Saves fitted vectorizer to disk."""
        path = output_path or TFIDF_VECTORIZER_PKL
        os.makedirs(path.parent, exist_ok=True)
        joblib.dump(self.vectorizer, path)
        logger.info(f"Saved TF-IDF vectorizer to {path}")

    def load(self, model_path: Optional[Path] = None):
        """Loads fitted vectorizer from disk."""
        path = model_path or TFIDF_VECTORIZER_PKL
        if not path.exists():
            raise FileNotFoundError(f"TF-IDF model file not found at {path}")
        self.vectorizer = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Loaded TF-IDF vectorizer from {path}")
        return self


class ResumableSentenceBERTExtractor:
    """
    Sentence Transformers SBERT (all-MiniLM-L6-v2) embedding generator.
    Features:
      - 384-dimensional dense semantic vectors.
      - L2 normalization (cosine similarity = dot product).
      - Automatic GPU / CPU hardware detection.
      - Resumable batch generation with disk checkpointing.
      - Preserves bug ID to row-index correspondence.
    """
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        cache_dir: Optional[Path] = None
    ):
        self.model_name = model_name
        self._model = None
        self.cache_dir = cache_dir or (MODELS_DIR / "embeddings_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        if device is None:
            torch_lib = r"C:\Users\ASUS\AppData\Local\Programs\Python\Python310\lib\site-packages\torch\lib"
            if os.path.exists(torch_lib):
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(torch_lib)
                    except Exception:
                        pass
                if torch_lib not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = torch_lib + ";" + os.environ.get("PATH", "")
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info(f"Initialized ResumableSentenceBERTExtractor using device: {self.device}")

    @property
    def model(self):
        """Lazy loader for SentenceTransformer."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer '{self.model_name}' on {self.device}...")
            # Set offline flags to ensure rapid load from local huggingface cache
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("SentenceTransformer successfully loaded.")
        return self._model

    def encode_corpus_resumable(
        self,
        df: pd.DataFrame,
        text_column: str = "clean_text",
        id_column: str = "bug_id",
        batch_size: int = 256,
        chunk_save_size: int = 5000,
        output_prefix: str = "corpus",
        final_embeddings_path: Optional[Path] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generates normalized embeddings for an entire DataFrame with resume capability.
        Saves chunks of chunk_save_size to cache_dir.
        If interrupted, restarts from the last completed chunk without recomputing.
        """
        final_path = final_embeddings_path or (MODELS_DIR / f"{output_prefix}_embeddings.npy")
        metadata_path = MODELS_DIR / f"{output_prefix}_metadata.csv"

        n_total = len(df)
        logger.info(f"Starting resumable encoding of {n_total:,} records (chunk_size={chunk_save_size})...")

        # Discover already generated chunks
        chunk_files = sorted(glob.glob(str(self.cache_dir / f"{output_prefix}_chunk_*.npy")))
        completed_records = 0
        loaded_chunks = []

        if chunk_files:
            for cf in chunk_files:
                chk = np.load(cf)
                loaded_chunks.append(chk)
                completed_records += len(chk)
            logger.info(f"Found {len(chunk_files)} existing chunk(s). Resuming from index {completed_records:,} / {n_total:,}")

        if completed_records < n_total:
            for start_idx in range(completed_records, n_total, chunk_save_size):
                end_idx = min(start_idx + chunk_save_size, n_total)
                chunk_df = df.iloc[start_idx:end_idx]
                chunk_texts = chunk_df[text_column].astype(str).tolist()

                logger.info(f"Encoding chunk {start_idx:,} to {end_idx:,} ({len(chunk_texts):,} records)...")
                chunk_emb = self.model.encode(
                    chunk_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )

                chunk_file = self.cache_dir / f"{output_prefix}_chunk_{start_idx:07d}_{end_idx:07d}.npy"
                np.save(chunk_file, chunk_emb)
                loaded_chunks.append(chunk_emb)
                logger.info(f"Saved chunk checkpoint to {chunk_file}")

        # Combine all chunks
        all_embeddings = np.vstack(loaded_chunks)
        assert len(all_embeddings) == n_total, f"Embedding count {len(all_embeddings)} != total rows {n_total}"

        # Save final consolidated files
        os.makedirs(final_path.parent, exist_ok=True)
        np.save(final_path, all_embeddings)
        logger.info(f"Successfully saved final embeddings to {final_path} (shape={all_embeddings.shape})")

        # Save corresponding metadata alignment
        meta_df = df[[id_column, "severity", "severity_source", "duplicate_id", "title", "created_time"]].copy()
        meta_df["row_index"] = np.arange(len(meta_df))
        meta_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata alignment to {metadata_path}")

        return all_embeddings, meta_df

    def encode_queries(
        self,
        texts: Union[List[str], pd.Series],
        batch_size: int = 128
    ) -> np.ndarray:
        """Encodes query texts on the fly with unit normalization."""
        text_list = list(texts) if isinstance(texts, pd.Series) else texts
        return self.model.encode(
            text_list,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
