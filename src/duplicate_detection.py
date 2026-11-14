"""
Duplicate Bug Detection Engine
Features:
- Model A: Lexical Sparse TF-IDF Cosine Similarity
- Model B: Dense Semantic Sentence-BERT Cosine Similarity
- Model C: Hybrid Similarity (w_tfidf * Sim_tfidf + w_sbert * Sim_sbert) with validation tuning
- Temporal candidate cutoff: candidates must have existed strictly before query creation timestamp
- Self-match filtering: query report cannot retrieve itself
- Ranked candidate return with complete metadata breakdown
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp

logger = logging.getLogger("DuplicateDetection")


class DuplicateBugDetector:
    """
    Intelligent Duplicate Bug Report Retrieval & Ranking Engine.
    """
    def __init__(
        self,
        tfidf_extractor=None,
        sbert_extractor=None,
        hybrid_weight_sbert: float = 0.70,
        hybrid_weight_tfidf: float = 0.30
    ):
        self.tfidf_extractor = tfidf_extractor
        self.sbert_extractor = sbert_extractor
        self.hybrid_weight_sbert = hybrid_weight_sbert
        self.hybrid_weight_tfidf = hybrid_weight_tfidf

        self.corpus_df: Optional[pd.DataFrame] = None
        self.corpus_tfidf: Optional[sp.csr_matrix] = None
        self.corpus_embeddings: Optional[np.ndarray] = None
        self.corpus_timestamps: Optional[pd.Series] = None
        self.corpus_bug_ids: Optional[np.ndarray] = None

    def set_corpus(
        self,
        corpus_df: pd.DataFrame,
        corpus_tfidf: Optional[sp.csr_matrix] = None,
        corpus_embeddings: Optional[np.ndarray] = None
    ):
        """
        Loads the indexed historical bug candidate database.
        """
        self.corpus_df = corpus_df.reset_index(drop=True).copy()
        self.corpus_df["bug_id"] = self.corpus_df["bug_id"].astype(int)
        self.corpus_df["created_dt"] = pd.to_datetime(self.corpus_df["created_time"], errors="coerce", utc=True)

        self.corpus_bug_ids = self.corpus_df["bug_id"].to_numpy()
        # Store as int64 nanoseconds for ultra-fast vectorised comparison
        self.corpus_timestamps_ns = self.corpus_df["created_dt"].astype("int64").to_numpy()

        self.corpus_tfidf = corpus_tfidf
        self.corpus_embeddings = corpus_embeddings
        logger.info(f"Indexed corpus set with {len(self.corpus_df):,} historical bug candidates.")
        return self

    def find_top_k_candidates(
        self,
        query_text: str,
        query_bug_id: Optional[int] = None,
        cutoff_timestamp: Optional[Union[str, pd.Timestamp]] = None,
        k: int = 5,
        method: str = "hybrid",
        query_tfidf_vec=None,
        query_embedding=None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves Top-K candidate duplicate reports under temporal & self-exclusion constraints.

        Constraints:
          - If cutoff_timestamp is provided, candidates must have created_dt < cutoff_timestamp.
          - If query_bug_id is provided, candidate bug_id != query_bug_id.

        Methods:
          - 'tfidf': Model A (TF-IDF Cosine Similarity)
          - 'sbert': Model B (SBERT Dense Cosine Similarity)
          - 'hybrid': Model C (Weighted combination)
        """
        if self.corpus_df is None:
            raise ValueError("Corpus not set. Call set_corpus() first.")

        # 1. Temporal & self-exclusion filter mask
        valid_mask = np.ones(len(self.corpus_df), dtype=bool)

        if query_bug_id is not None:
            valid_mask &= (self.corpus_bug_ids != int(query_bug_id))

        if cutoff_timestamp is not None:
            cutoff_dt = pd.to_datetime(cutoff_timestamp, utc=True)
            valid_mask &= (self.corpus_timestamps_ns < cutoff_dt.value)

        candidate_indices = np.where(valid_mask)[0]
        if len(candidate_indices) == 0:
            return []

        # 2. Compute Similarities
        sim_tfidf = None
        sim_sbert = None

        if method in ("tfidf", "hybrid"):
            if query_tfidf_vec is None:
                if self.tfidf_extractor is None:
                    raise ValueError("TF-IDF extractor required.")
                query_tfidf_vec = self.tfidf_extractor.transform([query_text])

            # Cosine similarity on sparse matrix subset
            cand_tfidf = self.corpus_tfidf[candidate_indices]
            sim_tfidf_arr = cand_tfidf.dot(query_tfidf_vec.T).toarray().flatten()

        if method in ("sbert", "hybrid"):
            if query_embedding is None:
                if self.sbert_extractor is None:
                    raise ValueError("SBERT extractor required.")
                query_embedding = self.sbert_extractor.encode_queries([query_text])[0]

            cand_embs = self.corpus_embeddings[candidate_indices]
            # Since vectors are L2-normalized, inner product = cosine similarity
            sim_sbert_arr = np.dot(cand_embs, query_embedding)

        # Combine
        if method == "tfidf":
            final_sims = sim_tfidf_arr
        elif method == "sbert":
            final_sims = sim_sbert_arr
        elif method == "hybrid":
            final_sims = (
                self.hybrid_weight_sbert * sim_sbert_arr +
                self.hybrid_weight_tfidf * sim_tfidf_arr
            )
        else:
            raise ValueError(f"Unknown retrieval method: {method}")

        # Top-K ranking
        top_k_rel_indices = np.argsort(final_sims)[::-1][:k]

        results = []
        for rank, rel_idx in enumerate(top_k_rel_indices, 1):
            global_idx = candidate_indices[rel_idx]
            row = self.corpus_df.iloc[global_idx]
            b_id = int(row["bug_id"])

            s_score = float(sim_sbert_arr[rel_idx]) if sim_sbert is not None or method in ("sbert", "hybrid") else 0.0
            t_score = float(sim_tfidf_arr[rel_idx]) if sim_tfidf is not None or method in ("tfidf", "hybrid") else 0.0
            h_score = float(final_sims[rel_idx])

            results.append({
                "rank": rank,
                "bug_id": b_id,
                "title": row.get("title", ""),
                "description": str(row.get("description", ""))[:250] + "...",
                "severity": row.get("severity", "normal"),
                "priority": row.get("priority", ""),
                "component": row.get("component", "General"),
                "created_time": str(row.get("created_time", "")),
                "tfidf_similarity": round(t_score, 4),
                "sbert_similarity": round(s_score, 4),
                "hybrid_similarity": round(h_score, 4),
                "is_known_duplicate": pd.notna(row.get("duplicate_id")),
                "duplicate_target": str(row.get("duplicate_id")) if pd.notna(row.get("duplicate_id")) else None
            })

        return results
