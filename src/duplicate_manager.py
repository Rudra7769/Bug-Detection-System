"""
Duplicate Relationship Manager
Features:
- Transitive resolution of duplicate chains to canonical master bug IDs.
- Construction and indexing of duplicate equivalence clusters.
- Verification of canonical master records within dataset.
- Chronological eligibility validation (ensuring target existed prior to query).
- Generation of ground-truth dictionaries for IR retrieval evaluation.
"""

import os
import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import pandas as pd
from pathlib import Path

from .config import REPORTS_DIR

logger = logging.getLogger("DuplicateManager")


class DuplicateRelationshipManager:
    """
    Manages bug duplicate graphs, transitive resolution, and chronological candidate constraints.
    """
    def __init__(self, full_df: pd.DataFrame):
        self.df = full_df.copy()
        self.df["bug_id"] = self.df["bug_id"].astype(int)
        self.df["duplicate_id_clean"] = self.df["duplicate_id"].apply(
            lambda x: int(float(x)) if pd.notna(x) else None
        )
        self.df["created_dt"] = pd.to_datetime(self.df["created_time"], errors="coerce", utc=True)

        self.all_bug_ids: Set[int] = set(self.df["bug_id"])
        self.id_to_created_time: Dict[int, pd.Timestamp] = dict(zip(self.df["bug_id"], self.df["created_dt"]))

        # Directed duplicate edges: bug_id -> duplicate_id_clean
        dup_rows = self.df[self.df["duplicate_id_clean"].notna()]
        self.direct_dup_map: Dict[int, int] = {
            int(b): int(m) for b, m in zip(dup_rows["bug_id"], dup_rows["duplicate_id_clean"])
        }

        # Canonical master mapping and cluster index
        self.canonical_map: Dict[int, int] = {}
        self.canonical_clusters: Dict[int, Set[int]] = defaultdict(set)
        self._build_canonical_clusters()

    def _build_canonical_clusters(self):
        """
        Resolves transitive duplicate relationships (e.g., A -> B -> C => canonical is C)
        and groups bugs into canonical duplicate equivalence clusters.
        """
        logger.info(f"Resolving transitive chains for {len(self.direct_dup_map):,} duplicate reports...")

        def find_canonical(b_id: int, visited: Optional[Set[int]] = None) -> int:
            if visited is None:
                visited = set()
            if b_id not in self.direct_dup_map:
                return b_id
            if b_id in visited:
                # Cycle prevention safeguard
                return b_id
            visited.add(b_id)
            return find_canonical(self.direct_dup_map[b_id], visited)

        for b_id in self.direct_dup_map:
            c_id = find_canonical(b_id)
            self.canonical_map[b_id] = c_id

        # Populate equivalence clusters
        for b_id, c_id in self.canonical_map.items():
            self.canonical_clusters[c_id].add(b_id)
            if c_id in self.all_bug_ids:
                self.canonical_clusters[c_id].add(c_id)

        logger.info(f"Resolved {len(self.direct_dup_map):,} duplicate links into {len(self.canonical_clusters):,} canonical clusters.")

    def get_canonical_master(self, bug_id: int) -> int:
        """Returns the canonical master ID for a given bug ID."""
        return self.canonical_map.get(bug_id, bug_id)

    def get_valid_evaluation_queries(
        self,
        query_df: pd.DataFrame,
        mode: str = "canonical_cluster"
    ) -> Dict[str, List[str]]:
        """
        Extracts valid evaluation queries and their historically available ground-truth targets.

        Rules strictly enforced:
        1. Query report cannot retrieve itself.
        2. Candidate reports must have existed prior to the query's creation timestamp (created_dt < query_created_dt).
        3. At least one valid target must exist in the historical candidate pool.
        4. Excludes queries without historically available targets.

        Modes:
          - 'canonical_cluster': All members of the query's canonical cluster created before query_dt.
          - 'direct_master': Only the directly referenced master bug if created before query_dt.
        """
        q_df = query_df.copy()
        q_df["bug_id"] = q_df["bug_id"].astype(int)
        q_df["duplicate_id_clean"] = q_df["duplicate_id"].apply(
            lambda x: int(float(x)) if pd.notna(x) else None
        )
        q_df["created_dt"] = pd.to_datetime(q_df["created_time"], errors="coerce", utc=True)

        dup_queries = q_df[q_df["duplicate_id_clean"].notna()]
        valid_queries: Dict[str, List[str]] = {}

        for _, row in dup_queries.iterrows():
            q_id = int(row["bug_id"])
            q_time = row["created_dt"]
            direct_master = int(row["duplicate_id_clean"])

            if pd.isna(q_time):
                continue

            if mode == "direct_master":
                if direct_master in self.all_bug_ids and direct_master != q_id:
                    m_time = self.id_to_created_time.get(direct_master)
                    if m_time is not None and m_time < q_time:
                        valid_queries[str(q_id)] = [str(direct_master)]

            elif mode == "canonical_cluster":
                c_id = self.get_canonical_master(q_id)
                cluster_members = self.canonical_clusters.get(c_id, {direct_master})

                # Valid historical targets: present in dataset, not query itself, created before query
                targets = [
                    str(m) for m in cluster_members
                    if m in self.all_bug_ids and m != q_id and self.id_to_created_time.get(m) is not None and self.id_to_created_time[m] < q_time
                ]
                if targets:
                    valid_queries[str(q_id)] = targets

        logger.info(f"Identified {len(valid_queries):,} valid historical evaluation queries (mode='{mode}').")
        return valid_queries

    def generate_relationship_summary(self, output_path: Optional[Path] = None) -> Path:
        """Saves a summary report of duplicate relationships to reports/."""
        out_dir = output_path.parent if output_path else REPORTS_DIR
        out_file = output_path or (out_dir / "duplicate_relationship_summary.json")
        os.makedirs(out_dir, exist_ok=True)

        total_dups = len(self.direct_dup_map)
        missing_masters = sum(1 for m in self.direct_dup_map.values() if m not in self.all_bug_ids)
        transitive_count = sum(1 for m in self.direct_dup_map.values() if m in self.direct_dup_map)
        unique_canonical = len(self.canonical_clusters)

        summary = {
            "total_duplicate_reports": total_dups,
            "direct_masters_present_in_dataset": total_dups - missing_masters,
            "direct_masters_missing_from_dataset": missing_masters,
            "transitive_duplicate_chains": transitive_count,
            "unique_canonical_master_clusters": unique_canonical,
            "sample_clusters": {
                str(c): [int(m) for m in list(members)[:5]]
                for c, members in list(self.canonical_clusters.items())[:5]
            }
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved duplicate relationship summary to {out_file}")
        return out_file
