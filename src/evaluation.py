"""
Evaluation Metrics & Visualization Module
Features:
- Classification report (Accuracy, Precision, Recall, Macro F1, Weighted F1, Per-class metrics)
- Confusion matrix plotting with Seaborn / Matplotlib
- Information Retrieval Duplicate Evaluation:
  - Hit@1, Hit@3, Hit@5
  - Recall@5, Recall@10
  - Mean Average Precision (MAP)
  - Mean Reciprocal Rank (MRR)
  - Explicit multi-relevant duplicate handling and self-match filtering
"""

import os
import logging
from typing import Dict, Any, List, Optional, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from pathlib import Path

from .config import REPORTS_DIR

logger = logging.getLogger("Evaluation")


def evaluate_severity_classifier(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    save_plot: bool = True,
    output_dir: Optional[Path] = None,
    plot_filename: str = "confusion_matrix.png"
) -> Dict[str, Any]:
    """
    Computes rigorous classification metrics and generates confusion matrix plot.
    """
    out_dir = output_dir or REPORTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

    clf_report = classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "detailed_report": clf_report
    }

    if save_plot:
        try:
            target_labels = labels or sorted(list(set(y_true).union(set(y_pred))))
            cm = confusion_matrix(y_true, y_pred, labels=target_labels)

            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=target_labels,
                yticklabels=target_labels
            )
            plt.title("Severity Prediction Confusion Matrix")
            plt.xlabel("Predicted Severity")
            plt.ylabel("Actual Severity")
            plt.tight_layout()

            plot_path = out_dir / plot_filename
            plt.savefig(plot_path, dpi=300)
            plt.close()
            metrics["confusion_matrix_plot"] = str(plot_path)
            logger.info(f"Saved confusion matrix plot to {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate confusion matrix plot: {e}")

    return metrics


def evaluate_duplicate_detection(
    ground_truth_duplicates: Dict[str, List[str]],
    retrieved_predictions: Dict[str, List[str]],
    hit_k_values: List[int] = [1, 3, 5],
    recall_k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    """
    Evaluates duplicate bug retrieval using standard Information Retrieval metrics:
      - Hit@K: Percentage of queries where at least one ground-truth duplicate appears in top-K.
      - Recall@K: Proportion of all relevant duplicates retrieved in top-K (|Relevant intersected with Top-K| / |Relevant|).
      - Mean Average Precision (MAP): Mean of average precisions across queries with multiple relevant duplicates.
      - Mean Reciprocal Rank (MRR): Average of 1 / (rank of first relevant duplicate).

    Multiple Duplicates Handling:
      - When a bug report has multiple ground-truth duplicate targets (e.g. within a duplicate cluster),
        Recall@K rewards retrieving each additional relevant master/duplicate.
      - MAP assesses precision across all true relevant duplicate ranks.
      - Self-matches are strictly excluded prior to ranking.
    """
    total_queries = len(ground_truth_duplicates)
    if total_queries == 0:
        return {
            "total_queries": 0,
            "mrr": 0.0,
            "map": 0.0,
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "recall_at_5": 0.0,
            "recall_at_10": 0.0
        }

    hits = {k: 0 for k in hit_k_values}
    recalls = {k: [] for k in recall_k_values}
    reciprocal_ranks = []
    average_precisions = []

    for q_id, true_dups in ground_truth_duplicates.items():
        # Exclude self-matches from true duplicates
        true_dup_set = set(str(d) for d in true_dups if str(d) != str(q_id))
        if not true_dup_set:
            continue

        # Exclude query ID from predictions if inadvertently present
        preds = [str(p) for p in retrieved_predictions.get(q_id, []) if str(p) != str(q_id)]

        # 1. Reciprocal Rank (Rank of the first relevant duplicate)
        found_first_rank = None
        for rank, p in enumerate(preds, 1):
            if p in true_dup_set:
                found_first_rank = rank
                break

        if found_first_rank is not None:
            reciprocal_ranks.append(1.0 / found_first_rank)
        else:
            reciprocal_ranks.append(0.0)

        # 2. Hit@K (Binary hit: at least one relevant in top-K)
        for k in hit_k_values:
            top_k_preds = set(preds[:k])
            if len(true_dup_set.intersection(top_k_preds)) > 0:
                hits[k] += 1

        # 3. Recall@K (|Relevant in Top-K| / |Total Relevant|)
        for k in recall_k_values:
            top_k_preds = set(preds[:k])
            retrieved_rel = len(true_dup_set.intersection(top_k_preds))
            recalls[k].append(retrieved_rel / len(true_dup_set))

        # 4. Average Precision (AP) for MAP
        num_rel_found = 0
        prec_sum = 0.0
        for rank, p in enumerate(preds, 1):
            if p in true_dup_set:
                num_rel_found += 1
                prec_sum += num_rel_found / rank

        ap = (prec_sum / len(true_dup_set)) if len(true_dup_set) > 0 else 0.0
        average_precisions.append(ap)

    valid_q_count = len(reciprocal_ranks)
    results = {
        "total_queries": valid_q_count,
        "mrr": round(float(np.mean(reciprocal_ranks)), 4) if reciprocal_ranks else 0.0,
        "map": round(float(np.mean(average_precisions)), 4) if average_precisions else 0.0,
    }

    for k in hit_k_values:
        results[f"hit_at_{k}"] = round(float(hits[k] / valid_q_count), 4) if valid_q_count > 0 else 0.0

    for k in recall_k_values:
        results[f"recall_at_{k}"] = round(float(np.mean(recalls[k])), 4) if recalls[k] else 0.0

    logger.info(f"Duplicate Retrieval Metrics: MRR={results['mrr']}, MAP={results['map']}, Hit@1={results.get('hit_at_1')}, Hit@5={results.get('hit_at_5')}, Recall@5={results.get('recall_at_5')}, Recall@10={results.get('recall_at_10')}")
    return results
