"""
Unit and Integration Tests for Bug-Detection-System
Run with:
    python -m unittest discover -s tests
"""

import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CANONICAL_SEVERITIES
from src.preprocessing import clean_text, normalize_severity, preprocess_dataframe
from src.feature_extraction import TFIDFFeatureExtractor
from src.duplicate_detection import DuplicateBugDetector
from src.severity_prediction import SeverityClassifier
from src.evaluation import evaluate_duplicate_detection, evaluate_severity_classifier


class TestPreprocessing(unittest.TestCase):
    def test_clean_text_html_and_urls(self):
        raw = "Error occurred at <a href='https://example.com'>link</a>: NullPointerException"
        cleaned = clean_text(raw, preserve_technical_tokens=True)
        self.assertNotIn("<a", cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertIn("nullpointerexception", cleaned)

    def test_clean_text_stack_trace_preservation(self):
        raw = "Crash\n at org.eclipse.ui.internal.Workbench.run(Workbench.java:1234)\nCaused by NullPointer"
        cleaned = clean_text(raw, preserve_technical_tokens=True)
        self.assertIn("at org.eclipse.ui.internal.workbench.run", cleaned)

    def test_normalize_severity(self):
        self.assertEqual(normalize_severity("blocker"), "blocker")
        self.assertEqual(normalize_severity("CRITICAL"), "critical")
        self.assertEqual(normalize_severity("P1"), "blocker")
        self.assertEqual(normalize_severity("P3"), "normal")
        self.assertEqual(normalize_severity("unknown"), "normal")

    def test_preprocess_dataframe(self):
        data = {
            "bug_id": [1, 2, 3],
            "title": ["Crash on start", "Typo in label", "Duplicate bug report"],
            "description": ["Full crash log with NullPointerException", "Minor spelling error in dialog", "Crash on start"],
            "severity": ["blocker", "trivial", "normal"],
            "duplicate_id": [None, None, 1]
        }
        df = pd.DataFrame(data)
        cleaned_df = preprocess_dataframe(df, min_length=5)
        self.assertIn("clean_text", cleaned_df.columns)
        self.assertEqual(len(cleaned_df), 3)


class TestDuplicateDetectionEvaluation(unittest.TestCase):
    def test_hit_at_k_and_mrr_and_map(self):
        ground_truth = {
            "101": ["1", "2"],
            "102": ["3"],
            "103": ["4"]
        }
        predictions = {
            "101": ["1", "5", "2"],
            "102": ["9", "3"],
            "103": ["20", "21", "22"]
        }
        eval_res = evaluate_duplicate_detection(ground_truth, predictions, hit_k_values=[1, 3, 5], recall_k_values=[5, 10])
        self.assertAlmostEqual(eval_res["mrr"], (1.0 + 0.5 + 0.0) / 3, places=2)
        self.assertAlmostEqual(eval_res["hit_at_1"], 1/3, places=2)
        self.assertAlmostEqual(eval_res["hit_at_3"], 2/3, places=2)
        self.assertAlmostEqual(eval_res["recall_at_5"], (1.0 + 1.0 + 0.0) / 3, places=2)
        self.assertIn("map", eval_res)
        self.assertIn("recall_at_10", eval_res)


class TestTFIDFAndSeverityClassifier(unittest.TestCase):
    def test_tfidf_pipeline(self):
        train_texts = [
            "eclipse crashes with nullpointerexception in workbench",
            "ui typo in dialog preferences",
            "cannot connect to team cvs server timeout error",
            "feature request for dark mode in package explorer"
        ]
        train_labels = ["blocker", "trivial", "major", "normal"]

        vectorizer = TFIDFFeatureExtractor(max_features=50, min_df=1)
        X_train = vectorizer.fit_transform(train_texts)
        self.assertEqual(X_train.shape[0], 4)

        clf = SeverityClassifier(classifier_type="logistic_regression")
        clf.train(X_train, train_labels)

        # Predict single
        test_vec = vectorizer.transform(["nullpointerexception crash occurred"])
        res = clf.predict_single(test_vec)
        self.assertIn("predicted_severity", res)
        self.assertIn("confidence", res)

    def test_duplicate_detector_temporal_filtering(self):
        df = pd.DataFrame({
            "bug_id": [201, 202, 203],
            "clean_text": ["crash in core runtime", "dialog font sizing issue", "workspace corrupted error"],
            "created_time": ["2010-01-01 10:00:00", "2010-01-02 10:00:00", "2010-01-03 10:00:00"]
        })
        embs = np.random.randn(3, 384).astype(np.float32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        detector = DuplicateBugDetector()
        detector.set_corpus(df, None, embs)

        results = detector.find_top_k_candidates(
            query_text="runtime crash",
            query_bug_id=203,
            cutoff_timestamp="2010-01-03 00:00:00",
            k=5,
            method="sbert",
            query_embedding=np.random.randn(384).astype(np.float32)
        )
        self.assertEqual(len(results), 2)
        retrieved_ids = [r["bug_id"] for r in results]
        self.assertNotIn(203, retrieved_ids)
        self.assertIn(201, retrieved_ids)
        self.assertIn(202, retrieved_ids)

    def test_saved_model_artifacts_exist(self):
        models_dir = PROJECT_ROOT / "models"
        self.assertTrue((models_dir / "tfidf_vectorizer.pkl").exists())
        self.assertTrue((models_dir / "severity_sbert_svm_verified.pkl").exists())
        self.assertTrue((models_dir / "severity_svm_full.pkl").exists())
        self.assertTrue((models_dir / "bug_embeddings.npy").exists())
        self.assertTrue((models_dir / "bug_metadata.csv").exists())
        self.assertTrue((models_dir / "model_config.json").exists())


if __name__ == "__main__":
    unittest.main()
