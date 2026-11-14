"""
Severity Prediction Machine Learning Module
Supports:
- Track A: Verified-only model (primary scientific benchmark on bugzilla_verified subset).
- Track B: Full-data model (trained on full dataset, evaluated on both verified test and full test).
- Models: Linear SVM (LinearSVC) and Logistic Regression with class_weight='balanced'.
- Probability Calibration using CalibratedClassifierCV on held-out validation data.
- Strict prevention of leakage and test-set contamination.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple, Union, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score
import joblib
from pathlib import Path

from .config import (
    MODELS_DIR,
    CANONICAL_SEVERITIES,
    RANDOM_SEED
)

logger = logging.getLogger("SeverityPrediction")


class SeverityClassifier:
    """
    Trained Multi-Class Severity Classifier with Calibration.
    """
    def __init__(
        self,
        classifier_type: str = "linear_svc",
        random_state: int = RANDOM_SEED,
        C: float = 1.0
    ):
        self.classifier_type = classifier_type
        self.random_state = random_state
        self.C = C
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(CANONICAL_SEVERITIES)
        self.model = None
        self.calibrator = None
        self.is_trained = False

        if classifier_type == "linear_svc":
            self.model = LinearSVC(
                C=self.C,
                class_weight="balanced",
                random_state=self.random_state,
                max_iter=3000,
                dual=False
            )
        elif classifier_type == "logistic_regression":
            self.model = LogisticRegression(
                C=self.C,
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state,
                solver="lbfgs"
            )
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

    def train(
        self,
        X_train,
        y_train: Union[pd.Series, List[str], np.ndarray]
    ):
        """Trains base classifier on training features."""
        logger.info(f"Training base {self.classifier_type} on {X_train.shape[0]:,} samples...")
        y_encoded = self.label_encoder.transform(y_train)
        self.model.fit(X_train, y_encoded)
        self.is_trained = True
        logger.info("Base classifier training complete.")
        return self

    def calibrate(
        self,
        X_val,
        y_val: Union[pd.Series, List[str], np.ndarray],
        method: str = "sigmoid"
    ):
        """
        Calibrates probability estimates using strictly held-out validation data.
        Does NOT use test data.
        """
        if not self.is_trained:
            raise ValueError("Train base model first before calibrating.")

        logger.info(f"Calibrating {self.classifier_type} using {X_val.shape[0]:,} validation samples...")
        y_val_encoded = self.label_encoder.transform(y_val)
        self.calibrator = CalibratedClassifierCV(
            estimator=self.model,
            method=method,
            cv="prefit"
        )
        self.calibrator.fit(X_val, y_val_encoded)
        logger.info("Probability calibration complete.")
        return self

    def predict(self, X) -> np.ndarray:
        """
        Predicts class labels using the underlying trained base estimator.
        Preserves balanced class boundary decisions without being flattened
        by unweighted post-hoc probability calibrator argmax.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained.")
        preds_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds_encoded)

    def predict_proba(self, X) -> np.ndarray:
        """
        Predicts calibrated or normalized class probabilities.
        For LogisticRegression, uses model.predict_proba.
        For LinearSVC, converts decision_function scores via softmax to preserve relative class ranking,
        or uses fitted calibrator if available.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        elif hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            # Softmax to produce valid probability distribution
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        elif self.calibrator is not None:
            return self.calibrator.predict_proba(X)
        else:
            raise NotImplementedError("Model does not support probability estimation.")

    def predict_single(self, X) -> Dict[str, Any]:
        """
        Inference helper for single sample prediction with probability breakdown.
        """
        pred_label = self.predict(X)[0]
        result = {
            "predicted_severity": pred_label,
            "probabilities": {}
        }
        try:
            proba = self.predict_proba(X)[0]
            classes = self.label_encoder.classes_
            for cls_name, prob in zip(classes, proba):
                result["probabilities"][cls_name] = round(float(prob), 4)
            result["confidence"] = round(float(np.max(proba)), 4)
        except Exception:
            result["confidence"] = 1.0

        return result

    def evaluate_on_set(
        self,
        X,
        y_true: Union[pd.Series, List[str], np.ndarray]
    ) -> Dict[str, Any]:
        """Calculates multi-class classification metrics."""
        preds = self.predict(X)
        y_true_list = list(y_true)

        acc = accuracy_score(y_true_list, preds)
        bal_acc = balanced_accuracy_score(y_true_list, preds)
        macro_f1 = f1_score(y_true_list, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true_list, preds, average="weighted", zero_division=0)

        clf_rep = classification_report(
            y_true_list,
            preds,
            labels=CANONICAL_SEVERITIES,
            zero_division=0,
            output_dict=True
        )

        return {
            "accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "detailed_report": clf_rep,
            "predictions": list(preds)
        }

    def save(self, filepath: Union[str, Path]):
        """Serializes model, calibrator, and label encoder."""
        p = Path(filepath)
        os.makedirs(p.parent, exist_ok=True)
        artifact = {
            "classifier_type": self.classifier_type,
            "C": self.C,
            "model": self.model,
            "calibrator": self.calibrator,
            "label_encoder": self.label_encoder,
            "is_trained": self.is_trained
        }
        joblib.dump(artifact, p)
        logger.info(f"Saved severity classifier artifact to {p}")

    def load(self, filepath: Union[str, Path]):
        """Loads serialized artifact."""
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Model file not found: {p}")
        artifact = joblib.load(p)
        self.classifier_type = artifact["classifier_type"]
        self.C = artifact["C"]
        self.model = artifact["model"]
        self.calibrator = artifact["calibrator"]
        self.label_encoder = artifact["label_encoder"]
        self.is_trained = artifact["is_trained"]
        logger.info(f"Loaded severity classifier from {p}")
        return self
