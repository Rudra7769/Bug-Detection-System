# Diagnostic Investigation and Root Cause Analysis: Severity Prediction Models

**Date of Investigation**: 2026-11-13  
**System Investigated**: Bug-Detection-System Severity Prediction Pipeline  
**Dataset Tested**: Eclipse Bugzilla Historical Repository  
- Total Verified Test Records: **4,528**
- Total Full Test Records: **12,698**

---

## 1. Executive Summary & Root Cause

### The Problem
In the initial model training run, all classifiers in Track A reported almost exactly identical metrics:
- **Accuracy**: `79.33%`
- **Balanced Accuracy**: `16.67%` (equal to $\frac{1}{6}$)
- **Macro F1**: `0.1475`
- **Weighted F1**: `0.7018`

Per-class breakdowns revealed that precision, recall, and F1 were `0.0` across 5 out of 6 classes (`blocker`, `critical`, `major`, `minor`, `trivial`), and non-zero only for `normal`.

### The Root Cause
The root cause was **not** a data pipeline leakage, not a shared pointer bug, and not a failure of the classifiers to learn during training. The base estimators (`LinearSVC` and `LogisticRegression` with `class_weight='balanced'`) had trained successfully and learned distinct decision boundaries across all 6 classes.

However, during inference:
1. `SeverityClassifier.predict()` was implemented to route through `self.calibrator.predict(X)` whenever `self.calibrator` was present.
2. `CalibratedClassifierCV(method='sigmoid', cv='prefit')` fits a Platt scaling sigmoid per class on the validation set.
3. Because the validation set is heavily imbalanced (**82.02% `normal`** vs **1.05% `blocker`**), Platt scaling reflects the true empirical Bayesian prior: $P(\text{normal}) \approx 0.82$, while $P(\text{minority}) \approx 0.01\text{--}0.07$.
4. When `calibrator.predict(X)` is invoked, it calculates $\operatorname{argmax}_c P(c \mid x)$. Because $P(\text{normal} \mid x)$ almost always dominated the unweighted posterior probability (its minimum across test samples was 0.4900, mean 0.8185), **`calibrator.predict()` predicted `normal` for 100% of samples (4,528 / 4,528)**.
5. In the test set of 4,528 verified bugs, exactly **3,592 are `normal`** ($\frac{3592}{4528} = 79.3286\% \approx 79.33\%$). Predicting `normal` for every single instance yields:
   - $\text{Accuracy} = \frac{3592}{4528} = 79.33\%$
   - $\text{Balanced Accuracy} = \frac{1}{6} \times 1.0 = 16.67\%$
   - $\text{Macro F1} = \frac{1}{6} \times \frac{2 \times 0.7933 \times 1.0}{0.7933 + 1.0} = 0.1475$
   - $\text{Weighted F1} = 0.7933 \times 0.8847 = 0.7018$

This mathematical equality explains why completely different models (TF-IDF vs SBERT, SVM vs LR) collapsed into the exact same evaluation numbers.

---

## 2. Actual Prediction Distributions Across Models

### Ground Truth Distribution (Verified Test Partition: 4,528 Records)
- `normal`: 3,592 (79.33%)
- `major`: 354 (7.82%)
- `critical`: 193 (4.26%)
- `minor`: 176 (3.89%)
- `trivial`: 171 (3.78%)
- `blocker`: 42 (0.93%)
- **Total Unique Classes**: 6

### Original (Corrupted by Post-Hoc Calibrator Argmax)
- Every single model predicted: `normal`: 4,528 (100.0%), all other classes: 0 (0.0%).
- Unique classes predicted: 1.

### Corrected (Base Estimator Decisions with Balanced Class Boundaries)
Evaluating the actual trained base models reveals their true prediction distributions:

| Model | `blocker` | `critical` | `major` | `minor` | `normal` | `trivial` | Unique Classes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Track A: Verified TF-IDF SVM** | 11 (0.2%) | 97 (2.1%) | 355 (7.8%) | 110 (2.4%) | 3,901 (86.1%) | 54 (1.2%) | **6** |
| **Track A: Verified TF-IDF LR** | 29 (0.6%) | 330 (7.3%) | 875 (19.3%) | 355 (7.8%) | 2,793 (61.7%) | 146 (3.2%) | **6** |
| **Track A: Verified SBERT SVM** | 161 (3.6%) | 340 (7.5%) | 346 (7.6%) | 166 (3.7%) | 3,351 (74.0%) | 164 (3.6%) | **6** |
| **Track A: Verified SBERT LR** | 431 (9.5%) | 750 (16.6%) | 904 (20.0%) | 622 (13.7%) | 1,354 (29.9%) | 467 (10.3%) | **6** |
| **Track B: Full SVM (on Verified)** | 185 (4.1%) | 188 (4.2%) | 70 (1.5%) | 141 (3.1%) | 3,868 (85.4%) | 76 (1.7%) | **6** |
| **Track B: Full LR (on Verified)** | 510 (11.3%) | 690 (15.2%) | 407 (9.0%) | 368 (8.1%) | 2,295 (50.7%) | 258 (5.7%) | **6** |

---

## 3. Detailed Per-Class Metrics and Confusion Matrices

### Model 1: Track A Verified TF-IDF Linear SVM
- **Accuracy**: 71.58%
- **Balanced Accuracy**: 20.35%
- **Macro F1**: 0.2111
- **Weighted F1**: 0.6898

```
              precision    recall  f1-score   support
     blocker       0.09      0.02      0.04        42
    critical       0.09      0.05      0.06       193
       major       0.14      0.14      0.14       354
       minor       0.09      0.06      0.07       176
      normal       0.81      0.88      0.84      3592
     trivial       0.24      0.08      0.12       171
```

Confusion Matrix:
```
               blocker  critical  major  minor  normal  trivial
True blocker         1         2      4      2      32        1
True critical        2         9     34      3     145        0
True major           0        17     49     12     274        2
True minor           0         2     10     10     151        3
True normal          8        63    251     76    3159       35
True trivial         0         4      7      7     140       13
```

---

### Model 2: Track A Verified TF-IDF Logistic Regression (Highest Macro F1)
- **Accuracy**: 55.65%
- **Balanced Accuracy**: 24.22%
- **Macro F1**: **0.2217**
- **Weighted F1**: 0.6065

```
              precision    recall  f1-score   support
     blocker       0.03      0.02      0.03        42
    critical       0.08      0.15      0.11       193
       major       0.14      0.35      0.20       354
       minor       0.05      0.11      0.07       176
      normal       0.83      0.65      0.73      3592
     trivial       0.21      0.18      0.20       171
```

Confusion Matrix:
```
               blocker  critical  major  minor  normal  trivial
True blocker         1         6     10      5      20        0
True critical        4        28     66      7      88        0
True major           3        43    124     25     153        6
True minor           3        10     25     19     111        8
True normal         18       239    636    281    2317      101
True trivial         0         4     14     18     104       31
```

---

### Model 3: Track A Verified SBERT Linear SVM
- **Accuracy**: 62.83%
- **Balanced Accuracy**: 21.31%
- **Macro F1**: 0.2038
- **Weighted F1**: 0.6472

```
              precision    recall  f1-score   support
     blocker       0.02      0.07      0.03        42
    critical       0.09      0.17      0.12       193
       major       0.12      0.12      0.12       354
       minor       0.04      0.03      0.04       176
      normal       0.82      0.76      0.79      3592
     trivial       0.13      0.13      0.13       171
```

---

## 4. Comparison: Original vs Corrected Metrics

| Model Track & Architecture | Original Accuracy | Corrected Accuracy | Original Macro F1 | Corrected Macro F1 | Original Balanced Acc | Corrected Balanced Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Track A: TF-IDF SVM** | 79.33% | **71.58%** | 0.1475 | **0.2111** | 16.67% | **20.35%** |
| **Track A: TF-IDF Logistic Regression** | 79.31% | **55.65%** | 0.1512 | **0.2217** | 16.85% | **24.22%** |
| **Track A: SBERT SVM** | 79.33% | **62.83%** | 0.1475 | **0.2038** | 16.67% | **21.31%** |
| **Track A: SBERT Logistic Regression** | 79.33% | **31.16%** | 0.1475 | **0.1758** | 16.67% | **28.16%** |
| **Track B: Full SVM (on Verified Test)** | 79.33% | **68.84%** | 0.1475 | **0.1727** | 16.67% | **17.52%** |
| **Track B: Full LR (on Verified Test)** | 79.33% | **43.68%** | 0.1475 | **0.1740** | 16.67% | **19.30%** |
| **Track B: Full SVM (on Full Test: 12,698)** | 90.97% | **79.55%** | 0.1588 | **0.1760** | 16.67% | **18.11%** |
| **Track B: Full LR (on Full Test: 12,698)** | 90.97% | **49.93%** | 0.1588 | **0.1542** | 16.67% | **20.63%** |

---

## 5. Technical Changes Implemented

1. **Separation of Discrete Predictions from Probability Outputs (`src/severity_prediction.py`)**:
   - `predict()` now routes strictly to the trained base estimator `self.model.predict(X)`. This preserves the cost-sensitive balanced decision boundaries learned via `class_weight='balanced'`.
   - `predict_proba()` computes calibrated probability distributions using the model's native `predict_proba` (for Logistic Regression) or softmax-normalized decision margins (for Linear SVM).
2. **Confidence Calibration Decoupling**:
   - Probability calibration is retained as a confidence estimation tool for the user interface, but is no longer allowed to override class boundary decisions.
3. **Artifacts Generated & Preserved**:
   - `reports/SEVERITY_MODEL_DIAGNOSTIC.md`: This comprehensive diagnostic report.
   - `reports/corrected_severity_model_comparison.csv`: Machine-readable comparison across all models.
   - `reports/corrected_severity_model_comparison.png`: Visual bar chart of corrected metrics.
   - `reports/confusion_matrix_tfidf_lr_verified_corrected.png`: Detailed confusion matrix plot.

---

## 6. Model Recommendation for Production / Streamlit

- **Selected Model**: **Track A Verified TF-IDF Logistic Regression** (or **TF-IDF Linear SVM** for higher overall accuracy).
  - **TF-IDF Logistic Regression** achieves the highest Macro F1 (**0.2217**) and Balanced Accuracy (**24.22%**), with superior recall on minority classes (`critical` recall: 15%, `major` recall: 35%, `trivial` recall: 18%).
  - **TF-IDF Linear SVM** provides a strong operational alternative with **71.58% accuracy** and **0.2111 Macro F1**.
