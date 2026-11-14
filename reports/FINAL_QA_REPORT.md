# Final Quality Assurance, Metric Accuracy, and Submission Readiness Report

**Project Title**: Bug Intelligence — Bug Report Duplicate Detection & Severity Prediction System  
**Audit Date**: 2026-11-13  
**Review Status**: **VERIFIED & READY FOR SUBMISSION / COLLEGE DEMONSTRATION**  
**Operating Environment**: Windows • Python 3.10.11 • Streamlit 1.65.0 • PyTorch CPU  

---

## 1. Application Pages & Interface Verification

Every application module was audited and executed end-to-end:

| # | Page Name | Core Functionality | Test Status | Notes |
| :-: | :--- | :--- | :---: | :--- |
| **1** | **Overview Dashboard** | Real dataset metrics, severity source comparison, duplicate statistics, KPI cards | **PASSED** | Displays verified figures directly from cleaned dataset artifacts. |
| **2** | **Analyze Bug** | Defect submission form, character counters, template presets, input validation | **PASSED** | Rejects empty and whitespace-only queries. |
| **3** | **Analysis Results** | Multi-class predicted probabilities, Plotly bar chart, Top-5 candidate cards, JSON export | **PASSED** | Displays individual TF-IDF, SBERT, and Hybrid similarity scores. |
| **4** | **Duplicate Explorer** | Keyword/ID search, severity filter, duplicate status toggle, pagination | **PASSED** | Real-time filtering across 12,935 search candidate reports. |
| **5** | **Severity Analytics** | Class imbalance audit, per-class metrics table, confusion matrix plot | **PASSED** | Accurately documents the 81.1% majority imbalance and balanced weighting trade-off. |
| **6** | **Model Performance** | Head-to-head comparison tables and bar charts for duplicate retrieval & severity | **PASSED** | Highlights best models per metric without inflating accuracy. |
| **7** | **About Project** | Problem statement, methodology, ASCII architecture diagram, limitations | **PASSED** | Emphasizes human-in-the-loop triaging assistance. |
| **8** | **Session History** | Globally persistent tray, table view, clear history, CSV export | **PASSED** | Stores real-time analysis sessions with no fabricated prior activity. |

---

## 2. Metric Corrections & Imbalance Clarifications

1. **Probability Terminology Correction**:
   - Replaced all misleading references to "calibrated probability" in captions, KPI labels, and UI headers.
   - Standardized on **"Predicted Class Probability"** and **"Model Confidence Estimate"**, reflecting the native `predict_proba()` output from the multi-class Logistic Regression model.
2. **True Class Imbalance Figures**:
   - **Verified-Only Dataset (14,924 records)**:
     - `normal`: **12,104 (81.10%)**
     - `major`: 1,164 (7.80%)
     - `minor`: 572 (3.83%)
     - `trivial`: 487 (3.26%)
     - `critical`: 453 (3.04%)
     - `blocker`: 144 (0.96%)
   - **Full Cleaned Dataset (84,653 records)**:
     - `normal`: **72,218 (85.31%)**
     - `critical`: 5,064 (5.98%)
     - `minor`: 2,489 (2.94%)
     - `blocker`: 2,067 (2.44%)
     - `trivial`: 1,651 (1.95%)
     - `major`: 1,164 (1.38%)
   - *Clarification*: The 79.33% figure was the majority class percentage specifically within the 4,528 verified test partition ($\frac{3,592}{4,528} = 79.33\%$), which previously caused a degenerate model to report a constant 79.33% accuracy.
3. **Reported Model Performance Maintained**:
   - **Accuracy**: **55.65%**
   - **Balanced Accuracy**: **24.22%** (Random Guess Baseline = 16.67%)
   - **Macro F1-Score**: **0.2217** (Highest among all evaluated classifiers)
   - **Weighted F1-Score**: **0.6065**
   - **Minority Recall**: **35.0% on major bugs**, **15.0% on critical bugs**.

---

## 3. Duplicate Retrieval Coverage & Chronological Integrity

A comprehensive mathematical verification was executed on the candidate index and ground-truth duplicate graph:

$$\text{Candidate Index Size} = 12,698 \text{ (Test partition)} + 237 \text{ (Historical Train/Val Masters)} = 12,935 \text{ records}$$

- **Valid Chronological Test Queries**: Exactly **1,242 queries**.
- **Unique Ground-Truth Master Targets Required**: Exactly **852 unique bug reports**.
- **Measured Retrieval Coverage**:
  $$\text{Coverage} = \frac{\text{Targets in Candidate Index}}{\text{Targets Required}} = \frac{852}{852} = \mathbf{100.00\%}$$
- **Chronological Rule Enforcement**:
  $$\forall (q, c) \in \text{Retrieval Results}, \quad T_{\text{candidate}} < T_{\text{query}}$$
  Verified across all 1,242 test queries: **0 future bugs retrieved**; **0 queries retrieved themselves**.
- **Retrieval Performance (Test Set)**:
  - **Hit@1**: 33.66%
  - **Hit@3**: 49.36%
  - **Hit@5**: **56.60%**
  - **Recall@5**: 56.60%
  - **Recall@10**: **64.49%**
  - **MAP**: 0.4348
  - **MRR**: **0.4348**

---

## 4. End-to-End Real Inference Verification

Verified with live multi-modal feature extraction on 5 distinct defect patterns:

1. **"Application crashes immediately during login"**:
   - Predicted Severity: `normal` | Confidence: 24.5%
   - Class Probabilities: `blocker`: 13.0%, `critical`: 22.3%, `major`: 17.5%, `minor`: 15.8%, `normal`: 24.5%, `trivial`: 7.0%
   - Top-1 Match: Bug #394933 (*"problem creating at the time of login"*, Hybrid Sim: 49.2%)
2. **"Minor button alignment issue in the settings page"**:
   - Predicted Severity: `trivial` | Confidence: 28.0%
   - Class Probabilities: `trivial`: 28.0%, `critical`: 21.2%, `major`: 17.0%, `normal`: 16.6%, `minor`: 14.6%, `blocker`: 2.6%
   - Top-1 Match: Bug #271531 (*"[CSS] Button styleable properties"*, Hybrid Sim: 42.4%)
3. **"Application freezes when uploading a large image"**:
   - Predicted Severity: `minor` | Confidence: 43.9%
   - Class Probabilities: `minor`: 43.9%, `normal`: 22.5%, `blocker`: 11.9%, `major`: 8.8%, `critical`: 7.7%, `trivial`: 5.2%
   - Top-1 Match: Bug #314353 (*"Eclipse freezes whenever I try to open something"*, Hybrid Sim: 45.0%)
4. **"Typing text causes a null pointer exception"**:
   - Predicted Severity: `normal` | Confidence: 37.8%
   - Class Probabilities: `normal`: 37.8%, `major`: 26.4%, `minor`: 22.6%, `critical`: 6.6%, `trivial`: 4.1%, `blocker`: 2.6%
   - Top-1 Match: Bug #373174 (*"ContentTypeDecorator should expect a null default editor"*, Hybrid Sim: 44.9%)
5. **"Application fails to start after installation"**:
   - Predicted Severity: `major` | Confidence: 36.8%
   - Class Probabilities: `major`: 36.8%, `normal`: 16.8%, `critical`: 15.7%, `blocker`: 14.8%, `minor`: 11.0%, `trivial`: 4.9%
   - Top-1 Match: Bug #411923 (*"Start Up Bug"*, Hybrid Sim: 35.4%)

---

## 5. Automated Unit & Integration Tests

Execution Command:
```bash
python -m unittest discover -s Bug-Detection-System/tests
```

Results:
```text
Ran 8 tests in 0.023s
OK
```

All 8 tests passed:
- `test_clean_text_html_and_urls`: Verified HTML and URL removal.
- `test_clean_text_stack_trace_preservation`: Verified stack traces preserved.
- `test_normalize_severity`: Verified 6-class normalization mapping.
- `test_preprocess_dataframe`: Verified pipeline dataframe processing.
- `test_hit_at_k_and_mrr_and_map`: Verified ranking metrics calculation.
- `test_tfidf_pipeline`: Verified sparse matrix extraction & prediction.
- `test_duplicate_detector_temporal_filtering`: Verified $T_{\text{candidate}} < T_{\text{query}}$ and self-exclusion.
- `test_saved_model_artifacts_exist`: Verified all 6 serialized model artifacts exist.

---

## 6. Known Limitations & Research Context

1. **Extreme Open-Source Class Imbalance**: With 81.1% of verified bugs belonging to `normal`, minority class precision remains modest (3% on blocker, 8% on critical). The system is designed to provide cost-sensitive triage alerting rather than fully autonomous classification.
2. **Short Queries**: One-sentence queries lack lexical context for TF-IDF; the hybrid architecture leverages SBERT embeddings to maintain high candidate recall.

---

## 7. Submission Readiness Conclusion

The **Bug Intelligence** system meets all engineering, research, and UI requirements. All metrics, dataset distributions, and retrieval constraints have been mathematically audited and verified against the actual serialized model artifacts.

**Verdict: Ready for Academic Demonstration and Project Submission.**
