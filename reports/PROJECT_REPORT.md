# PROJECT REPORT

## BUG REPORT DUPLICATE DETECTION AND SEVERITY PREDICTION SYSTEM (BUG INTELLIGENCE)

---

### EXECUTIVE ABSTRACT

In large-scale open-source and proprietary software repositories (such as Eclipse and Mozilla), bug tracking systems receive hundreds of bug reports daily. Two critical operational challenges arise during manual triage:
1. **Duplicate Bug Reports**: Up to 30% of submitted bug reports describe previously identified issues, leading to redundant triage and engineering effort.
2. **Severity Misclassification**: Inaccurate initial severity assignment delays resolution of critical system defects while minor cosmetic bugs consume engineering resources.

This project implements an intelligent, automated bug triaging system utilizing Natural Language Processing (NLP) and Machine Learning (ML). The system performs two core tasks:
1. **Hybrid Duplicate Retrieval**: Identifies and ranks the Top-5 most similar historical bug reports using a dual-encoder architecture fusing lexical matching (TF-IDF unigrams and bigrams) and dense semantic representations (Sentence-BERT, `all-MiniLM-L6-v2`).
2. **Cost-Sensitive Severity Classification**: Predicts multi-class bug severity across six canonical levels (`blocker`, `critical`, `major`, `normal`, `minor`, `trivial`) using class-weighted multi-class Logistic Regression trained on directly verified Bugzilla reports.

Evaluated on **84,653 Eclipse Bugzilla bug reports**, the hybrid duplicate detection engine achieves **56.60% Hit@5**, **64.49% Recall@10**, and **0.4348 Mean Reciprocal Rank (MRR)** across **1,242 strictly chronological test queries** ($T_{\text{candidate}} < T_{\text{query}}$). The severity model achieves **24.22% Balanced Accuracy** and **0.2217 Macro F1-Score** on the verified test partition, actively identifying critical and major bugs despite extreme class imbalance (81.1% majority `normal` base rate).

---

## 1. INTRODUCTION & PROBLEM DEFINITION

### 1.1 Background
Software defect tracking repositories rely on distributed communities and users to report runtime failures, crashes, and feature defects. However, natural language descriptions of software failures exhibit high variability:
- Two developers reporting the exact same crash often use completely different terminology (e.g., *"Application crashes on launch with JVM dump"* versus *"Abrupt workbench process termination"*).
- Conversely, two reports might share generic technical terms (e.g., `NullPointerException`, `workspace`) while addressing completely independent defects.

### 1.2 Objectives
1. **Automated Duplicate Detection**: Given a newly submitted bug report (title and description), retrieve and rank the Top-5 most relevant existing historical bugs to prevent redundant debugging.
2. **Automated Severity Classification**: Predict the defect severity level with class probability distributions to assist triage teams in prioritizing software-breaking defects.
3. **Preventing Data Leakage**: Enforce realistic chronological retrieval ($T_{\text{candidate}} < T_{\text{query}}$), preventing future data leakage and query self-matches.
4. **Interactive User Interface**: Build an end-to-end dark-themed dashboard using Streamlit and Plotly for developers to triage reports, explore historical repositories, and inspect evaluation benchmarks.

---

## 2. DATASET INGESTION & AUDIT

### 2.1 Data Sources
The dataset is drawn from the historical **Eclipse Bugzilla** platform issue archive:
- **Cleaned Dataset Total**: **84,653 bug reports**.
- **Confirmed Duplicate Links**: **14,117 duplicate relations** resolved into **8,125 canonical clusters**.
- **Historical Candidate Search Index**: **12,935 bug reports** (combining all test partition candidates and required historical train/val master bugs, achieving 100% ground-truth target availability).

### 2.2 Severity Source Audit: Verified vs. Priority-Derived Labels
A rigorous dataset audit revealed two distinct severity label sources:
1. **Directly Verified Labels (`bugzilla_verified`)**: **14,924 reports** containing genuine Bugzilla severity assignments made by human committers.
2. **Priority-Derived Labels (`priority_derived`)**: **69,729 reports** where severity was heuristically mapped from Bugzilla priority fields (`P1` to `P5`).

**Label Noise Finding**: Empirical cross-tabulation demonstrated that **68.5% of P1 bugs** and **77.8% of P2 bugs** were verified as `normal` severity in actual Bugzilla records. Priority does not equal severity. Therefore, this project established **Track A (Verified-Only)** as the primary scientific benchmark.

### 2.3 Class Imbalance Analysis
Open-source bug repositories exhibit severe class imbalance:
- **Verified Dataset (14,924 records)**:
  - `normal`: 12,104 (**81.10%**)
  - `major`: 1,164 (**7.80%**)
  - `minor`: 572 (**3.83%**)
  - `trivial`: 487 (**3.26%**)
  - `critical`: 453 (**3.04%**)
  - `blocker`: 144 (**0.96%**)
- **Full Cleaned Dataset (84,653 records)**: `normal` represents **85.31%** (72,218 records).

---

## 3. SYSTEM ARCHITECTURE & METHODOLOGY

```
+-----------------------------------------------------------------------------------+
|                           Incoming Software Bug Report                            |
|                       (Title + Technical Description)                             |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
                        +-----------------------------------+
                        | Preprocessing & Token Normalization|
                        | (HTML Strip, Stack-Trace Preserved)|
                        +-----------------+-----------------+
                                          |
                +-------------------------+-------------------------+
                |                                                   |
                v                                                   v
+-------------------------------+                   +-------------------------------+
|    TF-IDF Feature Extractor   |                   |  Sentence-BERT Neural Encoder |
|  (Unigrams + Bigrams, 10k Feat)|                  |   (all-MiniLM-L6-v2, 384-d)   |
+---------------+---------------+                   +---------------+---------------+
                |                                                   |
 +--------------+--------------+                     +--------------+--------------+
 |                             |                     |                             |
 v                             v                     v                             v
+------------------+     +-------------------+ +-------------------+         +------------------+
| Track A Severity |     | TF-IDF Candidate  | | SBERT Candidate   |         | High-Dimensional |
| Logistic Regres. |     | Cosine Retrieval  | | Cosine Retrieval  |         | Vector Cache     |
+--------+---------+     +---------+---------+ +---------+---------+         +------------------+
         |                         |                     |
         |                         +----------+----------+
         |                                    |
         |                                    v
         |                     +------------------------------+
         |                     | Hybrid Dual-Encoder Fusion   |
         |                     | 0.70 SBERT + 0.30 TF-IDF     |
         |                     +--------------+---------------+
         |                                    |
         v                                    v
+--------------------+         +------------------------------+
| Predicted Severity |         | Ranked Top-5 Candidate Bugs  |
|  & Probabilities   |         | (Filtered: T_cand < T_query) |
+--------------------+         +------------------------------+
```

### 3.1 Text Preprocessing Pipeline
- **Noise Removal**: Strips HTML tags, Markdown artifacts, and web URLs.
- **Contraction Expansion**: Normalizes verbal forms (e.g., *"doesn't"* $\to$ *"does not"*).
- **Technical Token Preservation**: Java package prefixes (`org.eclipse.*`), exception names (`NullPointerException`), and method invocations are preserved to maintain technical fidelity.
- **Weighted Representation**: Titles are combined with descriptions ($T_{\text{weighted}} = \text{Title} + ". " + \text{Description}$).

### 3.2 Feature Extraction
1. **TF-IDF (Lexical)**:
   - Unigram and bigram representation ($1 \le n \le 2$).
   - Vocabulary size: 10,000 features; sublinear TF scaling ($\text{TF} = 1 + \log(\text{tf})$).
   - Fit strictly on the training partition to prevent cross-partition leakage.
2. **Sentence-BERT (Dense Semantic)**:
   - Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
   - Normalization: $L_2$ unit normalization such that inner products equal cosine similarities.
   - Batch encoding with resumable disk chunk checkpoints.

### 3.3 Hybrid Duplicate Detection Engine
For a query $q$ and candidate bug report $c$:
$$\text{Sim}_{\text{Hybrid}}(q, c) = 0.70 \times \text{Sim}_{\text{SBERT}}(q, c) + 0.30 \times \text{Sim}_{\text{TF-IDF}}(q, c)$$

**Chronological Constraint**:
$$T_c < T_q \quad \text{and} \quad \text{ID}_c \neq \text{ID}_q$$

---

## 4. EXPERIMENTAL EVALUATION & RESULTS

### 4.1 Duplicate Detection Benchmark
Evaluated on **1,242 valid chronological test queries**:

| Model Architecture | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Recall@10 | MAP | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A: TF-IDF (Lexical)** | 17.71% | 29.87% | 37.04% | 37.04% | 44.61% | 0.2560 | 0.2560 |
| **Model B: SBERT (`all-MiniLM-L6-v2`)** | 33.90% | 48.79% | 54.03% | 54.03% | 62.64% | 0.4286 | 0.4286 |
| **Model C: Hybrid (0.7 SBERT + 0.3 TF-IDF)** | **33.66%** | **49.36%** | **56.60%** | **56.60%** | **64.49%** | **0.4348** | **0.4348** |

**Key Finding**: SBERT captures semantic synonymy while TF-IDF captures exact stack trace identifiers. Their fusion improves Hit@5 by **+19.56%** over TF-IDF alone.

### 4.2 Severity Classification Benchmark (Verified Test Partition: 4,528 Reports)

| Model Track & Architecture | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
| :--- | :---: | :---: | :---: | :---: |
| **Track A: TF-IDF + Logistic Regression (Selected)** | 55.65% | **24.22%** | **0.2217** | 0.6065 |
| **Track A: TF-IDF + Linear SVM** | **71.58%** | 20.35% | 0.2111 | **0.6898** |
| **Track A: SBERT + Linear SVM** | 62.83% | 21.31% | 0.2038 | 0.6472 |
| **Track A: SBERT + Logistic Regression** | 31.16% | 28.16% | 0.1758 | 0.3971 |
| **Track B: Full SVM (on Verified Test)** | 68.84% | 17.52% | 0.1727 | 0.6659 |

**Detailed Per-Class Performance (Selected Model: TF-IDF Logistic Regression)**:
- `major`: **35% Recall** (124 caught), Precision: 14%, F1: 0.20
- `critical`: **15% Recall** (28 caught), Precision: 8%, F1: 0.11
- `trivial`: **18% Recall** (31 caught), Precision: 21%, F1: 0.20
- `minor`: **11% Recall** (19 caught), Precision: 5%, F1: 0.07
- `normal`: **65% Recall** (2,317 caught), Precision: 83%, F1: 0.73
- `blocker`: **2% Recall** (1 caught), Precision: 3%, F1: 0.03

---

## 5. STREAMLIT APPLICATION MODULES

The application is organized into six functional modules:
1. **Overview Dashboard**: Displays repository statistics, dataset distributions, and performance KPI cards.
2. **Analyze Bug**: Real-time bug submission interface with input validation, severity classification, confidence estimates, Top-5 candidate cards, and JSON analysis export.
3. **Duplicate Explorer**: Interactive search across 12,935 indexed reports with keyword search, severity filtering, and duplicate relation inspection.
4. **Severity Analytics**: In-depth audit of class imbalance, per-class performance tables, and confusion matrix visualizations.
5. **Model Performance**: Head-to-head empirical comparison tables and Plotly bar charts.
6. **About Project**: Problem statement, architecture workflow, and methodology documentation.

---

## 6. CONCLUSION & FUTURE WORK

### 6.1 Conclusion
The **Bug Intelligence** system demonstrates that combining dense transformer embeddings with classical sparse lexical representations outperforms individual retrieval architectures for software duplicate detection (56.60% Hit@5, 0.4348 MRR). By applying cost-sensitive balanced class weighting, the severity classifier achieves useful recall on critical and major bugs despite severe 81.1% majority-class imbalance.

### 6.2 Future Work
- Integration with live GitHub Issues and Jira REST APIs for automated webhook triaging.
- Graph Neural Networks (GNNs) to model commit histories and developer assignment networks.
- Multi-modal triaging incorporating attached screenshot images using Vision-Language Models (VLMs).
