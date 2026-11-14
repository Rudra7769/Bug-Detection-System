# Dataset & Evaluation Audit Report: Bug-Detection-System

**Generated**: Comprehensive Pre-Training Audit  
**Dataset**: Eclipse Bugzilla Open-Source Repository & Curated Bugzilla Severity Benchmark  
**Total Cleaned Reports**: 84,653  

---

## Executive Summary & Readiness Assessment

| Audit Dimension | Status | Key Finding |
| :--- | :---: | :--- |
| **Severity Source Integrity** | ⚠️ BIFURCATED | 14,924 verified vs 69,729 priority-derived labels. Two distinct model tracks required. |
| **Priority-to-Severity Mapping** | ⚠️ EMPIRICAL MISMATCH | In Bugzilla, P1-P5 reflects triage urgency, not crash impact. Mapping introduces severe noise. |
| **Duplicate Master IDs** | ⚠️ 91.2% FOUND | 12,880 of 14,117 master bug IDs exist in the dataset; 1,237 reference external components. |
| **Cycle & Self-Match Integrity** | ✅ PASS | 0 cyclic reference loops, 0 self-referencing duplicates detected. |
| **Chronological Duplicate Queries**| ✅ IDENTIFIED | **10,035 valid queries** globally; **1,242 valid queries** in the Test Partition. |
| **Class Imbalance** | ⚠️ SEVERE | 'Normal' severity accounts for 81.1% of verified labels and 85.3% of full data. |
| **Readiness for Training** | ✅ READY | **Conditional on two-track training, class weighting, and filtered temporal retrieval.** |

---

## 1. Severity Source Analysis & Exact Mapping Investigation

### 1.1 Dataset Composition by Source

The dataset contains two distinct categories of severity labels:
- **Directly Verified Labels (`bugzilla_verified`)**: **14,924 records (17.6%)** sourced from the curated Bugzilla benchmark.
- **Priority-Derived Labels (`bugzilla_priority_derived`)**: **69,729 records (82.4%)** derived from Eclipse's `Priority` field.

```
Total Cleaned Reports: 84,653
├── Directly Verified:  14,924 (17.63%)
└── Priority-Derived:   69,729 (82.37%)
```

### 1.2 The Priority-to-Severity Mapping & Logical Defensibility

In software engineering research, naive pipelines sometimes map:
- `P1 -> blocker`
- `P2 -> critical`
- `P3 -> normal`
- `P4 -> minor`
- `P5 -> trivial`

#### Empirical Verification: Crosstab on Verified Subset (14,924 Bugs)

| Priority | blocker | critical | major | minor | normal | trivial | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **P1** | 5 (3.4%) | 13 (8.7%) | 28 (18.8%) | 1 (0.7%) | **102 (68.5%)** | 0 (0.0%) | 149 |
| **P2** | 3 (0.6%) | 16 (3.3%) | 72 (14.7%) | 12 (2.4%) | **381 (77.8%)** | 6 (1.2%) | 490 |
| **P3** | 136 (1.0%) | 424 (3.0%) | 1,062 (7.5%) | 552 (3.9%) | **11,574 (81.4%)** | 478 (3.4%) | 14,226 |
| **P4** | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 1 (4.2%) | **23 (95.8%)** | 0 (0.0%) | 24 |
| **P5** | 0 (0.0%) | 0 (0.0%) | 2 (5.7%) | 6 (17.1%) | **24 (68.6%)** | 3 (8.6%) | 35 |
| **All** | **144** | **453** | **1,164** | **572** | **12,104** | **487** | **14,924** |

#### Crucial Finding:
1. **Priority != Severity**: In Bugzilla workflow guidelines, *Priority* represents scheduling urgency assigned by project managers, whereas *Severity* represents the technical impact of the defect (data loss, crash, or cosmetic).
2. **68.5% of verified P1 bugs and 77.8% of verified P2 bugs are actually labeled `normal` severity!**
3. Furthermore, Bugzilla defaults new tickets to `P3`, meaning 81.4% of P3 tickets are `normal`, but 1,622 severe bugs (blocker/critical/major) were also left at default `P3`.
4. **Conclusion**: Deriving severity from priority introduces high label noise. Consequently, **the Verified-Only Model (Track A) must serve as the primary scientific benchmark**, while the Full-Data Model (Track B) must be explicitly presented as a high-coverage, noise-tolerant baseline.

---

## 2. Duplicate Relationship Graph Integrity Audit

Duplicate relationships in Bugzilla represent links from a newly reported duplicate bug to its existing primary master ticket.

### 2.1 Master ID Existence & Missingness

- **Total Reports with Duplicate Link**: **14,117**
- **Master IDs Present in Dataset**: **12,880 (91.24%)**
- **Master IDs Missing from Dataset**: **1,237 (8.76%)**
- **Reason for Missing Master IDs**: In the Eclipse ecosystem, bugs frequently cross project boundaries (e.g., a bug in `Platform` marked as duplicate of a ticket in `JDT` or `Equinox`). Since our raw platform dump covers `Eclipse Platform`, references to other project components are not within this table.
- **Audit Action**: Reports whose master ID is missing **must be excluded from IR retrieval evaluation** because the target report is physically absent from the candidate index.

### 2.2 Graph Topology: Cycles, Self-Matches, and Transitivity

- **Self-Matches (`bug_id == duplicate_id`)**: **0 (0.0%)** — Integrity Verified.
- **Cycles (`A -> B -> A` or cyclic loops)**: **0 (0.0%)** — Graph is a strict Directed Acyclic Graph (DAG).
- **Transitive Duplicate Chains (`A -> B -> C`)**: **916 reports (6.5%)** point to a ticket that was itself resolved as a duplicate of another master.
- **Unique Canonical Masters**: Through recursive graph path compression, the 14,117 duplicates resolve into **8,125 unique canonical master clusters**.

### 2.3 Chronological Validity & Temporal Constraints

In a real bug triaging workflow, duplicate detection operates forward in time: when an engineer triages a newly submitted bug at time $T$, the system can **only** search historical bugs submitted strictly before $T$.

Across the 12,880 duplicates with present master targets:
- **Query created strictly AFTER master (`created_dt > master_created_dt`)**: **10,035 (77.91%)** $ightarrow$ **Valid evaluation queries**.
- **Query created at SAME time as master**: **66 (0.5%)**
- **Query created BEFORE master (`created_dt < master_created_dt`)**: **2,779 (21.6%)** $ightarrow$ Occurs in Bugzilla when a later, better-documented report is designated as the canonical master and the earlier bug is retroactively resolved as a duplicate.

### 2.4 Verified Evaluation Query Counts

| Partition | Total Duplicate Reports | Master Present | Strictly Historical | Valid Evaluation Queries |
| :--- | :---: | :---: | :---: | :---: |
| **Full Dataset** | 14,117 | 12,880 | 10,035 | **10,035** |
| **Train Partition (70%)** | 10,478 | 9,562 | 7,422 | **7,422** |
| **Validation Partition (15%)** | 1,928 | 1,783 | 1,371 | **1,371** |
| **Test Partition (15%)** | 1,711 | 1,535 | 1,242 | **1,242** |

---

## 3. Class Imbalance Analysis

### 3.1 Severity Class Imbalance Breakdown

| Class | Verified Count | Verified Share | Full Dataset Count | Full Dataset Share | Imbalance Ratio (vs Blocker) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **normal** | 12,104 | 81.10% | 72,218 | 85.31% | **84.1 : 1** |
| **major** | 1,164 | 7.80% | 1,164 | 1.38% | 8.1 : 1 |
| **minor** | 572 | 3.83% | 2,489 | 2.94% | 4.0 : 1 |
| **trivial** | 487 | 3.26% | 1,651 | 1.95% | 3.4 : 1 |
| **critical** | 453 | 3.04% | 5,064 | 5.98% | 3.1 : 1 |
| **blocker** | 144 | 0.96% | 2,067 | 2.44% | **1.0 : 1** |
| **Total** | **14,924** | **100.0%** | **84,653** | **100.0%** | — |

### 3.2 Imbalance Mitigation Strategy

1. **Class-Weighted Linear SVM & Logistic Regression**: Apply inverse-frequency class weights (`class_weight='balanced'`) during optimization so minority classes (`blocker`, `critical`) carry proportionate penalty.
2. **Primary Metric**: Rely on **Macro F1** and **Balanced Accuracy** rather than Standard Accuracy (which achieves 81% simply by predicting `normal` every time).
3. **No Cross-Boundary Synthetic Resampling**: Do not apply SMOTE or oversampling across test boundaries to avoid synthetic data leakage.

---

## 4. Evaluation Metrics Definition & Leakage Prevention

### 4.1 Duplicate Detection Metrics
- **Hit@1, Hit@3, Hit@5**: Percentage of queries where at least one ground-truth duplicate is ranked in the top-K.
- **Recall@5, Recall@10**: Proportion of all known true duplicates retrieved in top-K:
  Recall@K = |Relevant in Top-K| / |Total Relevant|
- **Mean Average Precision (MAP)**: Measures ranking quality across multiple relevant duplicates in a cluster:
  AP = sum(Precision@k * rel(k)) / |Total Relevant|
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the first relevant retrieved duplicate:
  MRR = (1 / |Q|) * sum(1 / rank_i)

### 4.2 Data Leakage Safeguards
- **TF-IDF Vocabulary**: Fitted strictly on training text documents; unseen tokens in validation/test are mapped to zero.
- **Candidate Pool Constraint**: Test queries are evaluated only against historical reports ($T_{candidate} < T_{query}$).
- **Self-Match Filter**: Candidate matching explicitly eliminates the query ID itself.
- **Threshold Freezing**: Similarity parameters are tuned on Validation only; Test data is touched once for final reporting.

---

## 5. Recommended Training Configurations

### Track A: Verified-Only Severity Model
- **Dataset**: `severity_source == 'bugzilla_verified'` (14,924 records).
- **Split**: Chronological (70% Train: 10,446 | 15% Val: 2,239 | 15% Test: 2,239).
- **Classifier**: Linear SVM with probability calibration (`CalibratedClassifierCV`) + Class Weight Balancing.
- **Evaluation Target**: Direct comparison against verified ground-truth labels.

### Track B: Full-Data Severity Model
- **Dataset**: All 84,653 cleaned records with `severity_source` retained.
- **Split**: Chronological (59,257 Train | 12,698 Val | 12,698 Test).
- **Evaluation**: Evaluated both on all test records and on the verified test sub-segment to quantify priority label distortion.

### Track C: Duplicate Retrieval Engine
- **Dense Model**: Sentence Transformers `all-MiniLM-L6-v2` (384-dim normalized embeddings).
- **Lexical Model**: Sublinear TF-IDF (1-2 ngrams, 5,000 features).
- **Evaluation**: 1,242 valid test queries against the pre-query historical corpus.
