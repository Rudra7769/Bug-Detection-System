# Bug Report Duplicate Detection & Severity Prediction System

An end-to-end intelligent bug triaging platform that leverages Natural Language Processing, Sentence Transformers semantic embeddings (`all-MiniLM-L6-v2`), classical machine learning classifiers, and a professional Streamlit web frontend.

---

## 📌 Project Architecture

```
Bug-Detection-System/
├── data/
│   ├── raw/                 # Ingested Eclipse & Bugzilla defect reports
│   │   ├── eclipse_platform.csv
│   │   └── eclipse_severity.csv
│   ├── processed/           # Cleaned and split datasets (no leakage)
│   │   ├── cleaned_bug_reports.csv
│   │   ├── train_bug_reports.csv
│   │   ├── val_bug_reports.csv
│   │   └── test_bug_reports.csv
│   └── sample/              # Curated 1,000-sample dataset for fast testing
│       └── sample_test_bugs.csv
├── notebooks/               # Research & EDA experiments
├── src/                     # Core system modules
│   ├── __init__.py
│   ├── config.py            # Centralized settings and dynamic path resolution
│   ├── data_loader.py       # Ingestion & consolidation of Bugzilla datasets
│   ├── preprocessing.py     # Clean text, stack trace preservation & normalization
│   ├── feature_extraction.py# TF-IDF & Sentence Transformer semantic vectorizer
│   ├── duplicate_detection.py # Top-K duplicate retrieval engine
│   ├── severity_prediction.py # Multi-class severity classification
│   └── evaluation.py        # MRR, Hit@K, Confusion Matrix, and F1 reports
├── models/                  # Serialized ML vectorizers and embedding indexes
├── app/                     # Frontend Streamlit application
│   ├── main.py              # Application layout & interaction logic
│   ├── components.py        # Cards, Plotly charts, and KPI widgets
│   └── styles.py            # Custom GitHub/Linear-inspired styling
├── reports/                 # Evaluation figures and performance reports
├── tests/                   # Automated unit & integration test suite
│   └── test_pipeline.py
├── run_pipeline.py          # Data ingestion, preprocessing, and splitting runner
├── streamlit_app.py         # Root entrypoint for Streamlit UI
├── requirements.txt         # Project dependencies
├── README.md                # System documentation
└── .gitignore               # Ignored cache, models, and large data files
```

---

## 🚀 Key Capabilities

1. **Duplicate Bug Report Detection**:
   - Indexes existing bug reports in a vector space.
   - Computes dense semantic embeddings using `all-MiniLM-L6-v2` (Sentence Transformers).
   - Retrieves and ranks the Top-5 most similar existing bug reports with cosine similarity match percentages.
   - Flags known ground-truth duplicate relationships.

2. **Bug Severity Prediction**:
   - Multi-class classification: `blocker`, `critical`, `major`, `normal`, `minor`, `trivial`.
   - Balanced class-weight optimization for handling software engineering class imbalance.
   - Interactive probability breakdown rendered with Plotly.

3. **Production Data Quality**:
   - **No fabricated data**: Ingests 85,156 genuine Bugzilla reports from Eclipse Platform.
   - Merges 14,924 verified Bugzilla severity labels with full platform issue texts.
   - Preserves 14,117 real-world duplicate relation links.
   - Preserves Java stack traces, classpaths, and error codes during NLP tokenization.

---

## 🛠️ Installation & Setup

1. **Clone or Navigate to the Directory**:
   ```bash
   cd Bug-Detection-System
   ```

2. **Install Required Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Automated Data Preprocessing & Splitting Pipeline**:
   ```bash
   python run_pipeline.py
   ```

4. **Execute the Test Suite**:
   ```bash
   python -m unittest discover -s tests
   ```

5. **Launch the Streamlit Frontend**:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## 📊 Dataset & Model Benchmarks

### 1. Dataset Breakdown
- **Source**: Eclipse Bugzilla Defect Archive (LogPAI BugHub) + Bugzilla Severity Benchmark.
- **Cleaned Records**: 84,653 bug reports.
- **Severity Sources**:
  - **Directly Verified**: 14,924 reports (genuine triage labels).
  - **Priority-Derived**: 69,729 reports (heuristic mapping).
- **Class Imbalance**:
  - **Verified Dataset (14,924)**: `normal` = 81.10% (12,104), `major` = 7.80% (1,164), `minor` = 3.83% (572), `trivial` = 3.26% (487), `critical` = 3.04% (453), `blocker` = 0.96% (144).
  - **Full Dataset (84,653)**: `normal` = 85.31% (72,218), `critical` = 5.98% (5,064), `minor` = 2.94% (2,489), `blocker` = 2.44% (2,067), `trivial` = 1.95% (1,651), `major` = 1.38% (1,164).

### 2. Duplicate Detection Benchmark (1,242 Chronological Test Queries)
Evaluated strictly under $T_{candidate} < T_{query}$ with self-matches prohibited:
| Retrieval Model | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Recall@10 | MAP | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF (Lexical)** | 17.71% | 29.87% | 37.04% | 37.04% | 44.61% | 0.2560 | 0.2560 |
| **Sentence-BERT (`all-MiniLM-L6-v2`)** | **33.90%** | 48.79% | 54.03% | 54.03% | 62.64% | 0.4286 | 0.4286 |
| **Hybrid (0.70 SBERT + 0.30 TF-IDF)** | 33.66% | **49.36%** | **56.60%** | **56.60%** | **64.49%** | **0.4348** | **0.4348** |

### 3. Severity Prediction Benchmark (4,528 Verified Test Reports)
Selected Model: **Track A Verified TF-IDF Logistic Regression** (Cost-sensitive `class_weight='balanced'`):
- **Accuracy**: 55.65%
- **Balanced Accuracy**: 24.22% (Random Baseline = 16.67%)
- **Macro F1-Score**: 0.2217
- **Weighted F1-Score**: 0.6065
- **Minority Recall**: 35.0% on `major` bugs, 15.0% on `critical` bugs.

*Notice*: Minority class prediction remains constrained by severe open-source class imbalance. Model confidence estimates are designed to assist human triagers, not replace developer judgment.
