# RACE Reading Comprehension Quiz Generator

TF-IDF based reading comprehension and quiz generation system with classical ML models, a FastAPI backend, and a Vite React + Tailwind frontend.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If this machine uses `pyenv-win`, set a Python version first, for example:

```powershell
pyenv install 3.11.9
pyenv local 3.11.9
```

## Data

Place RACE files under `data/raw/` as either Parquet or CSV:

- `train.parquet` or `train.csv`
- `validation.parquet`, `validation.csv`, or `val.csv`
- `test.parquet` or `test.csv`

The loader normalizes common RACE schemas into:

`id`, `article`, `question`, `A`, `B`, `C`, `D`, `answer`

## Pipeline

```powershell
python -m src.preprocessing
python -m src.model_a_train
python -m src.model_b_train
python -m src.evaluate
```

Processed datasets are written to `data/processed/`. Model artifacts are written to `models/model_a/` and `models/model_b/`. The loader supports Parquet `options` values stored as array-like objects and expands them into populated `A/B/C/D` fields.

### Evaluation Metrics

The evaluation pipeline (`python -m src.evaluate`) computes text generation quality metrics:

- **ROUGE**: Measures content overlap (ROUGE-1, ROUGE-2, ROUGE-L) between generated and reference text
- **BLEU**: Evaluates n-gram precision (BLEU-1 through BLEU-4) with brevity penalty
- **METEOR**: Provides semantic-aware scoring with stemming, synonym matching, and word order penalties

These metrics are computed against the test dataset and saved to `models/evaluation_metrics.json` alongside Model A (verifier) and Model B (distractor/hint) metrics.

Model B fits a TF-IDF vectorizer plus a **Word2Vec** model (skip-gram) on passage sentences, questions, and options. Distractor ranking and hint sentence selection blend TF-IDF cosine similarity with mean-pooled Word2Vec cosine similarity (weights in `models/model_b/config.json` under `word2vec`). After retraining, `word2vec.kv` is written next to the TF-IDF joblib; if that file is missing, inference falls back to TF-IDF only.

Cloze **question stems** are chosen by ranking passage sentences against candidate answer phrases: for each pair where the answer text appears in the sentence, the score is cosine similarity between TF-IDF vectors of the sentence and of the answer (Model A’s vectorizer when Model A is loaded, otherwise Model B’s), optionally blended with Word2Vec when using Model B’s vectorizer (`generation_blend_weight` in config). Top-scoring unique (sentence, answer) pairs are emitted first; any remaining slots use the previous heuristic sentence ranker.

### Model A (verifier + question generator)

Model A trains a stack of classical ML models on the option-level binary task plus auxiliary direct multiclass baselines:

- **Logistic Regression** and **Linear SVM (calibrated)** on TF-IDF + cosine-similarity features.
- **Multinomial Naive Bayes** verifier and a separate **Wh-type Naive Bayes** question classifier.
- **Random Forest** (subsample) and optional **XGBoost** verifier.
- **One-Hot Encoding** baseline (binary `CountVectorizer`) wired to its own Logistic Regression for an OHE-vs-TF-IDF comparison.
- **Soft voting** (LR + SVM, and LR+SVM+NB+RF) and a **stacking classifier** (LR meta-learner over LR/SVM/NB/RF).
- **Unsupervised / semi-supervised**: KMeans purity + cluster counts, KMeans silhouette on a SVD-reduced subsample, **Gaussian Mixture** clustering with purity/silhouette, **Label Propagation** semi-supervised baseline (kNN affinity, 15% labels).
- A **question ranker** (Logistic Regression over CountVectorizer features + lexical signals) trained on real RACE questions vs templated synthetics; this scores Wh-template candidates at inference time.

Metrics, a unified `model_comparison` table, and clustering diagnostics are written to `models/model_a/metrics.json`.

### Model B (distractors + hints)

Model B writes TF-IDF + Word2Vec artifacts and two trained models:

- **Distractor ranker** — Logistic Regression on per-candidate features (TF-IDF answer/question cosine, Word2Vec answer/question cosine, passage frequency, character-level overlap, candidate length, token-overlap-with-answer). It blends with the heuristic weighted score at inference (`distractor_ranker_weight` in `config.json`).
- **Hint scorer** — a Ridge regression over (keyword overlap, position, length, first-token match, contains-answer-token). At inference its predictions blend with the TF-IDF/W2V cosine ranking; `R²` is reported on validation.
- **Frequency-substitution distractors** — alternative pipeline that picks article phrases with frequency closest to the gold answer.

`models/model_b/config.json` includes the distractor / hint blend weights, the trained-ranker validation metrics, and a `evaluation` block with **distractor Precision/Recall/F1**, **ranker top-1-not-answer accuracy**, and **hint Precision@K / contains-answer rate**.

### Evaluation

`python -m src.evaluate` writes `models/evaluation_metrics.json` with:
- **Model A**: Exact-match accuracy, macro F1, precision, recall, and confusion matrix on the test split
- **Model B Distractors**: Precision, Recall, F1, and ranker top-1-not-answer accuracy
- **Model B Hints**: Precision@3 and top-3 contains-answer rate
- **Text Generation**: ROUGE (1/2/L), BLEU (1-4), and METEOR scores computed on the test dataset

Customize evaluation with command-line options:
```powershell
python -m src.evaluate --split test --limit 100  # Limit to 100 samples
python -m src.evaluate --split validation         # Evaluate on validation set
python -m src.evaluate --split train              # Evaluate on training set
```

## Tests & EDA

- `notebooks/EDA.ipynb` — passage length distributions, answer label balance, Wh-type counts, option length statistics, and a heuristic answer-type taxonomy.
- `python -m unittest discover tests` — fast smoke tests for `QuizEngine`, Wh-template generation, character-level matching, distractor features, and frequency substitution.

## Backend

```powershell
uvicorn backend.main:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `GET /sample`
- `POST /generate`
- `POST /verify`
- `GET /metrics`
- `GET /logs/export`

`POST /generate` returns at least five AI-generated cloze-style quiz questions for each submitted article. Each generated item is grounded in a source sentence and includes its own A/B/C/D options, predicted answer, distractors, graduated hints, confidence, and latency fields.

## Frontend

```powershell
cd ui
npm install
npm run dev
```

The UI uses Tailwind utility classes, supports pasted passages and `.txt` article uploads, and expects the backend at `http://localhost:8000` by default. Set `VITE_API_BASE_URL` to override it. The frontend build is pinned to stable Vite 5 packages.

## Text Metrics Module

The `src/text_metrics.py` module provides functions to compute text generation quality metrics:

- `compute_rouge(references, hypotheses)` — ROUGE-1, ROUGE-2, ROUGE-L F-scores
- `compute_bleu(references, hypotheses)` — BLEU-1 through BLEU-4 scores
- `compute_meteor(references, hypotheses)` — METEOR score with stemming and synonym matching
- `compute_all_text_metrics(references, hypotheses)` — Combined computation of all metrics

```python
from src.text_metrics import compute_all_text_metrics

references = ["What is the capital of France?", "How does photosynthesis work?"]
hypotheses = ["What is Paris?", "How does plants make food?"]
metrics = compute_all_text_metrics(references, hypotheses)
print(metrics)  # {'rouge1_f': ..., 'bleu1': ..., 'meteor': ...}
```
