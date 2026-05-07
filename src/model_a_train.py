from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.svm import LinearSVC

from src.preprocessing import OPTION_LABELS, PROCESSED_DIR, preprocess_all

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "model_a"


def _load_options(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split}_options.csv"
    if not path.exists():
        preprocess_all()
    return pd.read_csv(path).fillna("")


def _cosine_similarity_rows(left, right):
    return np.nan_to_num(1.0 - paired_cosine_distances(left, right), nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_blocks(df: pd.DataFrame, vectorizer: TfidfVectorizer, fit: bool = False):
    verification = df["verification_text"].astype(str)
    articles = df["article"].astype(str)
    questions = df["question"].astype(str)
    options = df["option_text"].astype(str)
    x_text = vectorizer.fit_transform(verification) if fit else vectorizer.transform(verification)
    article_x = vectorizer.transform(articles)
    question_x = vectorizer.transform(questions)
    option_x = vectorizer.transform(options)
    dense_features = pd.DataFrame(
        {
            "article_question_sim": _cosine_similarity_rows(article_x, question_x),
            "article_option_sim": _cosine_similarity_rows(article_x, option_x),
            "question_option_sim": _cosine_similarity_rows(question_x, option_x),
            "option_len": options.map(lambda value: len(value.split())).to_numpy(),
        }
    )
    return hstack([x_text, dense_features.to_numpy()], format="csr")


def _evaluate_option_rows(df: pd.DataFrame, probabilities) -> dict[str, object]:
    pred_rows = (probabilities >= 0.5).astype(int)
    y_true = df["label"].astype(int).to_numpy()
    grouped = df[["id", "option_label", "answer"]].copy()
    grouped["probability"] = probabilities
    chosen = grouped.loc[grouped.groupby("id")["probability"].idxmax()]
    exact_match = float((chosen["option_label"] == chosen["answer"]).mean())
    return {
        "accuracy": float(accuracy_score(y_true, pred_rows)),
        "macro_f1": float(f1_score(y_true, pred_rows, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, pred_rows, zero_division=0)),
        "recall": float(recall_score(y_true, pred_rows, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred_rows).tolist(),
        "exact_match_answer_accuracy": exact_match,
    }


def train(model_dir: Path = MODEL_DIR) -> dict[str, object]:
    train_df = _load_options("train")
    val_df = _load_options("validation")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=30000, sublinear_tf=True, norm="l2", ngram_range=(1, 2))
    x_train = build_feature_blocks(train_df, vectorizer, fit=True)
    y_train = train_df["label"].astype(int)
    x_val = build_feature_blocks(val_df, vectorizer, fit=False)

    logistic = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
    logistic.fit(x_train, y_train)

    svm = CalibratedClassifierCV(LinearSVC(class_weight="balanced", random_state=42), cv=3)
    svm.fit(x_train, y_train)

    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    kmeans.fit(x_train)

    lr_prob = logistic.predict_proba(x_val)[:, 1]
    svm_prob = svm.predict_proba(x_val)[:, 1]
    ensemble_prob = (lr_prob + svm_prob) / 2.0
    metrics = {
        "logistic_regression": _evaluate_option_rows(val_df, lr_prob),
        "linear_svm_calibrated": _evaluate_option_rows(val_df, svm_prob),
        "soft_voting_ensemble": _evaluate_option_rows(val_df, ensemble_prob),
        "kmeans_cluster_counts": pd.Series(kmeans.labels_).value_counts().sort_index().to_dict(),
        "option_labels": list(OPTION_LABELS),
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    joblib.dump(logistic, model_dir / "logistic_regression.joblib")
    joblib.dump(svm, model_dir / "linear_svm_calibrated.joblib")
    joblib.dump(kmeans, model_dir / "kmeans.joblib")
    with (model_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (model_dir / "ensemble_config.json").open("w", encoding="utf-8") as handle:
        json.dump({"members": ["logistic_regression", "linear_svm_calibrated"], "strategy": "mean_probability"}, handle, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model A TF-IDF answer verifier.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    print(json.dumps(train(args.model_dir), indent=2))


if __name__ == "__main__":
    main()
