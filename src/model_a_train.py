from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity, paired_cosine_distances
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import MultinomialNB
from sklearn.semi_supervised import LabelPropagation
from sklearn.svm import LinearSVC

from src.model_b_train import _blend_tfidf_w2v, _w2v_cosine_to_ref
from src.preprocessing import OPTION_LABELS, PROCESSED_DIR, preprocess_all
from src.question_generation import fit_question_ranker
from src.text_utils import extract_answer_candidates, extract_candidate_phrases, split_sentences

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "model_a"


def _load_options(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split}_options.csv"
    if not path.exists():
        preprocess_all()
    return pd.read_csv(path).fillna("")


def _load_questions(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        preprocess_all()
    return pd.read_csv(path).fillna("")


def _cosine_similarity_rows(left, right):
    return np.nan_to_num(1.0 - paired_cosine_distances(left, right), nan=0.0, posinf=0.0, neginf=0.0)


def build_feature_blocks(df: pd.DataFrame, vectorizer: TfidfVectorizer, fit: bool = False):
    verification = (df["question"].astype(str) + " [OPTION] " + df["option_text"].astype(str)).astype(str)
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
            "option_len": options.map(lambda value: len(value.split())).to_numpy(), # type: ignore
        }
    )
    return hstack([x_text, dense_features.to_numpy()], format="csr")


def build_ohe_features(df: pd.DataFrame, vectorizer: CountVectorizer, fit: bool = False):
    """One-Hot Encoding (binary CountVectorizer) baseline."""
    verification = (df["question"].astype(str) + " [OPTION] " + df["option_text"].astype(str)).astype(str)
    if fit:
        return vectorizer.fit_transform(verification)
    return vectorizer.transform(verification)


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


def _similarity_scores(df: pd.DataFrame, vectorizer: TfidfVectorizer) -> np.ndarray:
    scores = np.zeros(len(df), dtype=float)
    for _, group in df.groupby("id", sort=False):
        article_x = vectorizer.transform([str(group.iloc[0]["article"])])
        question_x = vectorizer.transform([str(group.iloc[0]["question"])])
        option_x = vectorizer.transform(group["option_text"].astype(str))
        article_scores = (option_x @ article_x.T).toarray().ravel() # type: ignore
        question_scores = (option_x @ question_x.T).toarray().ravel() # type: ignore
        combined = (0.45 * article_scores) + (0.55 * question_scores)
        span = combined.max() - combined.min()
        normalized = np.full(len(group), 0.25) if span <= 1e-9 else (combined - combined.min()) / span
        scores[group.index.to_numpy()] = normalized
    return scores


def _question_text(df: pd.DataFrame) -> pd.Series:
    return (
        df["article"].astype(str)
        + " [QUESTION] "
        + df["question"].astype(str)
        + " [A] "
        + df["A"].astype(str)
        + " [B] "
        + df["B"].astype(str)
        + " [C] "
        + df["C"].astype(str)
        + " [D] "
        + df["D"].astype(str)
    )


def _evaluate_question_model(df: pd.DataFrame, predictions: np.ndarray) -> dict[str, object]:
    y_true = df["answer"].astype(str).to_numpy()
    return {
        "exact_match_answer_accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=list(OPTION_LABELS)).tolist(),
    }


_WH_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("who", re.compile(r"\bwho(?:m|se)?\b", re.IGNORECASE)),
    ("what", re.compile(r"\bwhat\b", re.IGNORECASE)),
    ("where", re.compile(r"\bwhere\b", re.IGNORECASE)),
    ("when", re.compile(r"\bwhen\b", re.IGNORECASE)),
    ("why", re.compile(r"\bwhy\b", re.IGNORECASE)),
    ("how", re.compile(r"\bhow\b", re.IGNORECASE)),
    ("which", re.compile(r"\bwhich\b", re.IGNORECASE)),
]


def wh_label(question: str) -> str:
    text = question or ""
    for label, pattern in _WH_PATTERNS:
        if pattern.search(text):
            return label
    return "other"


def _purity_score(true_labels: np.ndarray, cluster_labels: np.ndarray) -> float:
    pairs = list(zip(cluster_labels.tolist(), true_labels.tolist()))
    cluster_to_counter: dict[int, Counter] = {}
    for cluster_id, true_label in pairs:
        cluster_to_counter.setdefault(int(cluster_id), Counter())[int(true_label)] += 1
    correct = sum(c.most_common(1)[0][1] for c in cluster_to_counter.values())
    return float(correct / max(len(true_labels), 1))


def _safe_silhouette(features, labels, sample_size: int = 5000) -> float | None:
    if len(set(labels.tolist())) < 2:
        return None
    rng = np.random.default_rng(42)
    n = features.shape[0]
    if n > sample_size:
        idx = rng.choice(n, size=sample_size, replace=False)
        sub_features = features[idx]
        sub_labels = labels[idx]
    else:
        sub_features = features
        sub_labels = labels
    try:
        return float(silhouette_score(sub_features, sub_labels, metric="cosine"))
    except Exception:
        return None


def _subsample(features, labels: np.ndarray, sample_size: int, seed: int = 42):
    n = features.shape[0]
    if n <= sample_size:
        return features, labels, np.arange(n)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_size, replace=False)
    return features[idx], labels[idx], idx


def _evaluate_clustering(true_labels: np.ndarray, cluster_labels: np.ndarray, features=None) -> dict[str, object]:
    metrics: dict[str, object] = {
        "n_clusters": int(len(set(cluster_labels.tolist()))),
        "purity": _purity_score(true_labels, cluster_labels),
        "cluster_counts": pd.Series(cluster_labels).value_counts().sort_index().to_dict(),
    }
    if features is not None:
        sil = _safe_silhouette(features, cluster_labels)
        if sil is not None:
            metrics["silhouette"] = sil
    return metrics


def train(model_dir: Path = MODEL_DIR) -> dict[str, object]:
    train_df = _load_options("train")
    val_df = _load_options("validation")
    train_questions = _load_questions("train")
    val_questions = _load_questions("validation")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=30000, sublinear_tf=True, norm="l2", ngram_range=(1, 2))
    x_train = build_feature_blocks(train_df, vectorizer, fit=True)
    y_train = train_df["label"].astype(int)
    x_val = build_feature_blocks(val_df, vectorizer, fit=False)

    logistic = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    logistic.fit(x_train, y_train) # type: ignore

    svm = CalibratedClassifierCV(LinearSVC(random_state=42), cv=3)
    svm.fit(x_train, y_train) # type: ignore

    nb_text_train = vectorizer.transform(
        train_df["question"].astype(str) + " [OPTION] " + train_df["option_text"].astype(str)
    )
    nb_text_val = vectorizer.transform(
        val_df["question"].astype(str) + " [OPTION] " + val_df["option_text"].astype(str)
    )
    naive_bayes = MultinomialNB()
    naive_bayes.fit(nb_text_train, y_train)

    rf_features, rf_labels, _ = _subsample(x_train, y_train.to_numpy(), sample_size=25_000)
    random_forest = RandomForestClassifier(
        n_estimators=120,
        max_depth=24,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    )
    random_forest.fit(rf_features, rf_labels) # type: ignore

    xgb_model = None
    try:
        from xgboost import XGBClassifier  # type: ignore

        xgb_features, xgb_labels, _ = _subsample(x_train, y_train.to_numpy(), sample_size=40_000)
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=42,
        )
        xgb_model.fit(xgb_features, xgb_labels)
    except Exception:
        xgb_model = None

    ohe_vectorizer = CountVectorizer(stop_words="english", max_features=20000, binary=True, ngram_range=(1, 2))
    ohe_x_train = build_ohe_features(train_df, ohe_vectorizer, fit=True)
    ohe_x_val = build_ohe_features(val_df, ohe_vectorizer, fit=False)
    ohe_logistic = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42)
    ohe_logistic.fit(ohe_x_train, y_train)

    stack_features, stack_labels, _ = _subsample(x_train, y_train.to_numpy(), sample_size=30_000)
    stacking = StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=400, solver="liblinear", random_state=42)),
            ("svm", CalibratedClassifierCV(LinearSVC(random_state=42), cv=3)),
            ("rf", RandomForestClassifier(n_estimators=80, max_depth=20, n_jobs=-1, random_state=42)),
        ],
        final_estimator=LogisticRegression(max_iter=600, solver="lbfgs", random_state=42),
        stack_method="predict_proba",
        cv=3,
        n_jobs=-1,
    )
    stacking.fit(stack_features, stack_labels) # type: ignore

    kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
    kmeans.fit(x_train) # type: ignore

    cluster_features, cluster_labels, _ = _subsample(x_train, y_train.to_numpy(), sample_size=15_000)
    svd = TruncatedSVD(n_components=64, random_state=42)
    cluster_features_dense = svd.fit_transform(cluster_features) # type: ignore
    gmm = GaussianMixture(n_components=2, random_state=42, max_iter=120)
    gmm_assignments = gmm.fit_predict(cluster_features_dense)

    lp_features, lp_labels, _ = _subsample(x_train, y_train.to_numpy(), sample_size=4_000)
    lp_features_dense = svd.transform(lp_features) # type: ignore
    rng = np.random.default_rng(42)
    mask = rng.random(len(lp_labels)) < 0.85
    semi_labels = lp_labels.copy().astype(int)
    semi_labels[mask] = -1
    label_propagation = LabelPropagation(kernel="knn", n_neighbors=15, max_iter=400)
    label_propagation.fit(lp_features_dense, semi_labels)
    lp_predictions = label_propagation.predict(lp_features_dense)

    question_vectorizer = TfidfVectorizer(stop_words="english", max_features=50000, sublinear_tf=True, norm="l2", ngram_range=(1, 2))
    xq_train = question_vectorizer.fit_transform(_question_text(train_questions))
    xq_val = question_vectorizer.transform(_question_text(val_questions))
    yq_train = train_questions["answer"].astype(str)
    direct_logistic = LogisticRegression(max_iter=700, solver="saga", random_state=42)
    direct_logistic.fit(xq_train, yq_train)
    direct_svm = LinearSVC(random_state=42)
    direct_svm.fit(xq_train, yq_train)

    wh_vectorizer = CountVectorizer(stop_words="english", max_features=20000, ngram_range=(1, 2))
    wh_x_train = wh_vectorizer.fit_transform(train_questions["question"].astype(str))
    wh_x_val = wh_vectorizer.transform(val_questions["question"].astype(str))
    wh_y_train = train_questions["question"].astype(str).map(wh_label).to_numpy()
    wh_y_val = val_questions["question"].astype(str).map(wh_label).to_numpy()
    wh_naive_bayes = MultinomialNB()
    wh_naive_bayes.fit(wh_x_train, wh_y_train)
    wh_predictions = wh_naive_bayes.predict(wh_x_val)
    wh_labels_sorted = sorted({*wh_y_train.tolist(), *wh_y_val.tolist()}) # type: ignore

    question_ranker_vec, question_ranker_clf, question_ranker_metrics = fit_question_ranker(train_questions)

    lr_prob = logistic.predict_proba(x_val)[:, 1] # type: ignore
    svm_prob = svm.predict_proba(x_val)[:, 1] # type: ignore
    nb_prob = naive_bayes.predict_proba(nb_text_val)[:, 1]
    rf_prob = random_forest.predict_proba(x_val)[:, 1] # type: ignore
    ohe_prob = ohe_logistic.predict_proba(ohe_x_val)[:, 1]
    stack_prob = stacking.predict_proba(x_val)[:, 1] # type: ignore
    ensemble_prob = (lr_prob + svm_prob) / 2.0
    full_ensemble_prob = (lr_prob + svm_prob + nb_prob + rf_prob) / 4.0
    similarity_prob = _similarity_scores(val_df, vectorizer)
    blended_prob = (0.65 * ensemble_prob) + (0.35 * similarity_prob)

    metrics: dict[str, object] = {
        "logistic_regression": _evaluate_option_rows(val_df, lr_prob),
        "linear_svm_calibrated": _evaluate_option_rows(val_df, svm_prob),
        "naive_bayes": _evaluate_option_rows(val_df, nb_prob),
        "random_forest": _evaluate_option_rows(val_df, rf_prob),
        "ohe_logistic_regression": _evaluate_option_rows(val_df, ohe_prob),
        "soft_voting_ensemble": _evaluate_option_rows(val_df, ensemble_prob),
        "soft_voting_full_ensemble": _evaluate_option_rows(val_df, full_ensemble_prob),
        "stacking_classifier": _evaluate_option_rows(val_df, stack_prob),
        "blended_ensemble_similarity": _evaluate_option_rows(val_df, blended_prob),
        "direct_multiclass_logistic": _evaluate_question_model(val_questions, direct_logistic.predict(xq_val)),
        "direct_multiclass_svm": _evaluate_question_model(val_questions, direct_svm.predict(xq_val)),
        "wh_type_naive_bayes": {
            "accuracy": float(accuracy_score(wh_y_val, wh_predictions)),
            "macro_f1": float(f1_score(wh_y_val, wh_predictions, average="macro", zero_division=0)),
            "precision": float(precision_score(wh_y_val, wh_predictions, average="macro", zero_division=0)),
            "recall": float(recall_score(wh_y_val, wh_predictions, average="macro", zero_division=0)),
            "labels": wh_labels_sorted,
            "confusion_matrix": confusion_matrix(wh_y_val, wh_predictions, labels=wh_labels_sorted).tolist(),
        },
        "kmeans_full": {
            "n_clusters": int(kmeans.n_clusters), # type: ignore
            "cluster_counts": pd.Series(kmeans.labels_).value_counts().sort_index().to_dict(),
            "purity": _purity_score(y_train.to_numpy(), kmeans.labels_),
        },
        "kmeans_subsample_silhouette": _evaluate_clustering(
            cluster_labels, kmeans.predict(cluster_features), features=cluster_features # type: ignore
        ),
        "gaussian_mixture_subsample": _evaluate_clustering(
            cluster_labels, gmm_assignments, features=cluster_features_dense
        ),
        "question_ranker": question_ranker_metrics,
        "label_propagation_subsample": {
            "labelled_fraction": float((semi_labels != -1).mean()),
            "accuracy_on_full_subsample": float(accuracy_score(lp_labels, lp_predictions)),
            "macro_f1_on_full_subsample": float(f1_score(lp_labels, lp_predictions, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(lp_labels, lp_predictions).tolist(),
        },
        "option_labels": list(OPTION_LABELS),
    }
    if xgb_model is not None:
        xgb_prob = xgb_model.predict_proba(x_val)[:, 1]
        metrics["xgboost"] = _evaluate_option_rows(val_df, xgb_prob)

    metrics["model_comparison"] = _build_model_comparison_table(metrics)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    joblib.dump(logistic, model_dir / "logistic_regression.joblib")
    joblib.dump(svm, model_dir / "linear_svm_calibrated.joblib")
    joblib.dump(naive_bayes, model_dir / "naive_bayes.joblib")
    joblib.dump(random_forest, model_dir / "random_forest.joblib")
    joblib.dump(ohe_vectorizer, model_dir / "count_vectorizer.joblib")
    joblib.dump(ohe_logistic, model_dir / "ohe_logistic_regression.joblib")
    joblib.dump(stacking, model_dir / "stacking_classifier.joblib")
    joblib.dump(kmeans, model_dir / "kmeans.joblib")
    joblib.dump(svd, model_dir / "cluster_svd.joblib")
    joblib.dump(gmm, model_dir / "gmm.joblib")
    joblib.dump(label_propagation, model_dir / "label_propagation.joblib")
    joblib.dump(question_vectorizer, model_dir / "question_tfidf_vectorizer.joblib")
    joblib.dump(direct_logistic, model_dir / "direct_multiclass_logistic.joblib")
    joblib.dump(direct_svm, model_dir / "direct_multiclass_svm.joblib")
    joblib.dump(wh_vectorizer, model_dir / "wh_count_vectorizer.joblib")
    joblib.dump(wh_naive_bayes, model_dir / "wh_naive_bayes.joblib")
    joblib.dump(question_ranker_vec, model_dir / "question_ranker_vectorizer.joblib")
    joblib.dump(question_ranker_clf, model_dir / "question_ranker.joblib")
    if xgb_model is not None:
        joblib.dump(xgb_model, model_dir / "xgboost.joblib")

    with (model_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    with (model_dir / "ensemble_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "members": ["logistic_regression", "linear_svm_calibrated", "naive_bayes", "random_forest"],
                "soft_voting": "mean_probability",
                "stacking_meta_classifier": "logistic_regression",
                "stacking_subsample": 30_000,
            },
            handle,
            indent=2,
        )
    return metrics


def _build_model_comparison_table(metrics: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    candidates = [
        ("Logistic Regression (TF-IDF)", "logistic_regression", "supervised"),
        ("Linear SVM Calibrated (TF-IDF)", "linear_svm_calibrated", "supervised"),
        ("Multinomial Naive Bayes (TF-IDF)", "naive_bayes", "supervised"),
        ("Random Forest (subsample)", "random_forest", "supervised"),
        ("Logistic Regression (One-Hot)", "ohe_logistic_regression", "supervised"),
        ("XGBoost (subsample)", "xgboost", "supervised"),
        ("Soft Voting (LR+SVM)", "soft_voting_ensemble", "ensemble"),
        ("Soft Voting (LR+SVM+NB+RF)", "soft_voting_full_ensemble", "ensemble"),
        ("Stacking Classifier", "stacking_classifier", "ensemble"),
        ("Blended Ensemble + Similarity", "blended_ensemble_similarity", "ensemble"),
    ]
    for name, key, family in candidates:
        if key not in metrics or not isinstance(metrics[key], dict):
            continue
        entry = metrics[key]
        rows.append(
            {
                "model": name,
                "family": family,
                "accuracy": entry.get("accuracy"), # type: ignore
                "macro_f1": entry.get("macro_f1"), # type: ignore
                "precision": entry.get("precision"), # type: ignore
                "recall": entry.get("recall"), # type: ignore
                "exact_match_answer_accuracy": entry.get("exact_match_answer_accuracy"), # type: ignore
            }
        )
    return rows

def _leaves_enough_remainder(answer: str, sentences: list[str], min_words: int = 4) -> bool:
    answer_lower = answer.lower()
    pattern = re.compile(re.escape(answer), flags=re.IGNORECASE)
    for sentence in sentences:
        if answer_lower not in sentence.lower():
            continue
        scrubbed = pattern.sub(" ", sentence, count=1)
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", scrubbed) if len(w) >= 3]
        if len(words) >= min_words:
            return True
    return False


def _generation_answer_candidates(article: str, sentences: list[str], max_phrases: int = 100) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for phrase in extract_candidate_phrases(article, max_candidates=max_phrases):
        key = phrase.strip().lower()
        if len(key) < 4 or key in seen:
            continue
        if len(phrase.split()) > 8:
            continue
        if not any(key in s.lower() for s in sentences):
            continue
        if not _leaves_enough_remainder(phrase, sentences):
            continue
        seen.add(key)
        out.append(phrase.strip())
    for sentence in sentences:
        for c in extract_answer_candidates(sentence, max_candidates=8):
            key = c.strip().lower()
            if len(key) < 4 or key in seen:
                continue
            if len(c.split()) > 8:
                continue
            if not any(key in s.lower() for s in sentences):
                continue
            if not _leaves_enough_remainder(c, sentences):
                continue
            seen.add(key)
            out.append(c.strip())
    return out


def rank_generation_sentence_answer_pairs(
    article: str,
    vectorizer: TfidfVectorizer,
    w2v: "KeyedVectors | None" = None,
    generation_w2v_blend: float = 0.38,
) -> list[tuple[float, str, str]]:
    sentences = [s for s in split_sentences(article) if 25 <= len(s.strip()) <= 280]
    if not sentences and article.strip():
        sentences = [article.strip()[:400]]
    if not sentences:
        return []
    pool = _generation_answer_candidates(article, sentences, max_phrases=100)
    if not pool:
        return []
    triples: list[tuple[float, str, str]] = []
    sent_x = vectorizer.transform(sentences)
    for cand in pool:
        a_x = vectorizer.transform([cand])
        tfidf_sims = cosine_similarity(sent_x, a_x).ravel()
        if w2v is not None and generation_w2v_blend > 0:
            w2v_sims = _w2v_cosine_to_ref(sentences, cand, w2v)
            combined = _blend_tfidf_w2v(tfidf_sims, w2v_sims, generation_w2v_blend)
        else:
            combined = tfidf_sims.astype(np.float64)
        for i, sent in enumerate(sentences):
            if cand.lower() not in sent.lower():
                continue
            triples.append((float(combined[i]), sent, cand))
    if not triples:
        return []
    triples.sort(key=lambda x: -x[0])
    best_per_sentence: dict[str, tuple[float, str, str]] = {}
    for sc, s, a in triples:
        key = s.strip()
        if key not in best_per_sentence or sc > best_per_sentence[key][0]:
            best_per_sentence[key] = (sc, s, a)
    return sorted(best_per_sentence.values(), key=lambda x: -x[0])

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model A traditional ML, ensembles, and unsupervised baselines.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    print(json.dumps(train(args.model_dir), indent=2))


if __name__ == "__main__":
    main()
