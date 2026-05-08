from __future__ import annotations

import argparse
import json
import random
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import OPTION_LABELS, PROCESSED_DIR, preprocess_all
from src.text_utils import extract_answer_candidates, extract_candidate_phrases, split_sentences

if TYPE_CHECKING:
    from gensim.models import KeyedVectors

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "model_b"
W2V_KV_NAME = "word2vec.kv"
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]{1,}")


# ---------------------------------------------------------------------------
# Word2Vec helpers
# ---------------------------------------------------------------------------


def _w2v_tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def _text_to_unit_vector(text: str, wv: KeyedVectors) -> np.ndarray | None:
    vecs = [wv[t] for t in _w2v_tokens(text) if t in wv]
    if not vecs:
        return None
    v = np.mean(np.stack(vecs, axis=0), axis=0, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return None
    return (v / n).astype(np.float32)


def _batch_unit_vectors(texts: list[str], wv: KeyedVectors) -> tuple[np.ndarray, np.ndarray]:
    dim = int(wv.vector_size)
    mat = np.zeros((len(texts), dim), dtype=np.float32)
    mask = np.zeros(len(texts), dtype=np.float32)
    for i, text in enumerate(texts):
        u = _text_to_unit_vector(text, wv)
        if u is not None:
            mat[i] = u
            mask[i] = 1.0
    return mat, mask


def _w2v_cosine_to_ref(texts: list[str], ref_text: str, wv: KeyedVectors) -> np.ndarray:
    ref = _text_to_unit_vector(ref_text, wv)
    if ref is None:
        return np.full(len(texts), 0.25, dtype=np.float64)
    mat, mask = _batch_unit_vectors(texts, wv)
    sims = (mat @ ref.astype(np.float32)).astype(np.float64)
    sims = np.where(mask > 0, sims, 0.25)
    return sims


def _normalize01(scores: np.ndarray) -> np.ndarray:
    lo, hi = float(scores.min()), float(scores.max())
    span = hi - lo
    if span <= 1e-9:
        return np.full_like(scores, 0.25, dtype=np.float64)
    return (scores - lo) / span


def _blend_tfidf_w2v(tfidf: np.ndarray, w2v: np.ndarray, w2v_weight: float) -> np.ndarray:
    w = float(np.clip(w2v_weight, 0.0, 1.0))
    t = _normalize01(tfidf.astype(np.float64))
    v = _normalize01(w2v.astype(np.float64))
    return (1.0 - w) * t + w * v


def _w2v_training_sentences(train_df: pd.DataFrame) -> list[list[str]]:
    sentences: list[list[str]] = []
    for article in train_df["article"].astype(str):
        parts = split_sentences(article)
        if not parts and article.strip():
            parts = [article.strip()[:500]]
        for sent in parts:
            t = _w2v_tokens(sent)
            if len(t) >= 3:
                sentences.append(t)
    for question in train_df["question"].astype(str):
        t = _w2v_tokens(question)
        if len(t) >= 2:
            sentences.append(t)
    for label in OPTION_LABELS:
        for opt in train_df[label].astype(str):
            t = _w2v_tokens(opt)
            if len(t) >= 2:
                sentences.append(t)
    return sentences


def _train_word2vec(sentences: list[list[str]], vector_size: int = 128, window: int = 8, min_count: int = 3, epochs: int = 8):
    from gensim.models import Word2Vec

    if len(sentences) < 100:
        epochs = max(epochs, 15)
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        sg=1,
        epochs=epochs,
        seed=42,
    )
    return model.wv


def _load_questions(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        preprocess_all()
    return pd.read_csv(path).fillna("")


# ---------------------------------------------------------------------------
# Character-level overlap feature
# ---------------------------------------------------------------------------


def char_level_match(a: str, b: str) -> float:
    """Longest matching substring length divided by max length, in [0, 1]."""
    a = (a or "").lower()
    b = (b or "").lower()
    if not a or not b:
        return 0.0
    matcher = SequenceMatcher(a=a, b=b, autojunk=False)
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    return float(match.size) / float(max(len(a), len(b)))


# ---------------------------------------------------------------------------
# Distractor candidate features
# ---------------------------------------------------------------------------


def _content_tokens(text: str) -> set[str]:
    return {token for token in str(text).lower().replace("-", " ").split() if len(token.strip(".,;:!?()[]\"'")) > 2}


def _near_duplicate(candidate: str, answer: str) -> bool:
    candidate_tokens = _content_tokens(candidate)
    answer_tokens = _content_tokens(answer)
    if not candidate_tokens or not answer_tokens:
        return False
    overlap = len(candidate_tokens & answer_tokens)
    jaccard = overlap / len(candidate_tokens | answer_tokens)
    return jaccard >= 0.34 or (overlap > 0 and min(len(candidate_tokens), len(answer_tokens)) <= 2)


DISTRACTOR_FEATURE_NAMES = (
    "tfidf_answer_sim",
    "tfidf_question_sim",
    "w2v_answer_sim",
    "w2v_question_sim",
    "passage_frequency",
    "char_overlap_with_answer",
    "candidate_length",
    "shares_token_with_answer",
)


def compute_distractor_features(
    article: str,
    question: str,
    answer: str,
    candidates: list[str],
    vectorizer: TfidfVectorizer,
    w2v: "KeyedVectors | None" = None,
) -> np.ndarray:
    if not candidates:
        return np.zeros((0, len(DISTRACTOR_FEATURE_NAMES)), dtype=float)
    candidate_x = vectorizer.transform(candidates)
    answer_x = vectorizer.transform([answer or ""])
    question_x = vectorizer.transform([question or ""])
    tfidf_answer = cosine_similarity(candidate_x, answer_x).ravel()
    tfidf_question = cosine_similarity(candidate_x, question_x).ravel()
    if w2v is not None:
        w2v_answer = _w2v_cosine_to_ref(candidates, answer or "", w2v)
        w2v_question = _w2v_cosine_to_ref(candidates, question or "", w2v)
    else:
        w2v_answer = np.full(len(candidates), 0.0, dtype=np.float64)
        w2v_question = np.full(len(candidates), 0.0, dtype=np.float64)
    article_lower = (article or "").lower()
    answer_tokens = _content_tokens(answer)
    rows = np.zeros((len(candidates), len(DISTRACTOR_FEATURE_NAMES)), dtype=float)
    max_freq = max(article_lower.count(c.lower()) for c in candidates) if candidates else 0
    max_freq = max(max_freq, 1)
    for i, candidate in enumerate(candidates):
        freq = article_lower.count(candidate.lower())
        rows[i, 0] = float(tfidf_answer[i])
        rows[i, 1] = float(tfidf_question[i])
        rows[i, 2] = float(w2v_answer[i])
        rows[i, 3] = float(w2v_question[i])
        rows[i, 4] = float(freq) / float(max_freq)
        rows[i, 5] = char_level_match(candidate, answer or "")
        rows[i, 6] = float(len(candidate.split()))
        rows[i, 7] = float(bool(_content_tokens(candidate) & answer_tokens))
    return rows


# ---------------------------------------------------------------------------
# Frequency-Based Substitution distractor alternative
# ---------------------------------------------------------------------------


def frequency_substitution_distractors(
    article: str,
    answer: str,
    existing_options: list[str] | None = None,
    top_n: int = 3,
) -> list[dict[str, float | str]]:
    """Pick distractors by selecting article phrases with similar frequency to the answer."""
    answer_key = (answer or "").strip().lower()
    excluded = {answer_key}
    for opt in existing_options or []:
        excluded.add((opt or "").strip().lower())
    candidates = extract_candidate_phrases(article)
    article_lower = (article or "").lower()
    target_frequency = article_lower.count(answer_key) if answer_key else 0
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.strip().lower()
        if not key or key in excluded or key in seen or _near_duplicate(candidate, answer or ""):
            continue
        seen.add(key)
        freq = article_lower.count(key)
        delta = abs(freq - target_frequency)
        length_penalty = abs(len(key) - len(answer_key)) / max(len(answer_key), 1)
        scored.append((float(delta) + 0.25 * float(length_penalty), candidate.strip()))
    scored.sort(key=lambda x: x[0])
    out: list[dict[str, float | str]] = []
    for closeness, text in scored[:top_n]:
        out.append({"text": text, "score": float(1.0 / (1.0 + closeness))})
    return out


# ---------------------------------------------------------------------------
# ML-trained distractor ranker
# ---------------------------------------------------------------------------


def _build_distractor_training_set(
    train_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    w2v: "KeyedVectors | None",
    sample_rows: int = 1500,
    negatives_per_row: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(train_df) > sample_rows:
        idx = rng.choice(len(train_df), size=sample_rows, replace=False)
        sample = train_df.iloc[idx]
    else:
        sample = train_df
    feature_rows: list[np.ndarray] = []
    label_rows: list[int] = []
    for row in sample.itertuples(index=False):
        article = str(getattr(row, "article", ""))
        question = str(getattr(row, "question", ""))
        answer_letter = str(getattr(row, "answer", "")).strip().upper()
        if answer_letter not in OPTION_LABELS:
            continue
        answer_text = str(getattr(row, answer_letter, "")).strip()
        if not answer_text:
            continue
        gold_distractors = [
            str(getattr(row, label, "")).strip()
            for label in OPTION_LABELS
            if label != answer_letter and str(getattr(row, label, "")).strip()
        ]
        article_phrases = extract_candidate_phrases(article)
        candidates: list[str] = []
        labels: list[int] = []
        seen: set[str] = set()
        for distractor in gold_distractors:
            key = distractor.lower()
            if key in seen:
                continue
            seen.add(key)
            candidates.append(distractor)
            labels.append(1)
        negatives = []
        for phrase in article_phrases:
            key = phrase.strip().lower()
            if not key or key in seen or key == answer_text.lower():
                continue
            if any(_near_duplicate(phrase, gd) for gd in gold_distractors):
                continue
            negatives.append(phrase.strip())
            seen.add(key)
            if len(negatives) >= negatives_per_row:
                break
        for negative in negatives:
            candidates.append(negative)
            labels.append(0)
        if len(candidates) < 2:
            continue
        features = compute_distractor_features(article, question, answer_text, candidates, vectorizer, w2v)
        feature_rows.append(features)
        label_rows.extend(labels)
    if not feature_rows:
        return np.zeros((0, len(DISTRACTOR_FEATURE_NAMES))), np.zeros(0, dtype=int)
    return np.vstack(feature_rows), np.asarray(label_rows, dtype=int)


def _train_distractor_ranker(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    w2v: "KeyedVectors | None",
) -> tuple[LogisticRegression | None, dict[str, object]]:
    x_train, y_train = _build_distractor_training_set(train_df, vectorizer, w2v, sample_rows=1500)
    x_val, y_val = _build_distractor_training_set(val_df, vectorizer, w2v, sample_rows=400, seed=43)
    if len(x_train) < 50 or len(set(y_train.tolist())) < 2:
        return None, {"trained": False, "reason": "not enough samples"}
    classifier = LogisticRegression(max_iter=400, solver="liblinear", random_state=42, class_weight="balanced")
    classifier.fit(x_train, y_train)
    metrics: dict[str, object] = {
        "trained": True,
        "n_train_pairs": int(len(y_train)),
        "n_val_pairs": int(len(y_val)),
        "feature_names": list(DISTRACTOR_FEATURE_NAMES),
        "feature_weights": dict(zip(DISTRACTOR_FEATURE_NAMES, classifier.coef_.ravel().tolist())),
        "intercept": float(classifier.intercept_[0]),
    }
    if len(x_val) and len(set(y_val.tolist())) > 1:
        pred = classifier.predict(x_val)
        metrics["validation_accuracy"] = float(accuracy_score(y_val, pred))
        metrics["validation_macro_f1"] = float(f1_score(y_val, pred, average="macro", zero_division=0))
        metrics["validation_precision"] = float(precision_score(y_val, pred, zero_division=0))
        metrics["validation_recall"] = float(recall_score(y_val, pred, zero_division=0))
    return classifier, metrics


# ---------------------------------------------------------------------------
# Generation candidate sentence ranker (kept from earlier work)
# ---------------------------------------------------------------------------


def _leaves_enough_remainder(answer: str, sentences: list[str], min_words: int = 4) -> bool:
    """True if at least one sentence has >= *min_words* content words remaining
    after the answer span is removed.  Prevents cloze stems like ``____.``
    when the candidate answer dominates a short sentence."""
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
    """Build candidate answer spans for cloze / Wh question generation.

    Long noun phrases are great *distractors* but make poor cloze *answers*
    (the redacted sentence collapses to "In ____?").  So we cap the
    candidate length at 8 words and require >= 4 content words to remain in
    the sentence after redaction.  The definition pathway in ``inference``
    handles longer predicate clauses separately.
    """
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


# ---------------------------------------------------------------------------
# Public ranker used by inference
# ---------------------------------------------------------------------------


def rank_distractors(
    article: str,
    question: str,
    answer: str,
    vectorizer: TfidfVectorizer,
    existing_options: list[str] | None = None,
    w2v: "KeyedVectors | None" = None,
    distractor_w2v_blend: float = 0.38,
    diversity_w2v_weight: float = 0.12,
    ranker: LogisticRegression | None = None,
    ranker_weight: float = 0.6,
) -> list[dict[str, float | str]]:
    existing_options = existing_options or []
    answer_key = answer.strip().lower()
    candidates = extract_candidate_phrases(article)
    candidates.extend(option for option in existing_options if option)
    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.strip().lower()
        if not key or key == answer_key or answer_key in key or key in answer_key or key in seen or _near_duplicate(candidate, answer):
            continue
        seen.add(key)
        unique.append(candidate.strip())
    if not unique:
        return []

    candidate_x = vectorizer.transform(unique)
    answer_x = vectorizer.transform([answer])
    question_x = vectorizer.transform([question])
    answer_sim_tfidf = cosine_similarity(candidate_x, answer_x).ravel()
    question_sim_tfidf = cosine_similarity(candidate_x, question_x).ravel()

    if w2v is not None:
        answer_sim = _blend_tfidf_w2v(answer_sim_tfidf, _w2v_cosine_to_ref(unique, answer, w2v), distractor_w2v_blend)
        question_sim = _blend_tfidf_w2v(question_sim_tfidf, _w2v_cosine_to_ref(unique, question, w2v), distractor_w2v_blend)
    else:
        answer_sim = answer_sim_tfidf
        question_sim = question_sim_tfidf

    article_lower = article.lower()
    frequency = np.array([article_lower.count(candidate.lower()) for candidate in unique], dtype=float)
    if frequency.max() > 0:
        frequency = frequency / frequency.max()
    char_overlap = np.array([char_level_match(candidate, answer) for candidate in unique], dtype=float)
    base_scores = (
        (0.45 * answer_sim)
        + (0.30 * question_sim)
        + (0.15 * frequency)
        + (0.10 * char_overlap)
    )

    if ranker is not None:
        feature_matrix = compute_distractor_features(article, question, answer, unique, vectorizer, w2v)
        try:
            ranker_probabilities = ranker.predict_proba(feature_matrix)[:, 1]
        except Exception:
            ranker_probabilities = base_scores
        normalized_base = _normalize01(base_scores)
        normalized_ranker = _normalize01(ranker_probabilities)
        scores = (1.0 - ranker_weight) * normalized_base + ranker_weight * normalized_ranker
    else:
        scores = base_scores

    # Length-shape matching: a long phrase answer should attract long-phrase
    # distractors and a single-word answer should attract single-word ones.
    answer_wc = max(1, len(answer.split()))
    candidate_wcs = np.array([max(1, len(candidate.split())) for candidate in unique], dtype=float)
    length_ratio = np.minimum(candidate_wcs, answer_wc) / np.maximum(candidate_wcs, answer_wc)
    length_weight = 0.55 + 0.45 * length_ratio  # in [0.55, 1.0]
    scores = scores * length_weight

    ranked: list[dict[str, float | str]] = []
    selected_vectors: list = []
    selected_w2v: list[np.ndarray] = []
    for index in np.argsort(scores)[::-1]:
        candidate = unique[int(index)]
        vector = candidate_x[int(index)]
        diversity_penalty = 0.0
        if selected_vectors:
            diversity_penalty = float(max(cosine_similarity(vector, other).ravel()[0] for other in selected_vectors))
        if w2v is not None and selected_w2v:
            u = _text_to_unit_vector(candidate, w2v)
            if u is not None:
                w_div = max(float(np.dot(u, prev)) for prev in selected_w2v)
                diversity_penalty += diversity_w2v_weight * w_div
        final_score = float(scores[int(index)] - (0.25 * diversity_penalty))
        if final_score <= 0 and len(ranked) >= 3:
            continue
        ranked.append({"text": candidate, "score": final_score})
        selected_vectors.append(vector)
        if w2v is not None:
            u = _text_to_unit_vector(candidate, w2v)
            if u is not None:
                selected_w2v.append(u)
        if len(ranked) == 3:
            break
    return ranked


# ---------------------------------------------------------------------------
# Hint generation: extractive (existing) + ML-scored (new)
# ---------------------------------------------------------------------------


HINT_FEATURE_NAMES = (
    "keyword_overlap",
    "position_norm",
    "length",
    "first_token_match",
    "contains_answer_token",
)


def _hint_features(question: str, answer: str, sentences: list[str]) -> np.ndarray:
    if not sentences:
        return np.zeros((0, len(HINT_FEATURE_NAMES)), dtype=float)
    question_tokens = set(_w2v_tokens(question))
    answer_tokens = set(_w2v_tokens(answer))
    rows = np.zeros((len(sentences), len(HINT_FEATURE_NAMES)), dtype=float)
    n = len(sentences)
    for i, sentence in enumerate(sentences):
        tokens = _w2v_tokens(sentence)
        if not tokens:
            continue
        token_set = set(tokens)
        overlap = len(question_tokens & token_set) / max(len(question_tokens), 1)
        rows[i, 0] = float(overlap)
        rows[i, 1] = float(i) / float(max(n - 1, 1))
        rows[i, 2] = float(len(tokens))
        rows[i, 3] = float(bool(question_tokens) and tokens[0] in question_tokens)
        rows[i, 4] = float(bool(token_set & answer_tokens))
    return rows


def _build_hint_training_set(
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    sample_rows: int = 1500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(df) > sample_rows:
        idx = rng.choice(len(df), size=sample_rows, replace=False)
        sample = df.iloc[idx]
    else:
        sample = df
    feature_rows: list[np.ndarray] = []
    target_rows: list[float] = []
    for row in sample.itertuples(index=False):
        article = str(getattr(row, "article", ""))
        question = str(getattr(row, "question", ""))
        answer_letter = str(getattr(row, "answer", "")).strip().upper()
        answer_text = str(getattr(row, answer_letter, "")) if answer_letter in OPTION_LABELS else ""
        sentences = split_sentences(article)
        if len(sentences) < 2:
            continue
        sent_x = vectorizer.transform(sentences)
        question_x = vectorizer.transform([question])
        sims = cosine_similarity(sent_x, question_x).ravel()
        feats = _hint_features(question, answer_text, sentences)
        feature_rows.append(feats)
        target_rows.extend(sims.tolist())
    if not feature_rows:
        return np.zeros((0, len(HINT_FEATURE_NAMES))), np.zeros(0, dtype=float)
    return np.vstack(feature_rows), np.asarray(target_rows, dtype=float)


def _train_hint_scorer(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
) -> tuple[Ridge | None, dict[str, object]]:
    x_train, y_train = _build_hint_training_set(train_df, vectorizer, sample_rows=1500)
    x_val, y_val = _build_hint_training_set(val_df, vectorizer, sample_rows=400, seed=43)
    if len(x_train) < 100:
        return None, {"trained": False, "reason": "not enough samples"}
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(x_train, y_train)
    metrics: dict[str, object] = {
        "trained": True,
        "n_train_sentences": int(len(y_train)),
        "n_val_sentences": int(len(y_val)),
        "feature_names": list(HINT_FEATURE_NAMES),
        "feature_weights": dict(zip(HINT_FEATURE_NAMES, model.coef_.tolist())),
        "intercept": float(model.intercept_),
    }
    if len(x_val) > 1:
        predictions = model.predict(x_val)
        metrics["validation_r2"] = float(r2_score(y_val, predictions))
    return model, metrics


def generate_hints(
    article: str,
    question: str,
    answer: str,
    vectorizer: TfidfVectorizer,
    w2v: "KeyedVectors | None" = None,
    hint_w2v_blend: float = 0.42,
    hint_scorer: Ridge | None = None,
    hint_scorer_weight: float = 0.4,
) -> list[str]:
    sentences = split_sentences(article)
    if not sentences:
        return [
            "Review the passage for the part related to the question.",
            "Look for wording that overlaps with the question.",
            "The answer is supported directly by the passage.",
        ]
    sentence_x = vectorizer.transform(sentences)
    question_x = vectorizer.transform([question])
    scores_tfidf = cosine_similarity(sentence_x, question_x).ravel()
    if w2v is not None:
        scores_w2v = _w2v_cosine_to_ref(sentences, question, w2v)
        scores = _blend_tfidf_w2v(scores_tfidf, scores_w2v, hint_w2v_blend)
    else:
        scores = scores_tfidf
    if hint_scorer is not None:
        feats = _hint_features(question, answer, sentences)
        try:
            ml_scores = hint_scorer.predict(feats)
            scores = (1.0 - hint_scorer_weight) * _normalize01(scores) + hint_scorer_weight * _normalize01(ml_scores)
        except Exception:
            pass
    ordered = [sentences[int(index)] for index in np.argsort(scores)[::-1]]
    support = ordered[0]
    secondary = ordered[1] if len(ordered) > 1 else support
    near = support.replace(answer, "____") if answer else support
    return [
        "Focus on the sentence group that discusses the main subject of the question.",
        secondary,
        near,
    ]


# ---------------------------------------------------------------------------
# Distractor + Hint evaluation on the validation split
# ---------------------------------------------------------------------------


def _set_overlap_metrics(predicted: list[str], gold: list[str]) -> tuple[float, float, float]:
    pred_set = {p.strip().lower() for p in predicted if p}
    gold_set = {g.strip().lower() for g in gold if g}
    if not pred_set and not gold_set:
        return 1.0, 1.0, 1.0
    if not pred_set or not gold_set:
        return 0.0, 0.0, 0.0
    matched = 0
    for p in pred_set:
        if any(p == g or p in g or g in p for g in gold_set):
            matched += 1
    precision = matched / max(len(pred_set), 1)
    recall_matched = sum(1 for g in gold_set if any(g == p or g in p or p in g for p in pred_set))
    recall = recall_matched / max(len(gold_set), 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _evaluate_distractors(
    val_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    w2v: "KeyedVectors | None",
    ranker: LogisticRegression | None,
    sample_rows: int = 400,
    seed: int = 42,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    if len(val_df) > sample_rows:
        idx = rng.choice(len(val_df), size=sample_rows, replace=False)
        sample = val_df.iloc[idx]
    else:
        sample = val_df
    p_list, r_list, f_list = [], [], []
    top1_not_answer = 0
    total = 0
    for row in sample.itertuples(index=False):
        article = str(getattr(row, "article", ""))
        question = str(getattr(row, "question", ""))
        answer_letter = str(getattr(row, "answer", "")).strip().upper()
        if answer_letter not in OPTION_LABELS:
            continue
        answer_text = str(getattr(row, answer_letter, "")).strip()
        if not answer_text:
            continue
        gold_distractors = [
            str(getattr(row, label, "")).strip()
            for label in OPTION_LABELS
            if label != answer_letter and str(getattr(row, label, "")).strip()
        ]
        predicted = rank_distractors(article, question, answer_text, vectorizer, w2v=w2v, ranker=ranker)
        predicted_texts = [str(p["text"]) for p in predicted]
        if not predicted_texts:
            continue
        p, r, f = _set_overlap_metrics(predicted_texts, gold_distractors)
        p_list.append(p)
        r_list.append(r)
        f_list.append(f)
        if predicted_texts[0].strip().lower() != answer_text.lower():
            top1_not_answer += 1
        total += 1
    return {
        "n_questions": int(total),
        "precision": float(np.mean(p_list)) if p_list else 0.0,
        "recall": float(np.mean(r_list)) if r_list else 0.0,
        "f1": float(np.mean(f_list)) if f_list else 0.0,
        "ranker_top1_not_answer_accuracy": float(top1_not_answer / total) if total else 0.0,
    }


def _evaluate_hints(
    val_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    w2v: "KeyedVectors | None",
    hint_scorer: Ridge | None,
    sample_rows: int = 400,
    top_k: int = 3,
    seed: int = 42,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    if len(val_df) > sample_rows:
        idx = rng.choice(len(val_df), size=sample_rows, replace=False)
        sample = val_df.iloc[idx]
    else:
        sample = val_df
    precisions: list[float] = []
    contains_answer_topk: list[float] = []
    total = 0
    for row in sample.itertuples(index=False):
        article = str(getattr(row, "article", ""))
        question = str(getattr(row, "question", ""))
        answer_letter = str(getattr(row, "answer", "")).strip().upper()
        if answer_letter not in OPTION_LABELS:
            continue
        answer_text = str(getattr(row, answer_letter, "")).strip()
        if not answer_text:
            continue
        sentences = split_sentences(article)
        if len(sentences) < 2:
            continue
        sent_x = vectorizer.transform(sentences)
        question_x = vectorizer.transform([question])
        scores = cosine_similarity(sent_x, question_x).ravel()
        if hint_scorer is not None:
            feats = _hint_features(question, answer_text, sentences)
            try:
                ml_scores = hint_scorer.predict(feats)
                scores = 0.6 * _normalize01(scores) + 0.4 * _normalize01(ml_scores)
            except Exception:
                pass
        order = np.argsort(scores)[::-1][:top_k]
        gold_sentences = [s for s in sentences if answer_text.lower() in s.lower()]
        gold_set = {s.strip().lower() for s in gold_sentences}
        if not gold_set:
            continue
        retrieved = [sentences[int(i)].strip().lower() for i in order]
        hits = sum(1 for r in retrieved if r in gold_set)
        precisions.append(float(hits) / float(top_k))
        contains_answer_topk.append(float(any(answer_text.lower() in r for r in retrieved)))
        total += 1
    return {
        "n_questions": int(total),
        f"precision_at_{top_k}": float(np.mean(precisions)) if precisions else 0.0,
        f"top_{top_k}_contains_answer_rate": float(np.mean(contains_answer_topk)) if contains_answer_topk else 0.0,
    }


# ---------------------------------------------------------------------------
# Train entrypoint
# ---------------------------------------------------------------------------


def train(model_dir: Path = MODEL_DIR) -> dict[str, object]:
    train_df = _load_questions("train")
    try:
        val_df = _load_questions("validation")
    except Exception:
        val_df = train_df.tail(0)
    corpus = pd.concat(
        [
            train_df["article"].astype(str),
            train_df["question"].astype(str),
            train_df[list(OPTION_LABELS)].astype(str).stack().reset_index(drop=True),
        ],
        ignore_index=True,
    )
    vectorizer = TfidfVectorizer(stop_words="english", max_features=25000, sublinear_tf=True, norm="l2", ngram_range=(1, 2))
    vectorizer.fit(corpus)

    w2v_sentences = _w2v_training_sentences(train_df)
    w2v_sentence_total = len(w2v_sentences)
    max_w2v_sents = 200_000
    if len(w2v_sentences) > max_w2v_sents:
        w2v_sentences = random.Random(42).sample(w2v_sentences, max_w2v_sents)
    wv = _train_word2vec(w2v_sentences)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    w2v_path = model_dir / W2V_KV_NAME
    wv.save(str(w2v_path))

    distractor_ranker, distractor_ranker_metrics = _train_distractor_ranker(train_df, val_df, vectorizer, wv)
    if distractor_ranker is not None:
        joblib.dump(distractor_ranker, model_dir / "distractor_ranker.joblib")

    hint_scorer, hint_scorer_metrics = _train_hint_scorer(train_df, val_df, vectorizer)
    if hint_scorer is not None:
        joblib.dump(hint_scorer, model_dir / "hint_scorer.joblib")

    distractor_eval = _evaluate_distractors(val_df, vectorizer, wv, distractor_ranker)
    hint_eval = _evaluate_hints(val_df, vectorizer, wv, hint_scorer)

    w2v_cfg = {
        "file": W2V_KV_NAME,
        "vector_size": int(wv.vector_size),
        "vocab_size": len(wv),
        "distractor_blend_weight": 0.38,
        "hint_blend_weight": 0.42,
        "generation_blend_weight": 0.38,
    }
    config = {
        "max_candidates": 80,
        "distractor_count": 3,
        "hint_count": 3,
        "distractor_ranker_weight": 0.6,
        "hint_scorer_weight": 0.4,
        "word2vec": w2v_cfg,
        "distractor_ranker": distractor_ranker_metrics,
        "hint_scorer": hint_scorer_metrics,
        "evaluation": {
            "distractors": distractor_eval,
            "hints": hint_eval,
        },
    }
    with (model_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return {
        "trained_on_rows": len(train_df),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "word2vec_vocab": len(wv),
        "word2vec_sentence_total": w2v_sentence_total,
        "word2vec_sentences_used": len(w2v_sentences),
        "max_candidates": config["max_candidates"],
        "distractor_count": config["distractor_count"],
        "hint_count": config["hint_count"],
        "distractor_ranker": distractor_ranker_metrics,
        "hint_scorer": hint_scorer_metrics,
        "evaluation": config["evaluation"],
        "word2vec": w2v_cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model B TF-IDF + Word2Vec for distractors and hints.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    print(json.dumps(train(args.model_dir), indent=2, default=str))


if __name__ == "__main__":
    main()
