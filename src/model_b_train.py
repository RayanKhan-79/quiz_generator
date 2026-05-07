from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import OPTION_LABELS, PROCESSED_DIR, preprocess_all
from src.text_utils import extract_candidate_phrases, split_sentences

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "model_b"


def _load_questions(split: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        preprocess_all()
    return pd.read_csv(path).fillna("")


def train(model_dir: Path = MODEL_DIR) -> dict[str, object]:
    train_df = _load_questions("train")
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
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    config = {"max_candidates": 80, "distractor_count": 3, "hint_count": 3}
    with (model_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    return {"trained_on_rows": len(train_df), "vocabulary_size": len(vectorizer.vocabulary_), **config}


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


def rank_distractors(article: str, question: str, answer: str, vectorizer: TfidfVectorizer, existing_options: list[str] | None = None) -> list[dict[str, float | str]]:
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
    answer_sim = cosine_similarity(candidate_x, answer_x).ravel()
    question_sim = cosine_similarity(candidate_x, question_x).ravel()
    frequency = np.array([article.lower().count(candidate.lower()) for candidate in unique], dtype=float)
    if frequency.max() > 0:
        frequency = frequency / frequency.max()
    scores = (0.50 * answer_sim) + (0.35 * question_sim) + (0.15 * frequency)
    ranked: list[dict[str, float | str]] = []
    selected_vectors = []
    for index in np.argsort(scores)[::-1]:
        candidate = unique[int(index)]
        vector = candidate_x[int(index)]
        diversity_penalty = 0.0
        if selected_vectors:
            diversity_penalty = float(max(cosine_similarity(vector, other).ravel()[0] for other in selected_vectors))
        final_score = float(scores[int(index)] - (0.25 * diversity_penalty))
        if final_score <= 0 and len(ranked) >= 3:
            continue
        ranked.append({"text": candidate, "score": final_score})
        selected_vectors.append(vector)
        if len(ranked) == 3:
            break
    return ranked


def generate_hints(article: str, question: str, answer: str, vectorizer: TfidfVectorizer) -> list[str]:
    sentences = split_sentences(article)
    if not sentences:
        return ["Review the passage for the part related to the question.", "Look for wording that overlaps with the question.", "The answer is supported directly by the passage."]
    sentence_x = vectorizer.transform(sentences)
    question_x = vectorizer.transform([question])
    scores = cosine_similarity(sentence_x, question_x).ravel()
    ordered = [sentences[int(index)] for index in np.argsort(scores)[::-1]]
    support = ordered[0]
    secondary = ordered[1] if len(ordered) > 1 else support
    near = support.replace(answer, "____") if answer else support
    return [
        "Focus on the sentence group that discusses the main subject of the question.",
        secondary,
        near,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Model B TF-IDF vectorizer for distractors and hints.")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    print(json.dumps(train(args.model_dir), indent=2))


if __name__ == "__main__":
    main()
