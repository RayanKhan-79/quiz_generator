"""End-to-end evaluation across Model A (verifier), Model B (distractors + hints).

Run as ``python -m src.evaluate``. Writes ``models/evaluation_metrics.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.inference import QuizEngine
from src.model_b_train import _evaluate_distractors, _evaluate_hints
from src.preprocessing import OPTION_LABELS, PROCESSED_DIR, preprocess_all

ROOT = Path(__file__).resolve().parents[1]


def evaluate_model_a(engine: QuizEngine, df: pd.DataFrame) -> dict[str, object]:
    if not engine.model_a_loaded:
        return {"reason": "Model A artifacts missing"}
    y_true: list[str] = []
    y_pred: list[str] = []
    for row in df.itertuples(index=False):
        options = {label: str(getattr(row, label)) for label in OPTION_LABELS}
        result = engine.verify(row.article, row.question, options, row.answer)
        y_true.append(str(row.answer))
        y_pred.append(str(result["predicted_option"]))
    return {
        "rows": int(len(y_true)),
        "exact_match_answer_accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(OPTION_LABELS)).tolist(),
    }


def evaluate_split(split: str = "test", limit: int | None = None, distractor_sample: int = 400, hint_sample: int = 400) -> dict[str, object]:
    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        preprocess_all()
    df = pd.read_csv(path).fillna("")
    if limit:
        df = df.head(limit)

    engine = QuizEngine()
    model_a_metrics = evaluate_model_a(engine, df)

    if engine.model_b_loaded:
        distractor_metrics = _evaluate_distractors(
            df,
            engine.vectorizer_b,
            getattr(engine, "w2v_b", None),
            getattr(engine, "distractor_ranker", None),
            sample_rows=distractor_sample,
        )
        hint_metrics = _evaluate_hints(
            df,
            engine.vectorizer_b,
            getattr(engine, "w2v_b", None),
            getattr(engine, "hint_scorer", None),
            sample_rows=hint_sample,
        )
    else:
        distractor_metrics = {"reason": "Model B artifacts missing"}
        hint_metrics = {"reason": "Model B artifacts missing"}

    metrics = {
        "split": split,
        "rows": int(len(df)),
        "model_a": model_a_metrics,
        "model_b_distractors": distractor_metrics,
        "model_b_hints": hint_metrics,
    }
    output = ROOT / "models" / "evaluation_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Model A verifier and Model B distractor/hint quality.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--distractor-sample", type=int, default=400)
    parser.add_argument("--hint-sample", type=int, default=400)
    args = parser.parse_args()
    metrics = evaluate_split(
        split=args.split,
        limit=args.limit,
        distractor_sample=args.distractor_sample,
        hint_sample=args.hint_sample,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
