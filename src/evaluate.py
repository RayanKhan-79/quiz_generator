from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.inference import QuizEngine
from src.preprocessing import OPTION_LABELS, PROCESSED_DIR, preprocess_all

ROOT = Path(__file__).resolve().parents[1]


def evaluate_split(split: str = "test", limit: int | None = None) -> dict[str, object]:
    path = PROCESSED_DIR / f"{split}.csv"
    if not path.exists():
        preprocess_all()
    df = pd.read_csv(path).fillna("")
    if limit:
        df = df.head(limit)
    engine = QuizEngine()
    correct = 0
    total = 0
    for row in df.itertuples(index=False):
        options = {label: getattr(row, label) for label in OPTION_LABELS}
        result = engine.verify(row.article, row.question, options, row.answer)
        correct += int(result["predicted_option"] == row.answer)
        total += 1
    metrics = {"split": split, "rows": total, "exact_match_answer_accuracy": correct / total if total else 0.0}
    output = ROOT / "models" / "evaluation_metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


if __name__ == "__main__":
    print(json.dumps(evaluate_split(), indent=2))
