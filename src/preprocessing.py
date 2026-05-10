from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
OPTION_LABELS = ("A", "B", "C", "D")
REQUIRED_COLUMNS = ("id", "article", "question", "A", "B", "C", "D", "answer")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_full_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    dataset_path = raw_dir / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Could not find dataset.csv at {dataset_path}")
    return pd.read_csv(dataset_path)


def _split_dataset(df: pd.DataFrame, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into train (80%), validation (10%), and test (10%)."""
    # First split: 80% train, 20% temp (validation + test)
    train, temp = _train_test_split(df, test_size=0.2, random_state=random_state)
    # Second split: Split temp into equal parts (50-50 = 10-10 of original)
    validation, test = _train_test_split(temp, test_size=0.5, random_state=random_state)
    return train, validation, test


def _train_test_split(df: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into two parts."""
    shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(shuffled) * (1 - test_size))
    return shuffled[:split_idx], shuffled[split_idx:]


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lower_map = {column.lower(): column for column in df.columns}
    for name in names:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _extract_option(options: object, index: int) -> str:
    if isinstance(options, (list, tuple)):
        return clean_text(options[index]) if len(options) > index else ""
    if hasattr(options, "__len__") and hasattr(options, "__getitem__") and not isinstance(options, (str, bytes, dict)):
        return clean_text(options[index]) if len(options) > index else ""
    if isinstance(options, str):
        stripped = options.strip()
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return clean_text(parsed[index]) if len(parsed) > index else ""
        except json.JSONDecodeError:
            pass
        parts = re.split(r"\s*(?:\||;|\n)\s*", stripped)
        return clean_text(parts[index]) if len(parts) > index else ""
    return ""


def _normalize_answer(value: object) -> str:
    text = clean_text(value).upper()
    if text in OPTION_LABELS:
        return text
    if text in {"0", "1", "2", "3"}:
        return OPTION_LABELS[int(text)]
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    raise ValueError(f"Unsupported answer label: {value!r}")


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    article_col = _first_existing(df, ["article", "passage", "context", "text"])
    question_col = _first_existing(df, ["question", "query", "prompt"])
    answer_col = _first_existing(df, ["answer", "label", "correct", "correct_answer"])
    id_col = _first_existing(df, ["id", "example_id", "qid"])
    options_col = _first_existing(df, ["options", "choices"])
    if not article_col or not question_col or not answer_col:
        raise ValueError("Input must contain article/passage, question, and answer columns.")

    out = pd.DataFrame()
    out["id"] = df[id_col].map(clean_text) if id_col else [f"row-{i}" for i in range(len(df))]
    out["article"] = df[article_col].map(clean_text)
    out["question"] = df[question_col].map(clean_text)

    for idx, label in enumerate(OPTION_LABELS):
        option_col = _first_existing(df, [label, label.lower(), f"option_{label}", f"option{label}", f"choice_{label}"])
        if option_col:
            out[label] = df[option_col].map(clean_text)
        elif options_col:
            out[label] = df[options_col].map(lambda value, i=idx: _extract_option(value, i))
        else:
            raise ValueError("Input must contain A/B/C/D columns or an options/choices column.")

    out["answer"] = df[answer_col].map(_normalize_answer)
    missing = [column for column in REQUIRED_COLUMNS if column not in out.columns]
    if missing:
        raise ValueError(f"Missing normalized columns: {missing}")
    return out[list(REQUIRED_COLUMNS)]


def expand_options(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        base = row._asdict()
        for label in OPTION_LABELS:
            option_text = base[label]
            rows.append(
                {
                    "id": base["id"],
                    "article": base["article"],
                    "question": base["question"],
                    "option_label": label,
                    "option_text": option_text,
                    "answer": base["answer"],
                    "label": int(label == base["answer"]),
                    "verification_text": f"{base['article']} [QUESTION] {base['question']} [OPTION] {option_text}",
                }
            )
    return pd.DataFrame(rows)


def _preprocess_split_df(df: pd.DataFrame, split: str, processed_dir: Path = PROCESSED_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Process a dataframe split and save it."""
    normalized = normalize_schema(df)
    expanded = expand_options(normalized)
    processed_dir.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(processed_dir / f"{split}.csv", index=False)
    expanded.to_csv(processed_dir / f"{split}_options.csv", index=False)
    return normalized, expanded


def preprocess_all(raw_dir: Path = RAW_DIR, processed_dir: Path = PROCESSED_DIR) -> dict[str, dict[str, int]]:
    # Read the full dataset and perform 80-10-10 split
    full_dataset = _read_full_dataset(raw_dir)
    train_df, validation_df, test_df = _split_dataset(full_dataset)
    
    summary: dict[str, dict[str, int]] = {}
    for split_name, split_df in [("train", train_df), ("validation", validation_df), ("test", test_df)]:
        normalized, expanded = _preprocess_split_df(split_df, split_name, processed_dir)
        summary[split_name] = {"questions": len(normalized), "option_rows": len(expanded), "positives": int(expanded["label"].sum())}
    
    with (processed_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize RACE data and expand option-level rows.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    summary = preprocess_all(args.raw_dir, args.processed_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
