from __future__ import annotations

import csv
import io
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.model_a_train import MODEL_DIR as MODEL_A_DIR
from src.model_a_train import build_feature_blocks
from src.model_b_train import MODEL_DIR as MODEL_B_DIR
from src.model_b_train import generate_hints, rank_distractors
from src.preprocessing import OPTION_LABELS, PROCESSED_DIR, preprocess_all
from src.text_utils import extract_answer_candidates, redact_answer, split_sentences

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SessionStore:
    logs: list[dict[str, Any]] = field(default_factory=list)

    def add(self, record: dict[str, Any]) -> None:
        self.logs.append(record)

    def metrics(self) -> dict[str, Any]:
        latencies = [row["latency_ms"] for row in self.logs if "latency_ms" in row]
        return {
            "session_requests": len(self.logs),
            "average_latency_ms": round(float(np.mean(latencies)), 2) if latencies else 0.0,
            "last_requests": self.logs[-20:],
        }

    def csv(self) -> str:
        if not self.logs:
            return ""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=sorted({key for row in self.logs for key in row.keys()}))
        writer.writeheader()
        writer.writerows(self.logs)
        return output.getvalue()


class QuizEngine:
    def __init__(self, model_a_dir: Path = MODEL_A_DIR, model_b_dir: Path = MODEL_B_DIR):
        self.model_a_dir = model_a_dir
        self.model_b_dir = model_b_dir
        self.store = SessionStore()
        self.model_a_loaded = False
        self.model_b_loaded = False
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        try:
            self.vectorizer_a = joblib.load(self.model_a_dir / "tfidf_vectorizer.joblib")
            self.logistic = joblib.load(self.model_a_dir / "logistic_regression.joblib")
            self.svm = joblib.load(self.model_a_dir / "linear_svm_calibrated.joblib")
            self.question_vectorizer = joblib.load(self.model_a_dir / "question_tfidf_vectorizer.joblib")
            self.direct_logistic = joblib.load(self.model_a_dir / "direct_multiclass_logistic.joblib")
            self.model_a_loaded = True
        except Exception as exc:  # artifacts are optional during first setup
            self.model_a_error = str(exc)
        try:
            self.vectorizer_b = joblib.load(self.model_b_dir / "tfidf_vectorizer.joblib")
            self.model_b_loaded = True
        except Exception as exc:
            self.model_b_error = str(exc)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_a_loaded": self.model_a_loaded,
            "model_b_loaded": self.model_b_loaded,
            "model_a_error": getattr(self, "model_a_error", None),
            "model_b_error": getattr(self, "model_b_error", None),
        }

    def _question_frame(self, article: str, question: str, options: dict[str, str]) -> pd.DataFrame:
        rows = []
        for label in OPTION_LABELS:
            option = options.get(label, "")
            rows.append(
                {
                    "id": "request",
                    "article": article,
                    "question": question,
                    "option_label": label,
                    "option_text": option,
                    "answer": "",
                    "label": 0,
                    "verification_text": f"{article} [QUESTION] {question} [OPTION] {option}",
                }
            )
        return pd.DataFrame(rows)

    def verify(self, article: str, question: str, options: dict[str, str], selected_option: str, correct_option: str | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.model_a_loaded:
            raise RuntimeError("Model A artifacts are missing. Run: python -m src.model_a_train")
        df = self._question_frame(article, question, options)
        x = build_feature_blocks(df, self.vectorizer_a, fit=False)
        probabilities = self.logistic.predict_proba(x)[:, 1]
        best_index = int(np.argmax(probabilities))
        model_predicted = str(df.iloc[best_index]["option_label"])
        predicted = correct_option.upper() if correct_option else model_predicted
        selected = selected_option.upper()
        confidence_index = list(OPTION_LABELS).index(predicted) if predicted in OPTION_LABELS else best_index
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        result = {
            "predicted_option": predicted,
            "selected_option": selected,
            "is_correct": selected == predicted,
            "confidence": round(float(probabilities[confidence_index]), 4),
            "option_scores": {str(row.option_label): round(float(probabilities[i]), 4) for i, row in enumerate(df.itertuples())},
            "explanation": (
                f"This generated cloze question is keyed to option {predicted}; Model A independently ranked option {model_predicted} highest."
                if correct_option
                else f"TF-IDF verifier ranked option {predicted} highest using the trained ensemble plus article/question similarity evidence."
            ),
            "latency_ms": latency_ms,
        }
        self.store.add({"endpoint": "verify", "latency_ms": latency_ms, "predicted_option": predicted, "selected_option": selected})
        return result

    def generate(self, article: str, question: str | None = None, options: dict[str, str] | None = None, question_count: int = 5) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.model_b_loaded:
            raise RuntimeError("Model B artifacts are missing. Run: python -m src.model_b_train")
        options = options or {}
        generated_questions = self._question_specs(article, question, options, max(5, question_count))
        items = []
        for index, spec in enumerate(generated_questions):
            generated_question = spec["question"]
            answer = spec["answer"]
            distractors = rank_distractors(article, generated_question, answer, self.vectorizer_b, list(options.values()))
            final_options = self._merge_options(answer, distractors, options, seed=index)
            hints = self._hints_for_spec(article, generated_question, answer, spec.get("sentence", ""))
            confidence = 0.0
            predicted = self._label_for_answer(final_options, answer)
            if self.model_a_loaded:
                verification = self.verify(article, generated_question, final_options, "A")
                confidence = verification["confidence"]
            items.append(
                {
                    "question": generated_question,
                    "options": final_options,
                    "predicted_correct_option": predicted,
                    "predicted_answer_text": final_options[predicted],
                    "distractors": distractors,
                    "hints": hints,
                    "confidence": confidence,
                    "latency_ms": 0.0,
                    "ai_generated": True,
                }
            )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        per_item_latency = round(latency_ms / len(items), 2) if items else latency_ms
        for item in items:
            item["latency_ms"] = per_item_latency
        result = {
            "questions": items,
            "latency_ms": latency_ms,
            "ai_generated": True,
        }
        self.store.add({"endpoint": "generate", "latency_ms": latency_ms, "question_count": len(items)})
        return result

    def sample(self, debug: bool = False) -> dict[str, Any]:
        path = PROCESSED_DIR / "validation.csv"
        if not path.exists():
            preprocess_all()
        df = pd.read_csv(path).fillna("")
        row = df.iloc[random.randrange(len(df))]
        payload = {
            "id": row["id"],
            "article": row["article"],
            "question": row["question"],
            "options": {label: row[label] for label in OPTION_LABELS},
        }
        if debug:
            payload["answer"] = row["answer"]
        return payload

    def metrics(self) -> dict[str, Any]:
        metrics = self.store.metrics()
        for name, path in {
            "model_a": self.model_a_dir / "metrics.json",
            "model_b": self.model_b_dir / "config.json",
        }.items():
            if path.exists():
                with path.open("r", encoding="utf-8") as handle:
                    metrics[name] = json.load(handle)
        return metrics

    def _similarity_scores(self, article: str, question: str, options: dict[str, str]) -> np.ndarray:
        labels = list(OPTION_LABELS)
        article_x = self.vectorizer_a.transform([article])
        question_x = self.vectorizer_a.transform([question])
        option_x = self.vectorizer_a.transform([options.get(label, "") for label in labels])
        article_scores = (option_x @ article_x.T).toarray().ravel()
        question_scores = (option_x @ question_x.T).toarray().ravel()
        combined = (0.45 * article_scores) + (0.55 * question_scores)
        span = combined.max() - combined.min()
        if span <= 1e-9:
            return np.full(len(labels), 0.25)
        return (combined - combined.min()) / span

    def _direct_probabilities(self, article: str, question: str, options: dict[str, str]) -> np.ndarray:
        text = (
            f"{article} [QUESTION] {question} [A] {options.get('A', '')} [B] {options.get('B', '')} "
            f"[C] {options.get('C', '')} [D] {options.get('D', '')}"
        )
        x = self.question_vectorizer.transform([text])
        raw = self.direct_logistic.predict_proba(x)[0]
        by_label = {label: 0.0 for label in OPTION_LABELS}
        for label, probability in zip(self.direct_logistic.classes_, raw):
            by_label[str(label)] = float(probability)
        return np.array([by_label[label] for label in OPTION_LABELS], dtype=float)

    def _question_specs(self, article: str, provided_question: str | None, options: dict[str, str], count: int) -> list[dict[str, str]]:
        sentences = self._rank_generation_sentences(article)
        if not sentences:
            sentences = [article[index : index + 180] for index in range(0, max(len(article), 1), 180) if article[index : index + 180].strip()]
        specs: list[dict[str, str]] = []
        if provided_question and all(options.get(label) for label in OPTION_LABELS):
            specs.append({"question": provided_question, "answer": options["A"], "sentence": ""})
        index = 0
        while len(specs) < count:
            sentence = sentences[index % len(sentences)]
            answer = self._answer_for_sentence(sentence)
            redacted = redact_answer(sentence, answer, count=1)
            specs.append(
                {
                    "question": f"Which option best completes this sentence from the passage: {redacted}",
                    "answer": answer,
                    "sentence": sentence,
                }
            )
            index += 1
        return specs[:count]

    def _rank_generation_sentences(self, article: str) -> list[str]:
        sentences = split_sentences(article)
        scored = []
        for sentence in sentences:
            candidates = extract_answer_candidates(sentence, max_candidates=3)
            score = len(candidates) * 5 + min(len(sentence), 220) / 40
            if 45 <= len(sentence) <= 260 and candidates:
                scored.append((score, sentence))
        scored.sort(reverse=True, key=lambda item: item[0])
        return [sentence for _, sentence in scored] or sentences

    def _answer_for_sentence(self, sentence: str) -> str:
        candidates = extract_answer_candidates(sentence, max_candidates=8)
        if candidates:
            return candidates[0]
        words = [word.strip(".,;:!?()[]\"'") for word in sentence.split()]
        useful = [word for word in words if len(word) > 3]
        return " ".join(useful[:2] or words[:2]) or "the passage"

    def _hints_for_spec(self, article: str, question: str, answer: str, sentence: str) -> list[str]:
        if not sentence:
            return generate_hints(article, question, answer, self.vectorizer_b)
        return [
            "Look for the passage sentence that contains the missing detail.",
            redact_answer(sentence, answer, count=1),
            f"The missing phrase has {len(answer.split())} word(s) and appears in that sentence.",
        ]

    def _label_for_answer(self, options: dict[str, str], answer: str) -> str:
        answer_key = answer.strip().lower()
        for label in OPTION_LABELS:
            option_key = options.get(label, "").strip().lower()
            if option_key == answer_key:
                return label
        for label in OPTION_LABELS:
            option_key = options.get(label, "").strip().lower()
            if answer_key and (answer_key in option_key or option_key in answer_key):
                return label
        return "A"

    def _merge_options(self, answer: str, distractors: list[dict[str, Any]], options: dict[str, str], seed: int = 0) -> dict[str, str]:
        if all(options.get(label) for label in OPTION_LABELS):
            return {label: options[label] for label in OPTION_LABELS}
        values = []
        seen: set[str] = set()
        for value in [answer] + [str(item["text"]) for item in distractors]:
            key = value.strip().lower()
            if key and key not in seen:
                seen.add(key)
                values.append(value)
        while len(values) < 4:
            values.append(f"Not enough evidence {len(values)}")
        random.Random(42 + seed).shuffle(values)
        return {label: values[index] for index, label in enumerate(OPTION_LABELS)}
