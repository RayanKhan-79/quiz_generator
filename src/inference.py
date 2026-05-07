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
from src.text_utils import redact_answer, split_sentences

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

    def verify(self, article: str, question: str, options: dict[str, str], selected_option: str) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.model_a_loaded:
            raise RuntimeError("Model A artifacts are missing. Run: python -m src.model_a_train")
        df = self._question_frame(article, question, options)
        x = build_feature_blocks(df, self.vectorizer_a, fit=False)
        lr_prob = self.logistic.predict_proba(x)[:, 1]
        svm_prob = self.svm.predict_proba(x)[:, 1]
        probabilities = (lr_prob + svm_prob) / 2.0
        best_index = int(np.argmax(probabilities))
        predicted = str(df.iloc[best_index]["option_label"])
        selected = selected_option.upper()
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        result = {
            "predicted_option": predicted,
            "selected_option": selected,
            "is_correct": selected == predicted,
            "confidence": round(float(probabilities[best_index]), 4),
            "option_scores": {str(row.option_label): round(float(probabilities[i]), 4) for i, row in enumerate(df.itertuples())},
            "explanation": f"TF-IDF verifier ranked option {predicted} highest for the passage and question.",
            "latency_ms": latency_ms,
        }
        self.store.add({"endpoint": "verify", "latency_ms": latency_ms, "predicted_option": predicted, "selected_option": selected})
        return result

    def generate(self, article: str, question: str | None = None, options: dict[str, str] | None = None, question_count: int = 5) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.model_b_loaded:
            raise RuntimeError("Model B artifacts are missing. Run: python -m src.model_b_train")
        options = options or {}
        generated_questions = self._template_questions(article, question, max(5, question_count))
        items = []
        for index, generated_question in enumerate(generated_questions):
            answer = self._choose_answer(article, generated_question, options, index=index)
            distractors = rank_distractors(article, generated_question, answer, self.vectorizer_b, list(options.values()))
            final_options = self._merge_options(answer, distractors, options, seed=index)
            hints = generate_hints(article, generated_question, answer, self.vectorizer_b)
            confidence = 0.0
            predicted = next((label for label, text in final_options.items() if text == answer), "A")
            if self.model_a_loaded:
                verification = self.verify(article, generated_question, final_options, "A")
                confidence = verification["confidence"]
                predicted = verification["predicted_option"]
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

    def _template_questions(self, article: str, provided_question: str | None, count: int) -> list[str]:
        sentences = split_sentences(article)
        if not sentences:
            sentences = [article[index : index + 180] for index in range(0, max(len(article), 1), 180) if article[index : index + 180].strip()]
        prompts = []
        if provided_question:
            prompts.append(provided_question)
        templates = [
            "Which option is best supported by this part of the passage: {snippet}?",
            "What is the main idea expressed in this sentence: {snippet}?",
            "Which detail from the passage is connected to this statement: {snippet}?",
            "What can the reader infer from this passage detail: {snippet}?",
            "Which option best completes this idea from the passage: {snippet}?",
            "What does the passage suggest about this information: {snippet}?",
            "Which answer is most consistent with this evidence: {snippet}?",
        ]
        index = 0
        while len(prompts) < count:
            sentence = sentences[index % len(sentences)]
            snippet = redact_answer(sentence, "")[:150]
            prompts.append(templates[index % len(templates)].format(snippet=snippet))
            index += 1
        return prompts[:count]

    def _choose_answer(self, article: str, question: str, options: dict[str, str], index: int = 0) -> str:
        for label in OPTION_LABELS:
            if options.get(label):
                return options[label]
        sentences = split_sentences(article)
        source = sentences[index % len(sentences)] if sentences else article
        words = [word.strip(".,;:!?()[]\"'") for word in source.split()]
        useful = [word for word in words if len(word) > 3]
        answer_words = useful[: min(4, len(useful))] or words[: min(4, len(words))]
        return " ".join(answer_words) or "the passage"

    def _merge_options(self, answer: str, distractors: list[dict[str, Any]], options: dict[str, str], seed: int = 0) -> dict[str, str]:
        if all(options.get(label) for label in OPTION_LABELS):
            return {label: options[label] for label in OPTION_LABELS}
        values = [answer] + [str(item["text"]) for item in distractors]
        while len(values) < 4:
            values.append(f"Not enough evidence {len(values)}")
        random.Random(42 + seed).shuffle(values)
        return {label: values[index] for index, label in enumerate(OPTION_LABELS)}
