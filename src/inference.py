from __future__ import annotations

import csv
import io
import json
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.model_a_train import MODEL_DIR as MODEL_A_DIR
from src.model_a_train import build_feature_blocks, rank_generation_sentence_answer_pairs
from src.model_b_train import MODEL_DIR as MODEL_B_DIR
from src.model_b_train import rank_distractors
from src.question_generation import (
    make_definition_question,
    make_wh_question_candidates,
    select_question_candidate,
)
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
            self.question_ranker_vec = None
            self.question_ranker_clf = None
            ranker_vec_path = self.model_a_dir / "question_ranker_vectorizer.joblib"
            ranker_clf_path = self.model_a_dir / "question_ranker.joblib"
            if ranker_vec_path.exists() and ranker_clf_path.exists():
                try:
                    self.question_ranker_vec = joblib.load(ranker_vec_path)
                    self.question_ranker_clf = joblib.load(ranker_clf_path)
                except Exception:
                    self.question_ranker_vec = None
                    self.question_ranker_clf = None
            self.model_a_loaded = True
        except Exception as exc:  # artifacts are optional during first setup
            self.model_a_error = str(exc)
        try:
            self.vectorizer_b = joblib.load(self.model_b_dir / "tfidf_vectorizer.joblib")
            self.model_b_config: dict[str, Any] = {}
            cfg_path = self.model_b_dir / "config.json"
            if cfg_path.exists():
                with cfg_path.open("r", encoding="utf-8") as handle:
                    self.model_b_config = json.load(handle)
            self.w2v_b = None
            w2v_meta = self.model_b_config.get("word2vec") or {}
            w2v_file = w2v_meta.get("file", "word2vec.kv")
            w2v_path = self.model_b_dir / str(w2v_file)
            if w2v_path.exists():
                try:
                    from gensim.models import KeyedVectors

                    self.w2v_b = KeyedVectors.load(str(w2v_path), mmap="r")
                except Exception:
                    self.w2v_b = None
            self.distractor_ranker = None
            ranker_path = self.model_b_dir / "distractor_ranker.joblib"
            if ranker_path.exists():
                try:
                    self.distractor_ranker = joblib.load(ranker_path)
                except Exception:
                    self.distractor_ranker = None
            self.hint_scorer = None
            scorer_path = self.model_b_dir / "hint_scorer.joblib"
            if scorer_path.exists():
                try:
                    self.hint_scorer = joblib.load(scorer_path)
                except Exception:
                    self.hint_scorer = None
            self.model_b_loaded = True
        except Exception as exc:
            self.model_b_error = str(exc)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_a_loaded": self.model_a_loaded,
            "model_b_loaded": self.model_b_loaded,
            "model_b_word2vec_loaded": bool(getattr(self, "w2v_b", None)),
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
        evidence = self._passage_evidence_scores(article, question, options)


        prob_norm = probabilities / probabilities.sum() if probabilities.sum() > 0 else probabilities
        combined = 0.65 * prob_norm + 0.35 * evidence
        best_index = int(np.argmax(combined))
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
                else f"TF-IDF verifier ranked option {predicted} highest using Model A probabilities blended with passage co-occurrence evidence."
            ),
            "latency_ms": latency_ms,
        }
        self.store.add({"endpoint": "verify", "latency_ms": latency_ms, "predicted_option": predicted, "selected_option": selected})
        return result

    def _passage_evidence_scores(self, article: str, question: str, options: dict[str, str]) -> np.ndarray:
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "of", "to", "in", "on", "at", "by", "for", "with", "as", "and", "or",
            "but", "not", "so", "that", "this", "these", "those", "it", "its",
            "he", "she", "they", "we", "you", "i", "his", "her", "their", "our",
            "my", "your", "what", "who", "whom", "when", "where", "why", "how",
            "which", "do", "does", "did", "from", "into", "than", "then", "also",
        }

        def tokenize(text: str) -> list[str]:
            tokens = []
            for raw in text.split():
                cleaned = raw.strip(".,;:!?\"'()[]_").lower()
                if cleaned and cleaned not in stopwords and len(cleaned) > 1:
                    tokens.append(cleaned)
            return tokens

        sentences = split_sentences(article) or [article]
        sentence_tokens = [set(tokenize(sent)) for sent in sentences]
        question_tokens = set(tokenize(question))
        scores = np.zeros(len(OPTION_LABELS), dtype=float)

        for i, label in enumerate(OPTION_LABELS):
            option_text = options.get(label, "")
            option_tokens = set(tokenize(option_text))
            if not option_tokens:
                continue
            best = 0.0
            for sent_set in sentence_tokens:
                if not sent_set:
                    continue
                q_overlap = len(question_tokens & sent_set)
                o_overlap = len(option_tokens & sent_set)
                if o_overlap == 0:
                    continue
                pair_score = (q_overlap + 1) * o_overlap / max(len(sent_set), 1)
                if pair_score > best:
                    best = pair_score
            scores[i] = best

        if scores.max() > 0:
            scores = scores / scores.max()
        else:
            scores = np.full(len(OPTION_LABELS), 0.25)
        return scores

    def generate(self, article: str, question: str | None = None, options: dict[str, str] | None = None, question_count: int = 5) -> dict[str, Any]:
        started = time.perf_counter()
        if not self.model_b_loaded:
            raise RuntimeError("Model B artifacts are missing. Run: python -m src.model_b_train")
        options = options or {}
        generated_questions = self._question_specs(article, question, options, max(1, int(question_count)))
        cfg = getattr(self, "model_b_config", {}) or {}
        w2v_section = cfg.get("word2vec") or {}
        items = []
        for index, spec in enumerate(generated_questions):
            generated_question = spec["question"]
            answer = spec["answer"]
            use_user_options = bool(spec.get("use_provided_options"))
            merge_options = options if use_user_options else {}
            existing_for_rank = list(options.values()) if use_user_options else []
            distractors = rank_distractors(
                article,
                generated_question,
                answer,
                self.vectorizer_b,
                existing_for_rank,
                w2v=getattr(self, "w2v_b", None),
                distractor_w2v_blend=float(w2v_section.get("distractor_blend_weight", 0.38)),
                ranker=getattr(self, "distractor_ranker", None),
                ranker_weight=float(cfg.get("distractor_ranker_weight", 0.6)),
            )
            final_options = self._merge_options(answer, distractors, merge_options, seed=index)
            confidence = 0.0
            if use_user_options and self.model_a_loaded:
                # User supplied the question and all four options: we don't know the
                # ground-truth answer, so let Model A (blended with passage evidence)
                # pick the most likely correct option.
                verification = self.verify(article, generated_question, final_options, "A")
                predicted = verification["predicted_option"]
                answer = final_options.get(predicted, answer)
                confidence = verification["confidence"]
            else:
                predicted = self._label_for_answer(final_options, answer)
                if self.model_a_loaded:
                    verification = self.verify(article, generated_question, final_options, predicted, correct_option=predicted)
                    confidence = verification["confidence"]
            hints = self._hints_for_spec(article, generated_question, answer, spec.get("sentence", ""))
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
        specs: list[dict[str, str]] = []
        if provided_question and all(options.get(label) for label in OPTION_LABELS):
            specs.append(
                {
                    "question": provided_question,
                    "answer": options["A"],
                    "sentence": "",
                    "use_provided_options": True, # type: ignore
                }
            )
        if len(specs) >= count:
            return specs[:count]

        cfg = getattr(self, "model_b_config", {}) or {}
        w2v_section = cfg.get("word2vec") or {}
        if self.model_a_loaded:
            gen_vec = self.vectorizer_a
            gen_w2v = None
            gen_blend = 0.0
        else:
            gen_vec = self.vectorizer_b
            gen_w2v = getattr(self, "w2v_b", None)
            gen_blend = float(w2v_section.get("generation_blend_weight", w2v_section.get("distractor_blend_weight", 0.38)))

        ranked_pairs = rank_generation_sentence_answer_pairs(
            article,
            gen_vec,
            w2v=gen_w2v,
            generation_w2v_blend=gen_blend,
        )

        heuristic_sents = self._rank_generation_sentences(article)
        if not heuristic_sents:
            heuristic_sents = [
                article[index : index + 180] for index in range(0, max(len(article), 1), 180) if article[index : index + 180].strip()
            ]

        definition_specs: list[dict[str, str]] = []
        seen_def_subjects: set[str] = set()
        for sentence in split_sentences(article):
            built = make_definition_question(sentence)
            if not built:
                continue
            stem, predicate, original = built
            subject_match = re.search(r'"([^"]+)"', stem)
            subject_key = subject_match.group(1).lower() if subject_match else stem.lower()
            if subject_key in seen_def_subjects:
                continue
            seen_def_subjects.add(subject_key)
            definition_specs.append(
                {
                    "question": stem,
                    "answer": predicate,
                    "sentence": original,
                    "wh": "what",
                }
            )

        used_sentence_keys: set[str] = set()
        used_answer_keys: set[str] = set()
        pair_index = 0
        legacy_index = 0
        def_index = 0

        while len(specs) < count:
            inject_definition = def_index < len(definition_specs) and (len(specs) % 3 == 1)
            if inject_definition:
                spec = definition_specs[def_index]
                def_index += 1
                sk = spec["sentence"].strip().lower()
                ak = spec["answer"].strip().lower()
                if sk in used_sentence_keys or ak in used_answer_keys:
                    continue
                used_sentence_keys.add(sk)
                used_answer_keys.add(ak)
                specs.append(spec)
                continue

            if pair_index < len(ranked_pairs):
                _score, sentence, answer = ranked_pairs[pair_index]
                pair_index += 1
                sk, ak = sentence.strip().lower(), answer.strip().lower()
                if sk in used_sentence_keys or ak in used_answer_keys:
                    continue
                used_sentence_keys.add(sk)
                used_answer_keys.add(ak)
                question_text, wh_label_text = self._best_wh_question(article, sentence, answer, style_index=len(specs))
                specs.append(
                    {
                        "question": question_text,
                        "answer": answer,
                        "sentence": sentence,
                        "wh": wh_label_text,
                    }
                )
                continue

            if def_index < len(definition_specs):
                spec = definition_specs[def_index]
                def_index += 1
                sk = spec["sentence"].strip().lower()
                ak = spec["answer"].strip().lower()
                if sk in used_sentence_keys or ak in used_answer_keys:
                    continue
                used_sentence_keys.add(sk)
                used_answer_keys.add(ak)
                specs.append(spec)
                continue

            sentence = heuristic_sents[legacy_index % len(heuristic_sents)]
            legacy_index += 1
            answer = self._answer_for_sentence(sentence)
            sk, ak = sentence.strip().lower(), answer.strip().lower()
            allow_dup = legacy_index > len(heuristic_sents) * 8
            if (sk in used_sentence_keys or ak in used_answer_keys) and not allow_dup:
                continue
            used_sentence_keys.add(sk)
            used_answer_keys.add(ak)
            question_text, wh_label_text = self._best_wh_question(article, sentence, answer, style_index=len(specs))
            specs.append(
                {
                    "question": question_text,
                    "answer": answer,
                    "sentence": sentence,
                    "wh": wh_label_text,
                }
            )

        return specs[:count]

    def _best_wh_question(self, article: str, sentence: str, answer: str, style_index: int = 0) -> tuple[str, str]:
        cloze_fallback = (
            f"{redact_answer(sentence, answer, count=1).rstrip('.!? ')}.",
            "which",
        )
        candidates = make_wh_question_candidates(sentence, answer)
        if not candidates:
            return cloze_fallback
        chosen = select_question_candidate(
            candidates,
            article,
            style_index,
            self.question_ranker_vec,
            self.question_ranker_clf,
        )
        return chosen["question"], chosen.get("wh", "what")

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
        from sklearn.metrics.pairwise import cosine_similarity

        sentences = [s for s in split_sentences(article) if len(s.strip()) > 20]
        if not sentences:
            sentences = [sentence] if sentence else [article[:200]]

        sentence_x = self.vectorizer_b.transform(sentences)
        question_x = self.vectorizer_b.transform([question])
        scores = cosine_similarity(sentence_x, question_x).ravel()
        ordered = [sentences[int(i)] for i in scores.argsort()[::-1]]

        # Find the sentence that actually contains the answer — not just the top scored one
        answer_lower = answer.lower()
        support = next(
            (s for s in ordered if answer_lower in s.lower()),
            ordered[0]  # fallback to top scored if none contain answer
        )

        # Hint 1: relevant sentence that does NOT contain the answer
        hint1 = next(
            (s for s in ordered if answer_lower not in s.lower()),
            ordered[1] if len(ordered) > 1 else support
        )

        # Hint 2: the answer sentence with only the first word shown
        first_word = answer.split()[0] if answer.split() else ""
        hint2 = re.sub(re.escape(answer), f"{first_word}...", support, flags=re.IGNORECASE)

        # Hint 3: fully blanked with word count
        word_count = len(answer.split())
        hint3 = re.sub(re.escape(answer), "____", support, flags=re.IGNORECASE) + f" ({word_count} words)"

        def _scrub(hint: str) -> str:
            if answer_lower in hint.lower():
                return "Sorry, no more hints! At this point I can only tell you the answer."
            return hint

        hint1 = _scrub(hint1)
        hint2 = _scrub(hint2)

        # If hint2 is same as hint1, replace it
        if hint2 == hint1 or hint2 == "Sorry, no more hints! At this point I can only tell you the answer.":
            hint2 = "Sorry, no more hints! At this point I can only tell you the answer."

        # If hint3 is same as hint2 or hint1, replace it
        if hint3 == hint2 or hint3 == hint1:
            hint3 = "Sorry, no more hints! At this point I can only tell you the answer."

        return [hint1, hint2, hint3]

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
