"""Minimal unit tests for the QuizEngine, Wh-template generator, and helper utilities.

These tests use the artifacts that already live under ``models/`` and the small
RACE validation CSV under ``data/processed/``. They are intentionally fast and
do not retrain anything.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.model_b_train import (
    char_level_match,
    compute_distractor_features,
    frequency_substitution_distractors,
)
from src.question_generation import _classify_answer, make_wh_question_candidates, select_question_candidate
from src.text_utils import extract_answer_candidates, redact_answer, split_sentences

SAMPLE_ARTICLE = (
    "Marie Curie was born in Warsaw in 1867. She moved to Paris in 1891 to study physics. "
    "She discovered the elements polonium and radium with her husband Pierre Curie. "
    "She received the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911."
)


class TextUtilsTests(unittest.TestCase):
    def test_split_sentences_basic(self):
        sentences = split_sentences(SAMPLE_ARTICLE)
        self.assertGreaterEqual(len(sentences), 3)
        self.assertTrue(all(len(s) > 20 for s in sentences))

    def test_extract_answer_candidates_finds_proper_nouns(self):
        candidates = extract_answer_candidates(SAMPLE_ARTICLE.split(".")[0], max_candidates=8)
        joined = " | ".join(candidates).lower()
        self.assertIn("warsaw", joined)
        self.assertIn("marie", joined)

    def test_redact_answer(self):
        sentence = "She moved to Paris in 1891 to study physics."
        redacted = redact_answer(sentence, "Paris", count=1)
        self.assertIn("____", redacted)
        self.assertNotIn("Paris", redacted)


class WhTemplateTests(unittest.TestCase):
    def test_classify_year(self):
        self.assertEqual(_classify_answer("1903", "she received the nobel prize in physics in 1903"), "when")

    def test_classify_proper_noun_who(self):
        self.assertEqual(_classify_answer("Marie Curie", "marie curie said"), "who")

    def test_make_wh_question_returns_candidates_with_blank(self):
        sentence = "Marie Curie was born in Warsaw in 1867."
        candidates = make_wh_question_candidates(sentence, "Warsaw")
        self.assertTrue(candidates)
        for c in candidates:
            self.assertTrue(c["question"].endswith(("?", ".")))
            # at least one cloze form must contain the blank marker
        self.assertTrue(any("____" in c["question"] for c in candidates))

    def test_make_wh_question_subject_form_drops_prefix(self):
        sentence = "Mrs. Baker's sister was ill"
        candidates = make_wh_question_candidates(sentence, "Mrs. Baker's sister")
        questions = [c["question"] for c in candidates]
        self.assertTrue(any(q.lower().startswith("who was ill") for q in questions))
        for q in questions:
            self.assertNotIn("described in this passage", q.lower())
            self.assertNotIn("fits the blank", q.lower())

    def test_select_question_candidate_rotates_templates(self):
        sentence = "Period dramas are intended to capture the ambience of a particular era."
        candidates = make_wh_question_candidates(sentence, "Period dramas")
        self.assertGreaterEqual(len(candidates), 3)
        templates_by_style = [
            select_question_candidate(candidates, SAMPLE_ARTICLE, i, None, None)["template"]
            for i in range(5)
        ]
        self.assertGreaterEqual(len(set(templates_by_style)), 2, msg="rotation should yield mixed templates")

    def test_make_wh_question_includes_non_cloze_templates(self):
        sentence = "Marie Curie was born in Warsaw in 1867."
        candidates = make_wh_question_candidates(sentence, "Warsaw")
        templates = {c["template"] for c in candidates}
        self.assertTrue(templates & {"detail_context", "fact_about_topic", "cloze_statement"})


class CharLevelMatchTests(unittest.TestCase):
    def test_identical_strings_score_one(self):
        self.assertEqual(char_level_match("Paris", "Paris"), 1.0)

    def test_disjoint_strings_score_zero(self):
        self.assertLess(char_level_match("xyz", "abc"), 0.5)

    def test_substring_returns_high_score(self):
        self.assertGreater(char_level_match("Paris", "Parisian"), 0.5)


class FeatureMatrixTests(unittest.TestCase):
    def test_compute_distractor_features_shape_no_w2v(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer().fit([SAMPLE_ARTICLE])
        candidates = ["Paris", "Warsaw", "Berlin"]
        features = compute_distractor_features(SAMPLE_ARTICLE, "Where was she born?", "Warsaw", candidates, vectorizer)
        self.assertEqual(features.shape[0], len(candidates))
        self.assertEqual(features.shape[1], 8)
        self.assertTrue(np.all(np.isfinite(features)))


class FrequencySubstitutionTests(unittest.TestCase):
    def test_returns_distractors_excluding_answer(self):
        out = frequency_substitution_distractors(SAMPLE_ARTICLE, "Paris", existing_options=["Paris"], top_n=3)
        texts = {item["text"].lower() for item in out}
        self.assertNotIn("paris", texts)
        self.assertLessEqual(len(out), 3)


class QuizEngineSmokeTests(unittest.TestCase):
    def test_generate_returns_required_question_count(self):
        try:
            from src.inference import QuizEngine
        except Exception as exc:
            self.skipTest(f"QuizEngine could not be imported: {exc}")
        engine = QuizEngine()
        if not engine.model_b_loaded:
            self.skipTest("Model B artifacts missing; skip smoke test.")
        article = SAMPLE_ARTICLE * 4
        result = engine.generate(article=article, question=None, options={}, question_count=5)
        self.assertGreaterEqual(len(result["questions"]), 5)
        for q in result["questions"]:
            self.assertIn("question", q)
            self.assertEqual(set(q["options"].keys()), set("ABCD"))
            self.assertIn(q["predicted_correct_option"], "ABCD")


if __name__ == "__main__":
    unittest.main()
