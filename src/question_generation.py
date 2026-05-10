from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer

WH_WORDS = ("who", "what", "where", "when", "why", "how", "which")
_PERSON_HINT_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|President|King|Queen|Lord|Lady)\b")
_PROPER_NOUN_RE = re.compile(r"^[A-Z][\w'.\-]*(?:\s+[\w'.\-]+)*$")
_NUMERIC_RE = re.compile(r"^\d+(?:[.,]\d+)?$")
_YEAR_RE = re.compile(r"^(?:1\d{3}|20\d{2})$")
_PLACE_HINTS = {
    "city",
    "country",
    "town",
    "village",
    "school",
    "university",
    "river",
    "ocean",
    "mountain",
    "park",
    "street",
    "road",
    "building",
}
_TIME_HINTS = {"year", "month", "week", "day", "century", "decade"}
_NUMBER_HINTS = {"percent", "%", "people", "students", "dollars", "hours", "minutes", "kilometres", "miles", "km"}


def _classify_answer(answer: str, sentence_lower: str) -> str:
    answer = (answer or "").strip()
    lower = answer.lower()
    if not answer:
        return "what"
    if _NUMERIC_RE.match(answer):
        if _YEAR_RE.match(answer):
            return "when"
        return "how"
    if _PROPER_NOUN_RE.match(answer):
        if _PERSON_HINT_RE.search(answer) or any(h in sentence_lower for h in ("said", "thinks", "thought", "asked")):
            return "who"
        if any(h in sentence_lower for h in _PLACE_HINTS):
            return "where"
        return "who"
    if any(token in lower for token in _TIME_HINTS):
        return "when"
    if any(token in lower for token in _NUMBER_HINTS):
        return "how"
    if any(token in lower for token in _PLACE_HINTS):
        return "where"
    return "what"


def _strip_terminal_punct(text: str) -> str:
    return re.sub(r"[.!?\s]+$", "", text or "").strip()


_LEADING_VERB_RE = re.compile(
    r"^(?:was|were|is|are|am|has|have|had|did|does|do|will|would|can|could|should|might|may|must|"
    r"became|becomes|become|gets|got|gives|gave|takes|took|makes|made|said|says|tells|told|"
    r"founded|created|invented|wrote|written|built|discovered)\b",
    re.IGNORECASE,
)


def _trim_clause(text: str) -> str:
    """Drop trailing subordinate clauses introduced by ', which', ', that', etc."""
    text = text.strip()
    text = re.sub(r"\s*,\s*(which|that|where|when|who|whom|whose)\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*[;:]\s.*$", "", text)
    return _strip_terminal_punct(text)


_SUBJECT_BOUNDARY_RE = re.compile(
    r"\b(?:is|are|am|was|were|be|been|being|"
    r"has|have|had|"
    r"does|did|do|will|would|can|could|should|might|may|must|"
    r"became|becomes|become|gets|got|gives|gave|takes|took|makes|made|"
    r"said|says|tells|told|founded|created|invented|wrote|written|"
    r"built|discovered|considered|appears|seems|"
    r"includes|contains|stands|stood|works|worked|happens|happened)\b",
    re.IGNORECASE,
)

_DISCOURSE_LEADERS: tuple[str, ...] = (
    "for example",
    "for instance",
    "in addition",
    "in addition to that",
    "in particular",
    "in fact",
    "in conclusion",
    "in summary",
    "in short",
    "in brief",
    "in contrast",
    "by contrast",
    "on the other hand",
    "on the contrary",
    "as a result",
    "as such",
    "above all",
    "first of all",
    "to begin with",
    "to start",
    "to summarize",
    "to conclude",
    "of course",
    "however",
    "therefore",
    "moreover",
    "furthermore",
    "consequently",
    "meanwhile",
    "instead",
    "indeed",
    "still",
    "also",
    "yet",
    "but",
    "and",
    "or",
    "nor",
    "so",
    "thus",
    "hence",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "next",
    "then",
    "finally",
    "lastly",
    "again",
    "now",
    "today",
    "yesterday",
    "tomorrow",
)

_GENERIC_SUBJECTS: frozenset[str] = frozenset(
    {
        "he",
        "she",
        "it",
        "we",
        "they",
        "you",
        "i",
        "him",
        "her",
        "them",
        "us",
        "this",
        "that",
        "these",
        "those",
        "there",
        "here",
        "one",
        "ones",
        "someone",
        "anyone",
        "everyone",
        "something",
        "anything",
        "everything",
        "people",
        "many",
        "some",
        "few",
        "all",
        "most",
        "everybody",
        "anybody",
        "nobody",
    }
)

_TOPIC_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "with",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "from",
        "into",
        "than",
        "then",
        "that",
        "this",
        "these",
        "those",
    }
)


def _strip_discourse_prefix(text: str) -> str:
    cleaned = text.strip()
    cleaned_lower = cleaned.lower()
    for marker in sorted(_DISCOURSE_LEADERS, key=len, reverse=True):
        if cleaned_lower == marker:
            return ""
        if cleaned_lower.startswith(marker + " ") or cleaned_lower.startswith(marker + ","):
            cleaned = cleaned[len(marker) :].lstrip(", ").strip()
            return _strip_discourse_prefix(cleaned)
    return cleaned


def _topic_phrase(sentence: str, answer: str, *, min_len: int = 4, max_len: int = 60) -> str | None:
    if not sentence:
        return None
    leading_ws = re.match(r"^\W+", sentence)
    start = leading_ws.end() if leading_ws else 0
    match = _SUBJECT_BOUNDARY_RE.search(sentence, pos=start)
    if not match or match.start() <= start:
        return None
    subject = sentence[start : match.start()].strip(" ,.;:—–-\"'`()[]")
    subject = _strip_discourse_prefix(subject).strip(" ,.;:—–-\"'`()[]")
    if not subject:
        return None
    if not (min_len <= len(subject) <= max_len):
        return None

    subject_lower = subject.lower()
    if subject_lower in _DISCOURSE_LEADERS or subject_lower in _GENERIC_SUBJECTS:
        return None

    tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", subject)
    if not tokens:
        return None
    content_tokens = [
        token
        for token in tokens
        if token.lower() not in _TOPIC_FUNCTION_WORDS
        and token.lower() not in _GENERIC_SUBJECTS
        and token.lower() not in _DISCOURSE_LEADERS
    ]
    has_proper_noun = any(token[0].isupper() and token.lower() not in _DISCOURSE_LEADERS for token in tokens[1:])
    has_content_word = any(len(token) >= 4 for token in content_tokens)
    if not (has_proper_noun or has_content_word):
        return None

    answer_lower = (answer or "").strip().lower()
    if answer_lower:
        answer_tokens = {tok for tok in re.findall(r"[A-Za-z][A-Za-z'-]+", answer_lower)}
        subject_tokens = {tok.lower() for tok in tokens}
        if (
            answer_lower in subject_lower
            or subject_lower in answer_lower
            or (answer_tokens & subject_tokens) - _TOPIC_FUNCTION_WORDS
        ):
            return None

    return subject


_TEMPLATE_CYCLE: tuple[str, ...] = (
    "cloze_statement",  # Prioritize cloze for clarity
    "wh_subject",       # Then Wh-questions
    "cloze_trailing",   # Then trailing cloze
    "passage_according", # Then passage-according
    "detail_context",   # Deprioritize generic "which" templates
    "fact_about_topic",
)


def select_question_candidate(
    candidates: list[dict[str, str]],
    article: str,
    style_index: int,
    vectorizer: CountVectorizer | None,
    classifier,
) -> dict[str, str]:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    by_template: dict[str, list[dict[str, str]]] = {}
    for candidate in candidates:
        by_template.setdefault(candidate["template"], []).append(candidate)

    present = [name for name in _TEMPLATE_CYCLE if by_template.get(name)]
    if not present:
        return candidates[0]

    chosen_template = present[style_index % len(present)]

    def pick_from_group(group: list[dict[str, str]]) -> dict[str, str]:
        if vectorizer is not None and classifier is not None and len(group) > 1:
            scores = score_questions(
                vectorizer,
                classifier,
                [item["question"] for item in group],
                [article] * len(group),
            )
            return group[int(np.argmax(scores))]
        return group[0]

    return pick_from_group(by_template[chosen_template])


_COPULA_RE = re.compile(
    r"\b(?:is|are|was|were|means|represents|refers\s+to|describes|involves|"
    r"includes|signifies|indicates|denotes|implies)\b",
    re.IGNORECASE,
)

_NAMED_TERM_RE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:often\s+)?(?:also\s+)?"
    r"(?:known\s+as|called|named|referred\s+to\s+as|defined\s+as|termed)\b",
    re.IGNORECASE,
)


_DEMONSTRATIVE_LEADS: frozenset[str] = frozenset({"this", "that", "these", "those", "such"})


def _clean_subject_for_definition(subject_raw: str) -> str | None:
    subject = _strip_discourse_prefix(subject_raw or "").strip(" ,.;:\"'`()[]")
    if not subject:
        return None
    
    if re.match(r"^(?:in|on|by|during|before|after|since|when)\s+\d{3,4}|^[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}", subject):
        return None
    
    parts = re.split(
        r"\s+(?:in|on|at|of|to|for|with|from|by|about|that|which|who|whose)\s+",
        subject,
        maxsplit=1,
    )
    subject = parts[0].strip(" ,.;:\"'`()[]")
    if not subject:
        return None
    words = subject.split()
    if not words or not (1 <= len(words) <= 10):
        return None
    first = words[0].lower()
    if first in _DEMONSTRATIVE_LEADS:
        return None
    if subject.lower() in _GENERIC_SUBJECTS or subject.lower() in _DISCOURSE_LEADERS:
        return None
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", subject)
    if not tokens:
        return None
    has_content = any(
        len(token) >= 4 and token.lower() not in _TOPIC_FUNCTION_WORDS for token in tokens
    )
    has_proper_noun = any(token[0].isupper() for token in tokens)
    if not (has_content or has_proper_noun):
        return None
    return subject


def make_definition_question(sentence: str) -> tuple[str, str, str] | None:
    if not sentence:
        return None
    sentence_clean = sentence.strip()
    if len(sentence_clean) < 30:
        return None
    base = sentence_clean.rstrip(".!?")

    named = _NAMED_TERM_RE.search(base)
    if named:
        description_raw = base[: named.start()]
        term_raw = base[named.end():]
        term = term_raw.strip(" ,.;:\"'`()[]")
        if term and 1 <= len(term.split()) <= 6:
            description = description_raw.strip(" ,.;:\"'`()[]")
            description = _strip_discourse_prefix(description).strip(" ,.;:\"'`()[]")
            if description and 4 <= len(description.split()) <= 24:
                term_tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", term)
                if term_tokens and any(
                    token[0].isupper() or len(token) >= 4 for token in term_tokens
                ):
                    stem = f'According to the passage, what is "{term}"?'
                    return stem, description, sentence_clean

    match = _COPULA_RE.search(base)
    if not match:
        return None
    subject_raw = base[: match.start()]
    predicate_raw = base[match.end():]
    predicate = predicate_raw.strip(" ,.;:\"'`()[]")
    if not predicate or len(predicate.split()) < 4 or len(predicate.split()) > 16:
        return None
    
    predicate_lower = predicate.lower()
    
    complex_patterns = [
        r'\b(?:were|was|had)\s+(?:expelled|established|founded|overthrown)',
        r'\b(?:by|from|through)\s+(?:American|Taliban|coalition)',
        r',\s*(?:thus|therefore|consequently)',
        r'\b(?:and|or)\s+(?:which|that|thus|therefore)',
    ]
    
    for pattern in complex_patterns:
        if re.search(pattern, predicate_lower):
            return None
    
    subject = _clean_subject_for_definition(subject_raw)
    if not subject:
        return None
    verb = re.sub(r"\s+", " ", match.group(0).lower()).strip()
    if verb in {"means", "represents", "refers to", "signifies", "denotes", "implies"}:
        stem = f'According to the passage, what does "{subject}" {verb}?'
    elif verb in {"describes", "involves", "includes", "indicates"}:
        stem = f'According to the passage, what does "{subject}" {verb}?'
    else:
        stem = f'According to the passage, what is "{subject}"?'
    return stem, predicate, sentence_clean


def make_wh_question_candidates(sentence: str, answer: str) -> list[dict[str, str]]:
    sentence = (sentence or "").strip()
    answer = (answer or "").strip()
    if not sentence or not answer:
        return []
    sentence_lower = sentence.lower()
    if answer.lower() not in sentence_lower:
        return []
    pattern = re.compile(re.escape(answer), flags=re.IGNORECASE)
    redacted_full = pattern.sub("____", sentence, count=1).strip()

    primary_wh = _classify_answer(answer, sentence_lower)
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(text: str, wh: str, template: str, terminator: str) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return
        text = re.sub(r"[.!?]+$", "", text).rstrip()
        if not text:
            return
        text = text + terminator
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        candidates.append({"question": text, "wh": wh, "template": template})

    def _has_enough_remainder(redacted_text: str, *, min_content_words: int = 8) -> bool:
        scrubbed = re.sub(r"_+", " ", redacted_text)
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z'-]+", scrubbed) if len(w) >= 3]
        return len(words) >= min_content_words

    answer_index = sentence_lower.find(answer.lower())
    leading = answer_index == 0 or sentence[:answer_index].strip(" \t,") == ""

    if leading and primary_wh in {"who", "what", "where", "when", "why", "how"}:
        remainder = sentence[answer_index + len(answer):].lstrip(" ,;:-")
        remainder = _trim_clause(remainder)
        if remainder and _LEADING_VERB_RE.match(remainder):
            remainder_clean = _strip_terminal_punct(remainder)
            wh_text = f"{primary_wh.capitalize()} {remainder_clean}"
            add(wh_text, primary_wh, "wh_subject", "?")
            add(f"According to the passage, {primary_wh} {remainder_clean}", primary_wh, "passage_according", "?")

    topic = _topic_phrase(sentence, answer)
    if topic:
        pass

    if _has_enough_remainder(redacted_full):
        add(redacted_full, primary_wh, "cloze_statement", ".")

    if sentence_lower.rstrip(".!? ").endswith(answer.lower()):
        trailing = re.sub(re.escape(answer) + r"\s*[.!?]?\s*$", "____", sentence, count=1, flags=re.IGNORECASE)
        trailing = _strip_terminal_punct(trailing)
        if trailing and _has_enough_remainder(trailing):
            add(trailing, primary_wh, "cloze_trailing", ".")

    return candidates


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']+")


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]


def _question_dense_features(questions: list[str], articles: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for question, article in zip(questions, articles):
        tokens = _tokenize(question)
        article_tokens = set(_tokenize(article))
        n = max(len(tokens), 1)
        overlap = sum(1 for token in tokens if token in article_tokens) / n
        wh_present = float(any(question.lower().lstrip().startswith(w) for w in WH_WORDS))
        ends_question = float(question.strip().endswith("?"))
        length = float(len(tokens))
        avg_token_len = float(np.mean([len(token) for token in tokens])) if tokens else 0.0
        unique_ratio = len(set(tokens)) / n
        rows.append(
            [
                length,
                avg_token_len,
                wh_present,
                ends_question,
                overlap,
                unique_ratio,
            ]
        )
    return np.asarray(rows, dtype=float)


def _build_synthetic_pool(article: str, real_question: str) -> list[str]:
    from src.text_utils import extract_answer_candidates, split_sentences

    sentences = split_sentences(article)[:8]
    pool: list[str] = []
    for sentence in sentences:
        candidates = extract_answer_candidates(sentence, max_candidates=3)
        if not candidates:
            continue
        for answer in candidates[:2]:
            for templated in make_wh_question_candidates(sentence, answer):
                pool.append(templated["question"])
    if real_question and not pool:
        pool.append(_strip_terminal_punct(real_question) + " ?")
    return pool[:6]


def build_question_ranker_dataset(
    train_questions: pd.DataFrame, max_articles: int = 3000, seed: int = 42
) -> tuple[list[str], list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    if len(train_questions) > max_articles:
        idx = rng.choice(len(train_questions), size=max_articles, replace=False)
        sample = train_questions.iloc[idx]
    else:
        sample = train_questions
    questions: list[str] = []
    articles: list[str] = []
    labels: list[int] = []
    for row in sample.itertuples(index=False):
        article = str(getattr(row, "article", ""))
        real = str(getattr(row, "question", ""))
        if real:
            questions.append(real)
            articles.append(article)
            labels.append(1)
        for synthetic in _build_synthetic_pool(article, real):
            questions.append(synthetic)
            articles.append(article)
            labels.append(0)
    return questions, articles, np.asarray(labels, dtype=int)


def fit_question_ranker(
    train_questions: pd.DataFrame, max_articles: int = 3000, seed: int = 42
) -> tuple[CountVectorizer, "object", dict[str, object]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    questions, articles, labels = build_question_ranker_dataset(train_questions, max_articles=max_articles, seed=seed)
    if len(questions) < 100 or len(set(labels.tolist())) < 2:
        raise RuntimeError("Not enough samples to train the question ranker.")
    vectorizer = CountVectorizer(stop_words="english", max_features=15000, ngram_range=(1, 2))
    text_x = vectorizer.fit_transform(questions)
    dense_x = csr_matrix(_question_dense_features(questions, articles))
    features = hstack([text_x, dense_x], format="csr")
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    split = int(len(indices) * 0.85)
    train_idx, val_idx = indices[:split], indices[split:]
    classifier = LogisticRegression(max_iter=600, solver="liblinear", random_state=seed)
    classifier.fit(features[train_idx], labels[train_idx])
    pred = classifier.predict(features[val_idx])
    metrics = {
        "n_samples": int(len(labels)),
        "positive_rate": float(np.mean(labels)),
        "validation_accuracy": float(accuracy_score(labels[val_idx], pred)),
        "validation_macro_f1": float(f1_score(labels[val_idx], pred, average="macro", zero_division=0)),
    }
    return vectorizer, classifier, metrics


def score_questions(
    vectorizer: CountVectorizer,
    classifier,
    questions: list[str],
    articles: list[str],
) -> np.ndarray:
    if not questions:
        return np.zeros(0, dtype=float)
    text_x = vectorizer.transform(questions)
    dense_x = csr_matrix(_question_dense_features(questions, articles))
    features = hstack([text_x, dense_x], format="csr")
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(features)[:, 1]
    if hasattr(classifier, "decision_function"):
        return classifier.decision_function(features)
    return classifier.predict(features).astype(float)


def keyword_overlap_score(question: str, article: str) -> float:
    tokens = _tokenize(question)
    article_tokens = Counter(_tokenize(article))
    if not tokens:
        return 0.0
    overlap = sum(1 for token in tokens if article_tokens[token] > 0)
    return overlap / max(len(tokens), 1)
