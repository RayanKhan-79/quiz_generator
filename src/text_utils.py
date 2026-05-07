from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]


def tokenize_terms(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text or "") if token.lower() not in ENGLISH_STOP_WORDS]


def extract_candidate_phrases(text: str, max_candidates: int = 80) -> list[str]:
    tokens = tokenize_terms(text)
    counts = Counter(tokens)
    candidates = [word for word, _ in counts.most_common(max_candidates)]
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a != b]
    candidates.extend(phrase for phrase, _ in Counter(bigrams).most_common(max_candidates // 2))
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        key = candidate.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique[:max_candidates]


def redact_answer(sentence: str, answer: str) -> str:
    if not answer:
        return sentence
    pattern = re.compile(re.escape(answer), flags=re.IGNORECASE)
    return pattern.sub("____", sentence)
