from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


_ABBREVIATIONS = (
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Mt", "Ft",
    "vs", "etc", "e.g", "i.e", "Inc", "Ltd", "Co", "No",
)


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    protected = text.strip()
    for abbr in _ABBREVIATIONS:
        protected = re.sub(rf"\b{re.escape(abbr)}\.", f"{abbr}<DOT>", protected)
    sentences = re.split(r"(?<=[.!?])\s+", protected)
    restored = [s.replace("<DOT>", ".").strip() for s in sentences]
    return [sentence for sentence in restored if len(sentence) > 20]


def tokenize_terms(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text or "") if token.lower() not in ENGLISH_STOP_WORDS]


_DETERMINER_RE = re.compile(
    r"\b(?:a|an|the|its|their|his|her|our|my|your|this|that|these|those|"
    r"some|every|each|all|most|few|several|many|another|no)\b",
    re.IGNORECASE,
)

# Strong boundaries that end a noun-phrase span.  We deliberately keep
# "and"/"or" *inside* the phrase when they connect noun-level constituents
# (so "the bread and butter" stays whole) but treat them as clause
# boundaries when preceded by a comma ("a market before a competitor, or
# unintentional, like code...").
_PHRASE_BOUNDARY_RE = re.compile(
    r"[.;]|—|–|\(|\)|"
    r",\s+(?:and|or|but|so)\s+|"
    r"\s+(?:because|while|whereas|since|although|though|yet|but)\b",
    re.IGNORECASE,
)


def extract_long_phrases(
    text: str,
    *,
    min_words: int = 3,
    max_words: int = 18,
    max_phrases: int = 80,
) -> list[str]:
    """Extract determiner-headed noun-phrase-like spans of *min_words* to
    *max_words* words.  These read more like the multi-word distractor
    options that appear in real reading-comprehension MCQs (e.g.
    "the extra time and effort required to maintain or fix the code")."""
    if not text:
        return []
    sentences = split_sentences(text) or [text]
    candidates: list[str] = []
    for sentence in sentences:
        for match in _DETERMINER_RE.finditer(sentence):
            start = match.start()
            tail = sentence[start:]
            cut = _PHRASE_BOUNDARY_RE.search(tail)
            phrase = (tail[: cut.start()] if cut else tail).strip(" ,.;:!?\"'()")
            if not phrase:
                continue
            wc = len(phrase.split())
            if min_words <= wc <= max_words:
                candidates.append(phrase)
    seen: set[str] = set()
    out: list[str] = []
    for phrase in candidates:
        key = re.sub(r"\s+", " ", phrase.lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(phrase)
    return out[:max_phrases]


def extract_candidate_phrases(text: str, max_candidates: int = 80) -> list[str]:
    long_phrases = extract_long_phrases(text, max_phrases=max_candidates // 2)
    tokens = tokenize_terms(text)
    counts = Counter(tokens)
    short_unigrams = [word for word, _ in counts.most_common(max_candidates)]
    bigrams = [f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a != b]
    short_bigrams = [phrase for phrase, _ in Counter(bigrams).most_common(max_candidates // 2)]
    candidates = long_phrases + short_unigrams + short_bigrams
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        key = candidate.lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique[:max_candidates]


def extract_answer_candidates(sentence: str, max_candidates: int = 10) -> list[str]:
    text = sentence or ""
    candidates: list[str] = []
    candidates.extend(match.group(0).strip() for match in re.finditer(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text))
    candidates.extend(match.group(0).strip() for match in re.finditer(r"\b\d+(?:[.,]\d+)?(?:\s?(?:percent|years?|days?|miles?|km|dollars?))?\b", text, flags=re.IGNORECASE))
    tokens = [token for token in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text) if token.lower() not in ENGLISH_STOP_WORDS]
    for size in (3, 2, 1):
        for index in range(0, max(len(tokens) - size + 1, 0)):
            phrase = " ".join(tokens[index : index + size])
            if phrase:
                candidates.append(phrase)
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        cleaned = candidate.strip(" ,.;:!?()[]\"'")
        key = cleaned.lower()
        if len(cleaned) < 4 or key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
        if len(unique) >= max_candidates:
            break
    return unique


def redact_answer(sentence: str, answer: str, count: int = 0) -> str:
    if not answer:
        return sentence
    pattern = re.compile(re.escape(answer), flags=re.IGNORECASE)
    return pattern.sub("____", sentence, count=count)
