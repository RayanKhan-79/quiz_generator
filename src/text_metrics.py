"""Compute ROUGE, BLEU, and METEOR scores for generated text."""
from __future__ import annotations

from typing import Any

import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


def compute_rouge(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    
    Args:
        references: Ground truth texts
        hypotheses: Generated texts
        
    Returns:
        Dictionary with average ROUGE scores
    """
    if not rouge_scorer:
        return {"error": "rouge_score not available"}
    
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        scores = scorer.score(ref, hyp)
        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)
    
    return {
        "rouge1_f": sum(rouge1_scores) / len(rouge1_scores),
        "rouge2_f": sum(rouge2_scores) / len(rouge2_scores),
        "rougeL_f": sum(rougeL_scores) / len(rougeL_scores),
    }


def compute_bleu(references: list[list[str]], hypotheses: list[str], max_n: int = 4) -> dict[str, float]:
    """
    Compute BLEU scores (BLEU-1 through BLEU-4).
    
    Args:
        references: List of reference texts (each can have multiple valid references)
        hypotheses: Generated texts
        max_n: Maximum n-gram to compute (1-4)
        
    Returns:
        Dictionary with average BLEU scores
    """
    bleu_scores = {f"bleu{i}": [] for i in range(1, max_n + 1)}
    
    for ref_list, hyp in zip(references, hypotheses):
        # Tokenize
        ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in (ref_list if isinstance(ref_list, list) else [ref_list])]
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        
        # Compute BLEU for different n-grams
        for n in range(1, max_n + 1):
            weights = tuple([1.0 / n] * n)
            score = sentence_bleu(ref_tokens, hyp_tokens, weights=weights)
            bleu_scores[f"bleu{n}"].append(score)
    
    return {k: sum(v) / len(v) for k, v in bleu_scores.items()}


def compute_meteor(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """
    Compute METEOR scores.
    
    Args:
        references: Ground truth texts
        hypotheses: Generated texts
        
    Returns:
        Dictionary with average METEOR score
    """
    meteor_scores = []
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = nltk.word_tokenize(ref.lower())
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        score = meteor_score([ref_tokens], hyp_tokens)
        meteor_scores.append(score)
    
    return {
        "meteor": sum(meteor_scores) / len(meteor_scores),
    }


def compute_all_text_metrics(
    references: list[str],
    hypotheses: list[str],
) -> dict[str, Any]:
    """
    Compute all text generation metrics.
    
    Args:
        references: Ground truth texts
        hypotheses: Generated texts
        
    Returns:
        Dictionary with all metric scores
    """
    metrics = {}
    
    # ROUGE
    metrics.update(compute_rouge(references, hypotheses))
    
    # BLEU
    metrics.update(compute_bleu(references, hypotheses))
    
    # METEOR
    metrics.update(compute_meteor(references, hypotheses))
    
    return metrics
