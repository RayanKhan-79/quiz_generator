# RACE Quiz Generator: Technical Deep Dive & System Architecture

This document provides an exhaustive technical analysis of the **RACE Reading Comprehension Quiz Generator**. It covers the mathematical intuition, software architecture, and the detailed logic governing question generation and model evaluation.

---

## 1. System Architecture Overview

The project is built as a modular multi-process application:

*   **Frontend**: A React application bootstrapped with **Vite**. It uses **Tailwind CSS** for styling and **Lucide React** for icons. It communicates with the backend via asynchronous `fetch` requests.
*   **Backend**: A **FastAPI** REST server. It handles request validation (via Pydantic), model orchestration, and logging.
*   **Engine**: The `QuizEngine` class (in `src/inference.py`) serves as the "brain," loading trained models and executing the generation pipeline.
*   **Data Pipeline**: A custom preprocessing engine that transforms raw RACE datasets (CSV/Parquet) into a format suitable for supervised learning.

---

## 2. Data Preprocessing & Feature Engineering

### 2.1. Schema Normalization
Raw RACE data often has inconsistent headers (e.g., "passage" vs "article"). `src/preprocessing.py` uses a fuzzy matching strategy (`_first_existing`) to normalize all inputs into a standard schema:
`id`, `article`, `question`, `A`, `B`, `C`, `D`, `answer`.

### 2.2. Option Expansion
To train Model A (the verifier), the system converts each 4-option MCQ into 4 independent rows. 
*   **Original**: 1 Question -> 4 Options (A, B, C, D)
*   **Expanded**: 4 Rows -> (Question + Option A, Label=0), (Question + Option B, Label=1), etc.
This allows the model to learn the specific relationship between a question and a single candidate answer.

---

## 3. Model A: The Discriminative Verifier

### 3.1. TF-IDF Vectorization Logic
The system uses `TfidfVectorizer` with specific parameters:
*   **N-grams (1, 2)**: Captures both individual words and two-word phrases (e.g., "White House").
*   **Sublinear TF Scaling**: Replaces term frequency `tf` with `1 + log(tf)`. This prevents very frequent words in a long article from dominating the feature space.
*   **Max Features (20,000 - 30,000)**: Limits the vocabulary to the most significant terms to prevent overfitting and save memory.

### 3.2. Similarity Blocks (The Logic)
Model A doesn't just look at words; it looks at **overlaps**. It calculates **Cosine Similarity** between four key components:
1.  **Article ↔ Question**: Does the question actually pertain to the passage?
2.  **Article ↔ Option**: Is the option mentioned in the text?
3.  **Question ↔ Option**: Does the option semantically "fit" the question?
4.  **Verification Text**: A combined string `[Question] + [Option]` used to capture the cohesive meaning.

**Cosine Similarity Formula**: 
$$\text{similarity} = \frac{A \cdot B}{\|A\| \|B\|}$$
Where $A$ and $B$ are the TF-IDF vectors. A score of 1.0 means identical keyword distribution; 0.0 means no shared keywords.

### 3.3. The Ensemble Strategy
The system uses an **Ensemble of Classifiers**:
*   **Logistic Regression**: Provides a smooth probabilistic interpretation of the features.
*   **Linear SVM (Calibrated)**: Finds the "Maximum Margin Hyperplane" to separate correct from incorrect answers. Since standard SVMs don't provide probabilities, it uses **Platt Scaling** (`CalibratedClassifierCV`) to map SVM outputs to a 0-1 probability range.

---

## 4. Model B: The Generative Ranking Engine

Model B is responsible for creating the "wrong" choices and the hints.

### 4.1. Question Generation Heuristics
The generator doesn't just pick random sentences. It uses **Sentence Ranking**:
1.  **Sentence Splitting**: Uses Regex to split the article into sentences longer than 20 characters.
2.  **Candidate Extraction**: Uses Regex patterns to find:
    *   **Named Entities**: `\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b` (Capitalized names).
    *   **Numerical Data**: Dates, percentages, or measurements.
3.  **Scoring**: Sentences are scored based on the number of candidates they contain. The top-ranked sentences become the basis for questions.

### 4.2. Distractor Ranking Algorithm
To pick the 3 "distractors" for an MCQ, Model B scores candidates from the article using a weighted formula:
$$Score = (0.50 \times \text{Sim}_{Answer}) + (0.35 \times \text{Sim}_{Question}) + (0.15 \times \text{Frequency})$$

*   **Similarity to Answer (50%)**: A good distractor should be in the same "category" as the answer (e.g., if the answer is "Monday", a good distractor is "Tuesday").
*   **Similarity to Question (35%)**: The distractor must be relevant to the question's topic.
*   **Frequency (15%)**: Familiar terms mentioned elsewhere in the article are more likely to trick a casual reader.

### 4.3. Diversity Penalty (MMR Style)
To avoid having three distractors that are all basically the same, the system applies a **Diversity Penalty**. After picking the first distractor, it penalizes other candidates that are too similar to it. This ensures the quiz covers a broad range of potential misunderstandings.

---

## 5. Advanced Evaluation Metrics

When the project is evaluated (via `src/evaluate.py` or the `/metrics` endpoint), it computes multiple dimensions of quality:

### 5.1. BLEU (Bilingual Evaluation Understudy)
*   **Logic**: Measures **N-gram Overlap Precision**. It calculates how many words/phrases in the generated question match the reference.
*   **Brevity Penalty**: BLEU penalizes very short generations to prevent the model from "cheating" by only outputting one certain word.
*   **Formula Intuition**: 
    $$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^N w_n \ln p_n\right)$$
    *Where $p_n$ is the precision of n-grams.*
*   **Implementation**: Computed via `nltk.translate.bleu_score` on tokenized text; scores BLEU-1 through BLEU-4.

### 5.2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
*   **Logic**: Measures **Recall**. It checks how much of the "human knowledge" in the reference was successfully captured by the AI.
*   **Variants**:
    - **ROUGE-1**: Unigram overlap
    - **ROUGE-2**: Bigram overlap  
    - **ROUGE-L**: Longest Common Subsequence, capturing sentence structure better than BLEU
*   **Implementation**: Uses `rouge_score` library with stemming enabled; F-scores reported for each variant.

### 5.3. METEOR (Metric for Evaluation of Translation with Explicit ORdering)
*   **Logic**: The most "human-like" metric. Unlike BLEU/ROUGE, which look for exact word matches, METEOR uses:
    *   **Stemming**: Matches "running" with "run".
    *   **Synonymy**: Matches "quick" with "fast" via WordNet.
    *   **Paraphrasing**: Understands different ways of saying the same thing.
*   **Penalty**: It applies a penalty for "chunkiness" (how much the word order has been scrambled), ensuring the generated sentence is grammatical.
*   **Implementation**: Computed via `nltk.translate.meteor_score` with automatic tokenization and lemmatization.

### 5.4. Metrics Integration
The `src/text_metrics.py` module provides a unified interface for all three metrics:
```python
from src.text_metrics import compute_all_text_metrics
metrics = compute_all_text_metrics(reference_texts, generated_texts)
```
Output includes `rouge1_f`, `rouge2_f`, `rougeL_f`, `bleu1`–`bleu4`, and `meteor`.

These metrics are computed during evaluation on the test dataset (or any selected split) and recorded in `models/evaluation_metrics.json` alongside Model A and Model B performance metrics.

### 5.5. Evaluation Results Interpretation
- **High ROUGE/BLEU/METEOR** (>0.9): Generated text closely matches references (good for templated or predictable outputs)
- **Moderate scores** (0.6–0.9): Text is semantically similar but with variations in phrasing or order
- **Low scores** (<0.3): Significant divergence from reference; may indicate model failure or high diversity

---

## 6. Potential Issues & Scalability

### 6.1. Identified Limitations
*   **Lexical Overlap Bias**: Because the system is based on TF-IDF (word counts), it can be fooled by "distractors" that use the exact same words as the article but mean something different.
*   **Context Loss**: In cloze-style generation, if a sentence starts with "He did this...", the question might be impossible to answer if the reader doesn't know who "He" is from the previous sentence.
*   **Cold Start**: The system requires a pre-trained vectorizer. If the article is about a completely new topic (e.g., Quantum Computing) not seen in the training data, the similarity scores will be less accurate.

### 6.2. Future Improvements
*   **Neural Embeddings**: Transition from TF-IDF to **Sentence-BERT** or **RoBERTa**. This would allow the model to understand that "feline" and "cat" are related, even if they share no letters.
*   **Graph-based Distractors**: Use **WordNet** or **Knowledge Graphs** to find "semantic siblings" for distractors (e.g., finding other planets if the answer is "Mars").
*   **Question Transformation**: Use a **T5 (Text-to-Text Transfer Transformer)** model to turn cloze sentences into natural Wh-questions (Who/What/Where).

---
*Document Version: 1.2*
*Last Updated: May 2026*
*Author: Antigravity AI Assistant*
