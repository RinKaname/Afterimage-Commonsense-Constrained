# Afterimage-Commonsense-Constrained - EDA Analysis

This report documents the findings from the Exploratory Data Analysis (EDA) on the `train.jsonl` dataset.

## 1. Overview & Structural Statistics
*   **Train Dialogues:** 841
*   **Test Dialogues:** 225
*   **Gaps per dialogue:** Min: 1, Max: 3, Mean: 1.80
*   **Candidates per dialogue:** Min: 8, Max: 24, Mean: 15.17
*   **Footprints per dialogue:** Min: 10, Max: 40, Mean: 20.67
*   **Gold footprints per gap:** Min: 1, Max: 12, Mean: 2.88
*   **Gap Distribution:** {1 gap: 335 dialogues, 2 gaps: 337 dialogues, 3 gaps: 169 dialogues}

## 2. Text Lengths (Word Counts)
*   **Utterances:** Min 1, Max 76, Mean 10.55
*   **Candidates:** Min 2, Max 63, Mean 9.80
*   **Footprints:** Min 3, Max 33, Mean 10.96

## 3. Lexical & Semantic Overlap
**Jaccard Similarity (Word Overlap with Context):**
*   Gold Candidate vs Context: Mean: 0.093, Median: 0.088
*   Negative Candidate vs Context: Mean: 0.072, Median: 0.065
*   Inter-Candidate (Same Dialogue): Mean: 0.073, Max: 1.000

**TF-IDF Cosine Similarity:**
*   Gold Candidate vs Context: Mean 0.1112, Median 0.0528
*   Negative Candidate vs Context: Mean 0.0460, Median 0.0000

*Observation:* While Gold Candidates have higher average similarity to the context than negative candidates, the median scores for both are very low (0.05 vs 0.0). This highlights the core difficulty of the task: missing turns often cannot be recovered merely by looking for words that match the context. There are significant "hard negatives" that have overlapping vocabulary but are incorrect.

## 4. Footprint Distribution per Gap
*   1 footprint: 425 gaps
*   2 footprints: 186 gaps
*   3 footprints: 467 gaps
*   4 footprints: 233 gaps
*   5 footprints: 98 gaps
*   6 footprints: 57 gaps
*   7 footprints: 22 gaps
*   8 footprints: 17 gaps
*   9 footprints: 6 gaps
*   10 footprints: 3 gaps
*   11 footprints: 1 gap
*   12 footprints: 1 gap

*Observation:* A significant number of gaps are supported by 1 or 3 footprints, but the distribution has a long tail, up to 12 supporting footprints. Models need dynamic thresholding to predict anywhere from 0 to 12 footprints per gap.

## 5. Speaker Patterns
*   Most common speaker transitions: `[('A -> B', 3684), ('B -> A', 3203), ('B -> B', 1)]`
*   Dialogues are almost strictly dyadic alternating turns.

## 6. Takeaways for Modeling
*   **Hard Negatives:** Distractors are heavily curated. Relying strictly on surface-level TF-IDF cosine similarity will fail. Bilinear interactions or cross-encoders are necessary to capture latent relationships.
*   **Global Constraints:** The high candidate overlap within the same dialogue means candidate collision is a major threat. Utilizing assignment algorithms (like the Hungarian algorithm) is essential.
*   **Multi-Label Footprints:** The thresholding for footprint attachment must be flexible enough to allow for empty predictions and large lists alike.

## 7. Train / Validation Analysis of solution_chakra.py
A strict `GroupShuffleSplit` on `dialogue_id` was performed (80% train / 20% validation). The script was updated to match the official competition evaluation metrics.

**Results on Validation Set (169 dialogues):**
*   Gap Assignment Accuracy: 0.0830
*   Ranked Candidate MRR: 0.1943
*   Footprint Attachment Micro F1: 0.0237
*   Exact Dialogue Recovery: 0.0533
*   Dialogue-Balanced Accuracy: 0.0868

**FINAL SCORE:** 0.0891

*Observation:* The current `solution_chakra.py` AI Baseline performs poorly on a strictly split validation set compared to the leaderboard (AI Baseline public score: 0.4714). This suggests the current script's feature extraction (simple TF-IDF vectors + Bilinear scores mapped via simple GBDT) is not strong enough to handle unseen dialogues, likely suffering from heavy underfitting or inability to generalize the vocabulary, or the leaderboard public score is evaluated on a different distribution. The model is essentially guessing slightly above random chance.
