# Afterimage-Commonsense-Constrained - Model Comparison

This document tracks the iterative improvement in performance for the dataset. All models were evaluated strictly on the same 80/20 grouped shuffle split (grouped by `dialogue_id`) to prevent any data leakage.

## Overview of Iterations

1. **TF-IDF + Bilinear (Baseline)**: The original `solution_chakra.py` baseline relying on sparse bag-of-words exact matches.
2. **all-MiniLM-L6-v2 (Dense Bi-encoder)**: Transitioned to a dense semantic embedding approach. Extracted absolute differences and element-wise products of context vs. candidate embeddings to pass into a GBDT classifier.
3. **BAAI/bge-small-en-v1.5 (Dense Bi-encoder)**: Upgraded the semantic engine to the more powerful BAAI BGE model, keeping the same GBDT downstream classification logic.
4. **RinKana/bge-small-en-v1.5-afterimage**: Utilized the user's fine-tuned version of the BGE model, specifically tuned via Multiple Negatives Ranking Loss (MNRL) on this dataset's context-candidate matching task.

---

## Detailed Metric Comparison

| Metric | 1. TF-IDF Baseline | 2. MiniLM-L6-v2 | 3. BAAI bge-small | 4. RinKana bge (Fine-tuned) |
| :--- | :--- | :--- | :--- | :--- |
| **Gap Assignment Accuracy** | 0.0830 | 0.2353 | 0.2457 | **0.4291** |
| **Ranked Candidate MRR** | 0.1943 | 0.3996 | 0.4067 | **0.5938** |
| **Footprint Attachment Micro F1** | 0.0237 | 0.1075 | 0.1132 | **0.1184** |
| **Exact Dialogue Recovery** | 0.0533 | 0.1953 | 0.2189 | **0.4260** |
| **Dialogue-Balanced Accuracy** | 0.0868 | 0.2673 | 0.2811 | **0.4931** |
| --- | --- | --- | --- | --- |
| **FINAL COMPETITION SCORE** | **0.0891** | **0.2382** | **0.2491** | **0.4026** |

---

## Conclusion & Observations
* **Dense is necessary:** Transitioning from sparse TF-IDF matching to Dense Semantic Understanding (MiniLM) resulted in an immediate ~2.6x score multiplier. The dataset's "Hard Negatives" defeat surface-level word matching.
* **Fine-tuning makes the biggest difference:** While swapping from MiniLM to BAAI yielded a modest bump (0.238 -> 0.249), utilizing the *task-specific fine-tuned weights* (`RinKana`) caused the score to explode (0.249 -> 0.402).
* **State of the Pipeline:** The combination of `RinKana/bge-small-en-v1.5-afterimage` providing excellent base semantic representations and the HistGradientBoostingClassifier learning the specific gap thresholds allows the local validation score to comfortably approach the AI Baseline (0.471) on the public leaderboard. The `solution.py` script has been permanently updated to use these weights.