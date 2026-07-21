import json
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def load_jsonl(path):
    data = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
    return data

def build_gap_contexts(dialogue, gap_id):
    left_context = []
    right_context = []
    target_found = False

    for turn in dialogue:
        if turn.get("gap_id") == gap_id:
            target_found = True
            continue
        text = turn.get("text")
        if text:
            utt_str = text
            if not target_found:
                left_context.append(utt_str)
            else:
                right_context.append(utt_str)

    left_str = " ".join(left_context[-3:])
    right_str = " ".join(right_context[:3])
    full_context_text = f"{left_str} {right_str}"
    return full_context_text

def run_eda(train_path="train.jsonl"):
    train_data = load_jsonl(train_path)

    print("==================================================")
    print("DEEP EDA RESULTS")
    print("==================================================")

    utt_lengths = []
    cand_lengths = []
    fp_lengths = []

    for example in train_data:
        for turn in example.get("dialogue", []):
            if turn.get("text"):
                utt_lengths.append(len(turn["text"].split()))
        for cand in example.get("candidate_turns", []):
            cand_lengths.append(len(cand["text"].split()))
        for fp in example.get("footprints", []):
            fp_lengths.append(len(fp["text"].split()))

    print(f"Utterance Word Count: Min {np.min(utt_lengths)}, Max {np.max(utt_lengths)}, Mean {np.mean(utt_lengths):.2f}")
    print(f"Candidate Word Count: Min {np.min(cand_lengths)}, Max {np.max(cand_lengths)}, Mean {np.mean(cand_lengths):.2f}")
    print(f"Footprint Word Count: Min {np.min(fp_lengths)}, Max {np.max(fp_lengths)}, Mean {np.mean(fp_lengths):.2f}")

    # TF-IDF / Cosine Similarity
    gold_sims = []
    neg_sims = []

    all_texts = []
    for example in train_data:
        for turn in example.get("dialogue", []):
            if turn.get("text"): all_texts.append(turn["text"])
        for cand in example.get("candidate_turns", []):
            all_texts.append(cand["text"])

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(all_texts)

    for example in train_data:
        cands = {c["turn_id"]: c["text"] for c in example.get("candidate_turns", [])}
        for ans in example.get("answers", []):
            gap_id = ans["gap_id"]
            gold_id = ans["turn_id"]
            gold_text = cands.get(gold_id, "")

            ctx = build_gap_contexts(example["dialogue"], gap_id)
            if not ctx.strip() or not gold_text:
                continue

            vec_ctx = vectorizer.transform([ctx])
            vec_gold = vectorizer.transform([gold_text])
            gold_sims.append(cosine_similarity(vec_ctx, vec_gold)[0][0])

            for cid, ctext in cands.items():
                if cid != gold_id:
                    vec_neg = vectorizer.transform([ctext])
                    neg_sims.append(cosine_similarity(vec_ctx, vec_neg)[0][0])

    print(f"\nCosine Sim - Gold vs Context: Mean {np.mean(gold_sims):.4f}, Median {np.median(gold_sims):.4f}")
    print(f"Cosine Sim - Neg vs Context: Mean {np.mean(neg_sims):.4f}, Median {np.median(neg_sims):.4f}")

    # Footprints per gap distribution
    fp_counts = []
    for example in train_data:
        for ans in example.get("answers", []):
            fp_counts.append(len(ans.get("supporting_footprints", [])))

    print(f"\nFootprints per gap:")
    counts = Counter(fp_counts)
    for k in sorted(counts.keys()):
        print(f"  {k} footprints: {counts[k]} gaps")

if __name__ == '__main__':
    run_eda()
