import json
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm

# --- Configuration ---
MIN_FP_THRESHOLD = 0.40
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def load_jsonl(path):
    data = []
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def build_gap_contexts(dialogue, gap_id):
    """Extracts both full context and the immediate left context."""
    left_context = []
    right_context = []
    gap_speaker = None
    target_found = False

    for turn in dialogue:
        if turn.get("gap_id") == gap_id:
            gap_speaker = turn.get("speaker")
            target_found = True
            continue
        text = turn.get("text")
        if text:
            utt_str = f"Speaker {turn.get('speaker')}: {text}"
            if not target_found:
                left_context.append(utt_str)
            else:
                right_context.append(utt_str)

    left_str = " ".join(left_context[-3:])
    right_str = " ".join(right_context[:3])
    immediate_left = left_context[-1] if left_context else ""

    full_context_text = (
        f"Missing turn by Speaker {gap_speaker}. "
        f"Context before: {left_str} "
        f"Context after: {right_str}"
    )
    return full_context_text, immediate_left, gap_speaker


def extract_features(ctx_text, left_text, cand_text, vectorizer, T_cand, T_left):
    """Calculates standard overlap + Advanced Bilinear Transition Scores."""
    vec_ctx = vectorizer.transform([ctx_text])
    vec_left = vectorizer.transform([left_text])
    vec_cand = vectorizer.transform([cand_text])
    
    # 1. Base Cosine Similarities
    sim_ctx = cosine_similarity(vec_ctx, vec_cand)[0][0]
    sim_left = cosine_similarity(vec_left, vec_cand)[0][0]
    
    # 2. Bilinear Semantic Bridge Scores (Context^T * T * Candidate)
    # This captures semantic links even with ZERO word overlap
    bilinear_ctx = vec_ctx.dot(T_cand).dot(vec_cand.T).toarray()[0][0]
    bilinear_left = vec_left.dot(T_left).dot(vec_cand.T).toarray()[0][0]
    
    # 3. Structural Features
    len_diff = abs(len(ctx_text.split()) - len(cand_text.split()))
    
    return [sim_ctx, sim_left, bilinear_ctx, bilinear_left, len_diff]


def extract_fp_features(cand_text, fp_text, vectorizer, T_fp):
    """Calculates features mapping Candidate directly to Footprint."""
    vec_cand = vectorizer.transform([cand_text])
    vec_fp = vectorizer.transform([fp_text])
    
    sim_tfidf = cosine_similarity(vec_cand, vec_fp)[0][0]
    bilinear_fp = vec_cand.dot(T_fp).dot(vec_fp.T).toarray()[0][0]
    
    return [sim_tfidf, bilinear_fp]


def main():
    if len(sys.argv) >= 3:
        public_dir = sys.argv[1]
        output_path = sys.argv[2]
    else:
        public_dir = "public"
        output_path = "submission.csv"

    train_path = os.path.join(public_dir, "train.jsonl")
    test_path = os.path.join(public_dir, "test.jsonl")

    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    # --- STAGE 1: Build the Offline Vocabulary ---
    print("Building TF-IDF Vocabulary Space...")
    all_texts = []
    gold_pairs_ctx = []
    gold_pairs_left = []
    gold_pairs_cand = []
    gold_pairs_fp_cand = []
    gold_pairs_fp = []

    # Gather texts and specifically isolate Gold pairs to build the Transition Matrices
    for data_split in [train_data, test_data]:
        for example in data_split:
            dialogue = example.get("dialogue", [])
            for turn in dialogue:
                if turn.get("text"): all_texts.append(turn["text"])
            
            cands = {c["turn_id"]: c["text"] for c in example.get("candidate_turns", [])}
            for text in cands.values(): all_texts.append(text)
            
            fps = {f["footprint_id"]: f["text"] for f in example.get("footprints", [])}
            for text in fps.values(): all_texts.append(text)

            if "answers" in example:
                for ans in example["answers"]:
                    gap_id = ans["gap_id"]
                    gold_cand = cands.get(ans["turn_id"], "")
                    if gold_cand:
                        ctx, left, _ = build_gap_contexts(dialogue, gap_id)
                        gold_pairs_ctx.append(ctx)
                        gold_pairs_left.append(left)
                        gold_pairs_cand.append(gold_cand)
                        
                        for fid in ans.get("supporting_footprints", []):
                            if fid in fps:
                                gold_pairs_fp_cand.append(gold_cand)
                                gold_pairs_fp.append(fps[fid])

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, stop_words="english")
    vectorizer.fit(all_texts)

    # --- STAGE 2: Construct the Bilinear Transition Matrices ---
    print("Calculating Semantic Bilinear Transition Matrices...")
    vec_gold_ctx = vectorizer.transform(gold_pairs_ctx)
    vec_gold_left = vectorizer.transform(gold_pairs_left)
    vec_gold_cand = vectorizer.transform(gold_pairs_cand)
    
    vec_gold_fp_cand = vectorizer.transform(gold_pairs_fp_cand)
    vec_gold_fp = vectorizer.transform(gold_pairs_fp)

    # T = X^T * Y (Creates a mapping of Context Words -> Candidate Words)
    T_cand = vec_gold_ctx.T.dot(vec_gold_cand)
    T_left = vec_gold_left.T.dot(vec_gold_cand)
    T_fp = vec_gold_fp_cand.T.dot(vec_gold_fp)

    # Normalize Matrices to prevent score explosion
    if T_cand.max() > 0: T_cand = T_cand / T_cand.max()
    if T_left.max() > 0: T_left = T_left / T_left.max()
    if T_fp.max() > 0: T_fp = T_fp / T_fp.max()

    # --- STAGE 3: Build Training Features with Hard Negatives ---
    print("Extracting Bilinear Features for GBDT Training...")
    X_cand, y_cand = [], []
    X_fp, y_fp = [], []

    for example in tqdm(train_data, desc="Encoding Train Pairs"):
        dialogue = example["dialogue"]
        candidates = {c["turn_id"]: c["text"] for c in example["candidate_turns"]}
        footprints = {f["footprint_id"]: f["text"] for f in example.get("footprints", [])}
        answers = example.get("answers", [])
        
        for ans in answers:
            gap_id = ans["gap_id"]
            gold_turn_id = ans["turn_id"]
            gold_turn_text = candidates.get(gold_turn_id, "")
            
            full_ctx, left_ctx, _ = build_gap_contexts(dialogue, gap_id)
            if not gold_turn_text: continue
                
            # Positive Candidate
            X_cand.append(extract_features(full_ctx, left_ctx, gold_turn_text, vectorizer, T_cand, T_left))
            y_cand.append(1)
            
            # Hard Negative Mining (Most similar incorrect candidates)
            neg_cands = [tid for tid in candidates.keys() if tid != gold_turn_id]
            neg_scores = []
            for tid in neg_cands:
                score = cosine_similarity(vectorizer.transform([gold_turn_text]), vectorizer.transform([candidates[tid]]))[0][0]
                neg_scores.append((tid, score))
            
            neg_scores.sort(key=lambda x: x[1], reverse=True)
            for tid in [x[0] for x in neg_scores[:5]]:
                X_cand.append(extract_features(full_ctx, left_ctx, candidates[tid], vectorizer, T_cand, T_left))
                y_cand.append(0)
                
            # Positive Footprints
            gold_fps = set(ans.get("supporting_footprints", []))
            for fid in gold_fps:
                if fid in footprints:
                    X_fp.append(extract_fp_features(gold_turn_text, footprints[fid], vectorizer, T_fp))
                    y_fp.append(1)
                    
            # Hard Negative Footprints
            neg_fps = [fid for fid in footprints.keys() if fid not in gold_fps]
            neg_fp_scores = []
            for fid in neg_fps:
                score = cosine_similarity(vectorizer.transform([gold_turn_text]), vectorizer.transform([footprints[fid]]))[0][0]
                neg_fp_scores.append((fid, score))
                
            neg_fp_scores.sort(key=lambda x: x[1], reverse=True)
            for fid in [x[0] for x in neg_fp_scores[:5]]:
                X_fp.append(extract_fp_features(gold_turn_text, footprints[fid], vectorizer, T_fp))
                y_fp.append(0)

    # --- STAGE 4: Train GBDT Models ---
    print("Training GBDT Models...")
    clf_cand = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    clf_cand.fit(X_cand, y_cand)
    
    clf_fp = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    if len(X_fp) > 0: clf_fp.fit(X_fp, y_fp)

    # --- STAGE 5: Inference & Global Constrained Optimization ---
    print("Executing predictions on Test Set...")
    all_rows = []

    for example in tqdm(test_data, desc="Processing Test Set"):
        dialogue_id = example["dialogue_id"]
        dialogue = example["dialogue"]
        candidates = example["candidate_turns"]
        footprints = example.get("footprints", [])

        gaps = [turn for turn in dialogue if turn.get("text") is None]
        if not gaps: continue

        cand_ids = [c["turn_id"] for c in candidates]
        cand_texts = [c["text"] for c in candidates]
        
        fp_ids = [f["footprint_id"] for f in footprints]
        fp_texts = [f["text"] for f in footprints]
        gap_ids = [g["gap_id"] for g in gaps]
        
        prob_matrix = np.zeros((len(gap_ids), len(cand_ids)))

        for i, g_id in enumerate(gap_ids):
            full_ctx, left_ctx, _ = build_gap_contexts(dialogue, g_id)
            for j, c_text in enumerate(cand_texts):
                feats = extract_features(full_ctx, left_ctx, c_text, vectorizer, T_cand, T_left)
                prob_matrix[i, j] = clf_cand.predict_proba([feats])[0][1]

        # Hungarian Algorithm
        cost_matrix = -prob_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_candidates = {gap_ids[r]: cand_ids[c] for r, c in zip(row_ind, col_ind)}

        for idx, g_id in enumerate(gap_ids):
            selected_turn = assigned_candidates[g_id]
            selected_cand_idx = cand_ids.index(selected_turn)

            scores = prob_matrix[idx]
            sorted_cand_indices = np.argsort(-scores)
            ranked_cand_ids = [selected_turn]
            for c_idx in sorted_cand_indices:
                cand_code = cand_ids[c_idx]
                if cand_code != selected_turn and cand_code not in ranked_cand_ids:
                    ranked_cand_ids.append(cand_code)
                if len(ranked_cand_ids) == 5: break

            # Footprint Prediction with Dynamic Threshold
            selected_fp_ids = []
            if len(fp_ids) > 0 and len(X_fp) > 0:
                cand_text = cand_texts[selected_cand_idx]
                fp_probs = []
                for f_text in fp_texts:
                    feats = extract_fp_features(cand_text, f_text, vectorizer, T_fp)
                    fp_probs.append(clf_fp.predict_proba([feats])[0][1])
                
                fp_probs = np.array(fp_probs)
                if len(fp_probs) > 0:
                    dynamic_thresh = max(MIN_FP_THRESHOLD, np.mean(fp_probs) + 0.1)
                    for f_idx, prob in enumerate(fp_probs):
                        if prob >= dynamic_thresh:
                            selected_fp_ids.append(fp_ids[f_idx])

            all_rows.append({
                "dialogue_id": dialogue_id,
                "gap_id": g_id,
                "selected_turn": selected_turn,
                "ranked_turns": json.dumps(ranked_cand_ids),
                "supporting_footprints": json.dumps(selected_fp_ids),
            })

    sub_df = pd.DataFrame(all_rows)
    sub_df = sub_df[["dialogue_id", "gap_id", "selected_turn", "ranked_turns", "supporting_footprints"]]
    sub_df.to_csv(output_path, index=False)
    print(f"Submission successfully written to {output_path}")

if __name__ == "__main__":
    main()
