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
from sklearn.model_selection import GroupShuffleSplit
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
    vec_ctx = vectorizer.transform([ctx_text])
    vec_left = vectorizer.transform([left_text])
    vec_cand = vectorizer.transform([cand_text])

    sim_ctx = cosine_similarity(vec_ctx, vec_cand)[0][0]
    sim_left = cosine_similarity(vec_left, vec_cand)[0][0]

    bilinear_ctx = vec_ctx.dot(T_cand).dot(vec_cand.T).toarray()[0][0]
    bilinear_left = vec_left.dot(T_left).dot(vec_cand.T).toarray()[0][0]

    len_diff = abs(len(ctx_text.split()) - len(cand_text.split()))

    return [sim_ctx, sim_left, bilinear_ctx, bilinear_left, len_diff]

def extract_fp_features(cand_text, fp_text, vectorizer, T_fp):
    vec_cand = vectorizer.transform([cand_text])
    vec_fp = vectorizer.transform([fp_text])

    sim_tfidf = cosine_similarity(vec_cand, vec_fp)[0][0]
    bilinear_fp = vec_cand.dot(T_fp).dot(vec_fp.T).toarray()[0][0]

    return [sim_tfidf, bilinear_fp]

def main():
    train_path = "train.jsonl"
    all_data = load_jsonl(train_path)

    dialogue_ids = [ex["dialogue_id"] for ex in all_data]
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    train_idx, val_idx = next(gss.split(all_data, groups=dialogue_ids))
    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in val_idx]

    print(f"Train Dialogues: {len(train_data)}")
    print(f"Val Dialogues: {len(val_data)}")

    # STAGE 1: Vocabulary
    all_texts = []
    gold_pairs_ctx, gold_pairs_left, gold_pairs_cand, gold_pairs_fp_cand, gold_pairs_fp = [], [], [], [], []

    for example in train_data: # Only fit on train
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

    # STAGE 2: Transition Matrices
    vec_gold_ctx = vectorizer.transform(gold_pairs_ctx)
    vec_gold_left = vectorizer.transform(gold_pairs_left)
    vec_gold_cand = vectorizer.transform(gold_pairs_cand)
    vec_gold_fp_cand = vectorizer.transform(gold_pairs_fp_cand)
    vec_gold_fp = vectorizer.transform(gold_pairs_fp)

    T_cand = vec_gold_ctx.T.dot(vec_gold_cand)
    T_left = vec_gold_left.T.dot(vec_gold_cand)
    T_fp = vec_gold_fp_cand.T.dot(vec_gold_fp)

    if T_cand.max() > 0: T_cand = T_cand / T_cand.max()
    if T_left.max() > 0: T_left = T_left / T_left.max()
    if T_fp.max() > 0: T_fp = T_fp / T_fp.max()

    # STAGE 3: Extract Features
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

            X_cand.append(extract_features(full_ctx, left_ctx, gold_turn_text, vectorizer, T_cand, T_left))
            y_cand.append(1)

            neg_cands = [tid for tid in candidates.keys() if tid != gold_turn_id]
            neg_scores = []
            for tid in neg_cands:
                score = cosine_similarity(vectorizer.transform([gold_turn_text]), vectorizer.transform([candidates[tid]]))[0][0]
                neg_scores.append((tid, score))

            neg_scores.sort(key=lambda x: x[1], reverse=True)
            for tid in [x[0] for x in neg_scores[:5]]:
                X_cand.append(extract_features(full_ctx, left_ctx, candidates[tid], vectorizer, T_cand, T_left))
                y_cand.append(0)

            gold_fps = set(ans.get("supporting_footprints", []))
            for fid in gold_fps:
                if fid in footprints:
                    X_fp.append(extract_fp_features(gold_turn_text, footprints[fid], vectorizer, T_fp))
                    y_fp.append(1)

            neg_fps = [fid for fid in footprints.keys() if fid not in gold_fps]
            neg_fp_scores = []
            for fid in neg_fps:
                score = cosine_similarity(vectorizer.transform([gold_turn_text]), vectorizer.transform([footprints[fid]]))[0][0]
                neg_fp_scores.append((fid, score))

            neg_fp_scores.sort(key=lambda x: x[1], reverse=True)
            for fid in [x[0] for x in neg_fp_scores[:5]]:
                X_fp.append(extract_fp_features(gold_turn_text, footprints[fid], vectorizer, T_fp))
                y_fp.append(0)

    # STAGE 4: Train Models
    clf_cand = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    clf_cand.fit(X_cand, y_cand)

    clf_fp = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    if len(X_fp) > 0: clf_fp.fit(X_fp, y_fp)

    # STAGE 5: Evaluation on Validation
    all_expected_gaps = 0
    correct_gaps = 0
    mrr_sum = 0
    tp_fp, fp_fp, fn_fp = 0, 0, 0
    exact_dialogues = 0
    dialogue_accuracies = []

    for example in tqdm(val_data, desc="Validating"):
        dialogue = example["dialogue"]
        candidates = example["candidate_turns"]
        footprints = example.get("footprints", [])
        answers = example.get("answers", [])

        gold_map = {ans["gap_id"]: ans for ans in answers}

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

        cost_matrix = -prob_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_candidates = {gap_ids[r]: cand_ids[c] for r, c in zip(row_ind, col_ind)}

        dialogue_correct = 0
        dialogue_all_valid = True

        for idx, g_id in enumerate(gap_ids):
            all_expected_gaps += 1
            if g_id not in gold_map: continue

            gold_ans = gold_map[g_id]
            gold_turn = gold_ans["turn_id"]
            gold_fps = set(gold_ans.get("supporting_footprints", []))

            selected_turn = assigned_candidates[g_id]
            selected_cand_idx = cand_ids.index(selected_turn)

            # Accuracy
            is_correct = (selected_turn == gold_turn)
            if is_correct:
                correct_gaps += 1
                dialogue_correct += 1
            else:
                dialogue_all_valid = False

            # Ranking MRR
            scores = prob_matrix[idx]
            sorted_cand_indices = np.argsort(-scores)
            ranked_cand_ids = [selected_turn]
            for c_idx in sorted_cand_indices:
                cand_code = cand_ids[c_idx]
                if cand_code != selected_turn and cand_code not in ranked_cand_ids:
                    ranked_cand_ids.append(cand_code)
                if len(ranked_cand_ids) == 5: break

            try:
                rank = ranked_cand_ids.index(gold_turn) + 1
                mrr_sum += (1.0 / rank)
            except ValueError:
                pass # Rank is 0 if not in top 5

            # Footprint Prediction
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

            pred_fps = set(selected_fp_ids)
            # F1 Calculation components (only counted if assigned to correct gap)
            # The prompt says: "A correct footprint assigned to the wrong gap is incorrect."
            tp_footprint_local = len(pred_fps.intersection(gold_fps))
            fp_footprint_local = len(pred_fps - gold_fps)
            fn_footprint_local = len(gold_fps - pred_fps)

            tp_fp += tp_footprint_local
            fp_fp += fp_footprint_local
            fn_fp += fn_footprint_local

        dialogue_accuracies.append(dialogue_correct / len(gap_ids))
        if dialogue_all_valid and dialogue_correct == len(gap_ids):
            exact_dialogues += 1

    # Final Metric Calculation
    gap_acc = correct_gaps / all_expected_gaps if all_expected_gaps > 0 else 0
    mrr = mrr_sum / all_expected_gaps if all_expected_gaps > 0 else 0

    fp_f1 = 0
    if (2 * tp_fp + fp_fp + fn_fp) > 0:
        fp_f1 = (2 * tp_fp) / (2 * tp_fp + fp_fp + fn_fp)

    exact_rec = exact_dialogues / len(val_data) if val_data else 0
    bal_acc = np.mean(dialogue_accuracies) if dialogue_accuracies else 0

    final_score = (0.40 * gap_acc) + (0.20 * mrr) + (0.20 * fp_f1) + (0.15 * exact_rec) + (0.05 * bal_acc)

    print("==================================================")
    print("VALIDATION METRICS")
    print("==================================================")
    print(f"Gap Assignment Accuracy:      {gap_acc:.4f}")
    print(f"Ranked Candidate MRR:         {mrr:.4f}")
    print(f"Footprint Attachment Micro F1:{fp_f1:.4f}")
    print(f"Exact Dialogue Recovery:      {exact_rec:.4f}")
    print(f"Dialogue-Balanced Accuracy:   {bal_acc:.4f}")
    print(f"--------------------------------------------------")
    print(f"FINAL SCORE:                  {final_score:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
