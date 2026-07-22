import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm.auto import tqdm

# --- Configuration ---
BI_ENCODER_MODEL = 'BAAI/bge-small-en-v1.5'
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
MIN_FP_THRESHOLD = 0.35
TOP_K_CANDIDATES = 15 # Stage 1 retrieval cutoff
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

    full_context_text = (
        f"Represent this dialogue context for retrieving the missing turn: "
        f"Speaker {gap_speaker} is replying. "
        f"Previous context: {left_str} "
        f"Following context: {right_str}"
    )
    return full_context_text, left_str, right_str, gap_speaker

def build_cand_contexts(cand_text):
    return f"Candidate response: {cand_text}"

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

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

    print(f"Loading Bi-Encoder ({BI_ENCODER_MODEL})...")
    bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
    print(f"Loading Cross-Encoder ({CROSS_ENCODER_MODEL})...")
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    # Evaluation
    all_expected_gaps = 0
    correct_gaps = 0
    mrr_sum = 0
    tp_fp, fp_fp, fn_fp = 0, 0, 0
    exact_dialogues = 0
    dialogue_accuracies = []

    for example in tqdm(val_data, desc="Validating Stage 1 & 2 Pipeline"):
        dialogue = example["dialogue"]
        candidates = example["candidate_turns"]
        footprints = example.get("footprints", [])
        answers = example.get("answers", [])

        gold_map = {ans["gap_id"]: ans for ans in answers}
        gaps = [turn for turn in dialogue if turn.get("text") is None]
        if not gaps: continue

        cand_ids = [c["turn_id"] for c in candidates]
        cand_texts = [c["text"] for c in candidates]
        cand_embs_texts = [build_cand_contexts(c) for c in cand_texts]
        cand_embs = bi_encoder.encode(cand_embs_texts, show_progress_bar=False)

        fp_ids = [f["footprint_id"] for f in footprints]
        fp_texts = [f["text"] for f in footprints]
        fp_embs = bi_encoder.encode(fp_texts, show_progress_bar=False) if fp_texts else []
        gap_ids = [g["gap_id"] for g in gaps]

        # We will build a prob matrix based on CROSS ENCODER scores
        prob_matrix = np.zeros((len(gap_ids), len(cand_ids)))
        prob_matrix.fill(-9999) # Default to very low score so non-retrieved candidates aren't picked

        # STAGE 1: Bi-Encoder Retrieval
        for i, g_id in enumerate(gap_ids):
            full_ctx, left_str, right_str, gap_speaker = build_gap_contexts(dialogue, g_id)
            ctx_emb = bi_encoder.encode([full_ctx], show_progress_bar=False)[0]

            sims = []
            for j in range(len(cand_texts)):
                sim = cosine_sim(ctx_emb, cand_embs[j])
                sims.append((j, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            top_cand_indices = [x[0] for x in sims[:TOP_K_CANDIDATES]]

            # Retrieve relevant footprints for the context to inject into Cross-Encoder
            top_fp_str = ""
            if len(fp_embs) > 0:
                fp_sims = [(f_idx, cosine_sim(ctx_emb, f_emb)) for f_idx, f_emb in enumerate(fp_embs)]
                fp_sims.sort(key=lambda x: x[1], reverse=True)
                top_fp_texts = [fp_texts[x[0]] for x in fp_sims[:3]]
                top_fp_str = " Latent footprints: " + " ".join(top_fp_texts)

            # STAGE 2: Cross-Encoder Reranking
            cross_inputs = []
            for j in top_cand_indices:
                cand_text = cand_texts[j]
                query = f"Dialogue context: {left_str} [MISSING TURN] {right_str}.{top_fp_str}"
                cross_inputs.append([query, cand_text])

            cross_scores = cross_encoder.predict(cross_inputs)

            for rank_idx, j in enumerate(top_cand_indices):
                prob_matrix[i, j] = cross_scores[rank_idx]

        # Assignment
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

            # Footprint Prediction (Fallback to Bi-Encoder thresholding for footprints)
            selected_fp_ids = []
            if len(fp_ids) > 0:
                cand_emb = cand_embs[selected_cand_idx]
                fp_probs = [cosine_sim(cand_emb, f_emb) for f_emb in fp_embs]
                dynamic_thresh = max(MIN_FP_THRESHOLD, np.mean(fp_probs) + 0.15)
                for f_idx, prob in enumerate(fp_probs):
                    if prob >= dynamic_thresh:
                        selected_fp_ids.append(fp_ids[f_idx])

            pred_fps = set(selected_fp_ids)
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
    print("VALIDATION METRICS (STAGE 1 & 2 CROSS-ENCODER)")
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
