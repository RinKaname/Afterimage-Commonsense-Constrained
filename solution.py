import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import HistGradientBoostingClassifier
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# --- Configuration ---
MIN_FP_THRESHOLD = 0.35
RANDOM_STATE = 42
MODEL_NAME = 'RinKana/bge-small-en-v1.5-afterimage'
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

    left_str = " ".join(left_context[-5:])
    right_str = " ".join(right_context[:5])

    immediate_left = left_context[-1] if left_context else ""
    immediate_right = right_context[0] if right_context else ""

    full_context_text = (
        f"Represent this dialogue context for retrieving the missing turn: "
        f"Speaker {gap_speaker} is replying. "
        f"Previous context: {left_str} "
        f"Following context: {right_str}"
    )
    return full_context_text, immediate_left, immediate_right

def build_cand_contexts(cand_text):
    return f"Candidate response: {cand_text}"

def cosine_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

def extract_cand_features(ctx_emb, left_emb, right_emb, cand_emb, fp_embs, ctx_text, cand_text):
    # Context features
    sim = cosine_sim(ctx_emb, cand_emb)
    diff = np.abs(ctx_emb - cand_emb)
    mult = ctx_emb * cand_emb
    len_diff = abs(len(ctx_text.split()) - len(cand_text.split())) / 100.0

    # Local coherence
    sim_left = cosine_sim(left_emb, cand_emb) if left_emb is not None else 0
    sim_right = cosine_sim(right_emb, cand_emb) if right_emb is not None else 0

    # Footprint Awareness!
    fp_max = 0
    fp_top3_mean = 0
    fp_top5_mean = 0

    if fp_embs is not None and len(fp_embs) > 0:
        fp_sims = [cosine_sim(cand_emb, f) for f in fp_embs]
        fp_sims.sort(reverse=True)
        fp_max = fp_sims[0]
        fp_top3_mean = np.mean(fp_sims[:3]) if len(fp_sims) >= 3 else np.mean(fp_sims)
        fp_top5_mean = np.mean(fp_sims[:5]) if len(fp_sims) >= 5 else np.mean(fp_sims)

    return np.concatenate([[sim, len_diff, sim_left, sim_right, fp_max, fp_top3_mean, fp_top5_mean], diff, mult])

def extract_fp_features(cand_emb, fp_emb):
    sim = cosine_sim(cand_emb, fp_emb)
    diff = np.abs(cand_emb - fp_emb)
    mult = cand_emb * fp_emb
    return np.concatenate([[sim], diff, mult])


def main():
    if len(sys.argv) >= 3:
        public_dir = sys.argv[1]
        output_path = sys.argv[2]
    else:
        public_dir = "."
        output_path = "submission.csv"

    train_path = os.path.join(public_dir, "train.jsonl")
    test_path = os.path.join(public_dir, "test.jsonl")

    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    print("Loading SentenceTransformer model...")
    encoder = SentenceTransformer(MODEL_NAME)

    print("Pre-computing train embeddings...")
    X_cand, y_cand = [], []
    X_fp, y_fp = [], []

    for example in tqdm(train_data, desc="Encoding Train Pairs"):
        dialogue = example["dialogue"]
        candidates = {c["turn_id"]: c["text"] for c in example["candidate_turns"]}
        footprints = {f["footprint_id"]: f["text"] for f in example.get("footprints", [])}
        answers = example.get("answers", [])

        cand_ids = list(candidates.keys())
        cand_texts = [build_cand_contexts(candidates[cid]) for cid in cand_ids]
        cand_embs = encoder.encode(cand_texts, show_progress_bar=False)
        cand_emb_map = {cid: emb for cid, emb in zip(cand_ids, cand_embs)}

        fp_ids = list(footprints.keys())
        fp_texts = [footprints[fid] for fid in fp_ids]
        fp_embs = encoder.encode(fp_texts, show_progress_bar=False) if fp_texts else []
        fp_emb_map = {fid: emb for fid, emb in zip(fp_ids, fp_embs)}

        for ans in answers:
            gap_id = ans["gap_id"]
            gold_turn_id = ans["turn_id"]

            full_ctx, left_ctx, right_ctx = build_gap_contexts(dialogue, gap_id)
            ctx_emb = encoder.encode([full_ctx], show_progress_bar=False)[0]
            left_emb = encoder.encode([left_ctx], show_progress_bar=False)[0] if left_ctx else None
            right_emb = encoder.encode([right_ctx], show_progress_bar=False)[0] if right_ctx else None

            # Positive Candidate
            if gold_turn_id in cand_emb_map:
                gold_emb = cand_emb_map[gold_turn_id]
                X_cand.append(extract_cand_features(ctx_emb, left_emb, right_emb, gold_emb, fp_embs, full_ctx, candidates[gold_turn_id]))
                y_cand.append(1)

                # Hard Negative Mining (Most similar incorrect candidates)
                neg_cands = [tid for tid in candidates.keys() if tid != gold_turn_id]
                neg_scores = [(tid, cosine_sim(ctx_emb, cand_emb_map[tid])) for tid in neg_cands]
                neg_scores.sort(key=lambda x: x[1], reverse=True)
                for tid, _ in neg_scores[:5]:
                    X_cand.append(extract_cand_features(ctx_emb, left_emb, right_emb, cand_emb_map[tid], fp_embs, full_ctx, candidates[tid]))
                    y_cand.append(0)

                # Positive Footprints
                gold_fps = set(ans.get("supporting_footprints", []))
                for fid in gold_fps:
                    if fid in fp_emb_map:
                        X_fp.append(extract_fp_features(gold_emb, fp_emb_map[fid]))
                        y_fp.append(1)

                # Hard Negative Footprints
                neg_fps = [fid for fid in footprints.keys() if fid not in gold_fps]
                neg_fp_scores = [(fid, cosine_sim(gold_emb, fp_emb_map[fid])) for fid in neg_fps]
                neg_fp_scores.sort(key=lambda x: x[1], reverse=True)
                for fid, _ in neg_fp_scores[:5]:
                    X_fp.append(extract_fp_features(gold_emb, fp_emb_map[fid]))
                    y_fp.append(0)

    X_cand, y_cand = np.array(X_cand), np.array(y_cand)
    X_fp, y_fp = np.array(X_fp), np.array(y_fp)

    print("Training GBDT Candidate Model...")
    clf_cand = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=500, early_stopping=True, l2_regularization=0.1, learning_rate=0.05)
    clf_cand.fit(X_cand, y_cand)

    print("Training GBDT Footprint Model...")
    clf_fp = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=500, early_stopping=True, l2_regularization=0.1, learning_rate=0.05)
    if len(X_fp) > 0: clf_fp.fit(X_fp, y_fp)

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
        cand_embs_texts = [build_cand_contexts(c) for c in cand_texts]
        cand_embs = encoder.encode(cand_embs_texts, show_progress_bar=False)

        fp_ids = [f["footprint_id"] for f in footprints]
        fp_texts = [f["text"] for f in footprints]
        fp_embs = encoder.encode(fp_texts, show_progress_bar=False) if fp_texts else []
        gap_ids = [g["gap_id"] for g in gaps]

        prob_matrix = np.zeros((len(gap_ids), len(cand_ids)))

        for i, g_id in enumerate(gap_ids):
            full_ctx, left_ctx, right_ctx = build_gap_contexts(dialogue, g_id)
            ctx_emb = encoder.encode([full_ctx], show_progress_bar=False)[0]
            left_emb = encoder.encode([left_ctx], show_progress_bar=False)[0] if left_ctx else None
            right_emb = encoder.encode([right_ctx], show_progress_bar=False)[0] if right_ctx else None

            feats_list = []
            for j, c_text in enumerate(cand_texts):
                feats = extract_cand_features(ctx_emb, left_emb, right_emb, cand_embs[j], fp_embs, full_ctx, c_text)
                feats_list.append(feats)

            probs = clf_cand.predict_proba(feats_list)[:, 1]
            prob_matrix[i, :] = probs

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

            selected_fp_ids = []
            if len(fp_ids) > 0 and len(X_fp) > 0:
                cand_emb = cand_embs[selected_cand_idx]
                feats_list = []
                for f_emb in fp_embs:
                    feats = extract_fp_features(cand_emb, f_emb)
                    feats_list.append(feats)

                fp_probs = clf_fp.predict_proba(feats_list)[:, 1]
                if len(fp_probs) > 0:
                    dynamic_thresh = max(MIN_FP_THRESHOLD, np.mean(fp_probs) + 0.15)
                    for f_idx, prob in enumerate(fp_probs):
                        if prob >= dynamic_thresh:
                            selected_fp_ids.append(fp_ids[f_idx])

            ranking_value = json.dumps(ranked_cand_ids)
            footprint_value = json.dumps(selected_fp_ids)

            all_rows.append({
                "dialogue_id": dialogue_id,
                "gap_id": g_id,
                "selected_turn": selected_turn,
                "ranked_turns": ranking_value,
                "supporting_footprints": footprint_value,
            })

    sub_df = pd.DataFrame(all_rows)
    sub_df = sub_df[["dialogue_id", "gap_id", "selected_turn", "ranked_turns", "supporting_footprints"]]
    sub_df.to_csv(output_path, index=False)
    print(f"Submission successfully written to {output_path}")

if __name__ == "__main__":
    main()
