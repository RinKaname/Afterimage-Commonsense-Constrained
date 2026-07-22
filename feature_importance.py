import json
import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.inspection import permutation_importance
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# --- Configuration ---
RANDOM_STATE = 42
MODEL_NAME = 'RinKana/bge-small-en-v1.5-afterimage'
np.random.seed(RANDOM_STATE)

FEATURE_NAMES = [
    "Context Cosine Sim",
    "Length Difference",
    "Left Turn Cosine Sim",
    "Right Turn Cosine Sim",
    "Left Turn Lexical Overlap",
    "Right Turn Lexical Overlap",
    "Max Footprint Cosine Sim",
    "Top-3 Footprint Sim Mean",
    "Top-5 Footprint Sim Mean",
    "Max Footprint Lexical Overlap",
    # Note: The remaining 384*2 features are the absolute differences and element-wise products.
    # We will group them as "Dense Interaction Features" for the final output.
]

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

def compute_lexical_overlap(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / float(len(set1.union(set2)))

def extract_cand_features(ctx_emb, left_emb, right_emb, cand_emb, fp_embs, ctx_text, cand_text, left_text, right_text, fp_texts):
    sim = cosine_sim(ctx_emb, cand_emb)
    diff = np.abs(ctx_emb - cand_emb)
    mult = ctx_emb * cand_emb
    len_diff = abs(len(ctx_text.split()) - len(cand_text.split())) / 100.0

    lex_left = compute_lexical_overlap(left_text, cand_text) if left_text else 0
    lex_right = compute_lexical_overlap(right_text, cand_text) if right_text else 0

    sim_left = cosine_sim(left_emb, cand_emb) if left_emb is not None else 0
    sim_right = cosine_sim(right_emb, cand_emb) if right_emb is not None else 0

    fp_max = 0
    fp_top3_mean = 0
    fp_top5_mean = 0
    fp_lex_max = 0

    if fp_embs is not None and len(fp_embs) > 0:
        fp_sims = [cosine_sim(cand_emb, f) for f in fp_embs]
        fp_sims.sort(reverse=True)
        fp_max = fp_sims[0]
        fp_top3_mean = np.mean(fp_sims[:3]) if len(fp_sims) >= 3 else np.mean(fp_sims)
        fp_top5_mean = np.mean(fp_sims[:5]) if len(fp_sims) >= 5 else np.mean(fp_sims)

    if fp_texts:
        lex_sims = [compute_lexical_overlap(cand_text, f) for f in fp_texts]
        fp_lex_max = max(lex_sims) if lex_sims else 0

    scalar_features = [sim, len_diff, sim_left, sim_right, lex_left, lex_right, fp_max, fp_top3_mean, fp_top5_mean, fp_lex_max]
    return np.concatenate([scalar_features, diff, mult])


def main():
    train_path = "train.jsonl"
    all_data = load_jsonl(train_path)

    dialogue_ids = [ex["dialogue_id"] for ex in all_data]
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    train_idx, val_idx = next(gss.split(all_data, groups=dialogue_ids))
    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in val_idx]

    print("Loading SentenceTransformer model...")
    encoder = SentenceTransformer(MODEL_NAME)

    def process_split(data, desc):
        X, y = [], []
        for example in tqdm(data, desc=desc):
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

            for ans in answers:
                gap_id = ans["gap_id"]
                gold_turn_id = ans["turn_id"]

                full_ctx, left_ctx, right_ctx = build_gap_contexts(dialogue, gap_id)
                ctx_emb = encoder.encode([full_ctx], show_progress_bar=False)[0]
                left_emb = encoder.encode([left_ctx], show_progress_bar=False)[0] if left_ctx else None
                right_emb = encoder.encode([right_ctx], show_progress_bar=False)[0] if right_ctx else None

                if gold_turn_id in cand_emb_map:
                    gold_emb = cand_emb_map[gold_turn_id]
                    X.append(extract_cand_features(ctx_emb, left_emb, right_emb, gold_emb, fp_embs, full_ctx, candidates[gold_turn_id], left_ctx, right_ctx, fp_texts))
                    y.append(1)

                    neg_cands = [tid for tid in candidates.keys() if tid != gold_turn_id]
                    neg_scores = [(tid, cosine_sim(ctx_emb, cand_emb_map[tid])) for tid in neg_cands]
                    neg_scores.sort(key=lambda x: x[1], reverse=True)
                    for tid, _ in neg_scores[:5]:
                        X.append(extract_cand_features(ctx_emb, left_emb, right_emb, cand_emb_map[tid], fp_embs, full_ctx, candidates[tid], left_ctx, right_ctx, fp_texts))
                        y.append(0)
        return np.array(X), np.array(y)

    print("Extracting Train Features...")
    X_train, y_train = process_split(train_data, "Train Data")
    print("Extracting Val Features...")
    X_val, y_val = process_split(val_data, "Val Data")

    print("Training GBDT Candidate Model...")
    clf_cand = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=500, early_stopping=True, l2_regularization=0.1, learning_rate=0.05)
    clf_cand.fit(X_train, y_train)

    print(f"Val Accuracy: {clf_cand.score(X_val, y_val):.4f}")

    print("Calculating Permutation Feature Importance (this may take a minute)...")
    # We only permute the first 10 scalar features.
    # Permuting all 700+ dense interaction features is too slow and not interpretable.
    results = []
    for idx, feature_name in enumerate(FEATURE_NAMES):
        # Temporarily shuffle just this column in X_val to see how much the score drops
        X_val_shuffled = X_val.copy()
        np.random.shuffle(X_val_shuffled[:, idx])
        score_drop = clf_cand.score(X_val, y_val) - clf_cand.score(X_val_shuffled, y_val)
        results.append((feature_name, score_drop))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\n==================================================")
    print("FEATURE IMPORTANCE (Accuracy Drop when shuffled)")
    print("==================================================")
    for name, drop in results:
        print(f"{name:35s}: {drop:.4f}")
    print("==================================================")

if __name__ == "__main__":
    main()
