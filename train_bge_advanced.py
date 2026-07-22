import json
import os
import random
import torch
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments, BatchSamplers
from datasets import Dataset

# --- Configuration ---
MODEL_NAME = 'BAAI/bge-small-en-v1.5'
RANDOM_STATE = 42
OUTPUT_DIR = "bge_finetuned_advanced"
EPOCHS = 4
BATCH_SIZE = 16

random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
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

    full_context_text = (
        f"Represent this dialogue context for retrieving the missing turn: "
        f"Speaker {gap_speaker} is replying. "
        f"Previous context: {left_str} "
        f"Following context: {right_str}"
    )
    return full_context_text

def build_cand_contexts(cand_text):
    return f"Candidate response: {cand_text}"

def build_fp_contexts(fp_text):
    return f"Footprint context: {fp_text}"

def compute_lexical_overlap(text1, text2):
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / float(len(set1.union(set2)))

def prepare_advanced_data(data):
    """
    Creates Multi-Task data with Hard Negative Mining.
    Task 1: Context -> Candidate (with Hard Negatives)
    Task 2: Candidate -> Footprint
    """
    examples_context_cand = []
    examples_cand_fp = []

    queries = {}
    corpus = {}
    relevant_docs = {}
    global_query_idx = 0

    for example in data:
        dialogue = example["dialogue"]
        candidates = {c["turn_id"]: c["text"] for c in example["candidate_turns"]}
        footprints = {f["footprint_id"]: f["text"] for f in example.get("footprints", [])}

        # Add all candidates to corpus for the evaluator
        for cid, text in candidates.items():
            corpus_key = f"{example['dialogue_id']}_{cid}"
            corpus[corpus_key] = build_cand_contexts(text)

        answers = example.get("answers", [])

        for ans in answers:
            gap_id = ans["gap_id"]
            gold_turn_id = ans["turn_id"]
            gold_fps = set(ans.get("supporting_footprints", []))

            if gold_turn_id not in candidates:
                continue

            full_ctx = build_gap_contexts(dialogue, gap_id)
            gold_text = build_cand_contexts(candidates[gold_turn_id])

            # --- Hard Negative Mining ---
            # We want to find a candidate that is WRONG, but shares a lot of words with the context
            neg_cands = []
            for cid, ctext in candidates.items():
                if cid != gold_turn_id:
                    lex_sim = compute_lexical_overlap(full_ctx, ctext)
                    neg_cands.append((cid, lex_sim))

            # Sort by highest lexical overlap (the "trickiest" surface-level matches)
            neg_cands.sort(key=lambda x: x[1], reverse=True)

            # Take the hardest negative
            hard_neg_text = build_cand_contexts(candidates[neg_cands[0][0]]) if neg_cands else ""

            if hard_neg_text:
                # Triplet: [Anchor, Positive, Hard Negative]
                examples_context_cand.append(InputExample(texts=[full_ctx, gold_text, hard_neg_text]))
            else:
                examples_context_cand.append(InputExample(texts=[full_ctx, gold_text]))

            # --- Task 2: Candidate to Footprint Mapping ---
            # Teach the model that the gold candidate embeds similarly to its supporting footprints
            for fid in gold_fps:
                if fid in footprints:
                    fp_text = build_fp_contexts(footprints[fid])
                    examples_cand_fp.append(InputExample(texts=[gold_text, fp_text]))

            # Setup evaluator data
            q_id = f"q_{global_query_idx}"
            global_query_idx += 1
            queries[q_id] = full_ctx
            corpus_key = f"{example['dialogue_id']}_{gold_turn_id}"
            relevant_docs[q_id] = set([corpus_key])

    return examples_context_cand, examples_cand_fp, queries, corpus, relevant_docs

def main():
    train_path = "train.jsonl"
    all_data = load_jsonl(train_path)

    dialogue_ids = [ex["dialogue_id"] for ex in all_data]
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=RANDOM_STATE)

    train_idx, val_idx = next(gss.split(all_data, groups=dialogue_ids))
    train_data = [all_data[i] for i in train_idx]
    val_data = [all_data[i] for i in val_idx]

    print("Preparing Advanced Data with Hard Negatives...")
    train_ctx_cand, train_cand_fp, _, _, _ = prepare_advanced_data(train_data)
    _, _, val_queries, val_corpus, val_relevant_docs = prepare_advanced_data(val_data)

    print(f"Context-Candidate Pairs: {len(train_ctx_cand)}")
    print(f"Candidate-Footprint Pairs: {len(train_cand_fp)}")

    # Convert to HuggingFace Dataset
    # We pad the Cand-FP task with empty strings so it fits in the same dataset structure (triplets)
    # The MultipleNegativesRankingLoss ignores the third column if it's empty during standard batching
    anchors = [ex.texts[0] for ex in train_ctx_cand] + [ex.texts[0] for ex in train_cand_fp]
    positives = [ex.texts[1] for ex in train_ctx_cand] + [ex.texts[1] for ex in train_cand_fp]

    # Only Context-Cand has hard negatives
    hard_negatives = [ex.texts[2] if len(ex.texts) > 2 else "" for ex in train_ctx_cand]
    hard_negatives += [""] * len(train_cand_fp)

    train_dataset = Dataset.from_dict({
        "anchor": anchors,
        "positive": positives,
        "negative": hard_negatives
    })

    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. Setup Loss
    # MultipleNegativesRankingLoss naturally supports (Anchor, Positive, Negative) triplets!
    # It will use the hard negatives + all other items in the batch as negatives.
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # 4. Setup Evaluator
    evaluator = InformationRetrievalEvaluator(
        queries=val_queries,
        corpus=val_corpus,
        relevant_docs=val_relevant_docs,
        name="gap-to-candidate-eval",
        show_progress_bar=True,
    )

    # 5. Training Arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=0.1,
        fp16=True if torch.cuda.is_available() else False,
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=50,
        load_best_model_at_end=True,
    )

    # 6. Train!
    print("Starting Multi-Task Hard-Negative Fine-tuning...")
    from sentence_transformers import SentenceTransformerTrainer

    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    trainer.train()

    print(f"Training complete. Best model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
