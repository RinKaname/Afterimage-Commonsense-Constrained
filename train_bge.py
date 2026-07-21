import json
import os
import random
import torch
from sklearn.model_selection import GroupShuffleSplit
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from datasets import Dataset

# --- Configuration ---
MODEL_NAME = 'BAAI/bge-small-en-v1.5'
RANDOM_STATE = 42
OUTPUT_DIR = "bge_finetuned"
EPOCHS = 3
BATCH_SIZE = 16

random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

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

def prepare_data(data):
    """
    Creates pairs for MultipleNegativesRankingLoss.
    We will focus primarily on context -> correct candidate matching.
    """
    examples = []
    queries = {} # query_id -> query_text
    corpus = {}  # corpus_id -> corpus_text
    relevant_docs = {} # query_id -> set([corpus_id])

    global_query_idx = 0

    for example in data:
        dialogue = example["dialogue"]
        candidates = {c["turn_id"]: c["text"] for c in example["candidate_turns"]}

        # Add all candidates to corpus for the evaluator
        for cid, text in candidates.items():
            corpus_key = f"{example['dialogue_id']}_{cid}"
            corpus[corpus_key] = build_cand_contexts(text)

        answers = example.get("answers", [])

        for ans in answers:
            gap_id = ans["gap_id"]
            gold_turn_id = ans["turn_id"]

            if gold_turn_id not in candidates:
                continue

            full_ctx = build_gap_contexts(dialogue, gap_id)
            gold_text = build_cand_contexts(candidates[gold_turn_id])

            # Create training example (Anchor, Positive)
            examples.append(InputExample(texts=[full_ctx, gold_text]))

            # Setup evaluator data
            q_id = f"q_{global_query_idx}"
            global_query_idx += 1
            queries[q_id] = full_ctx

            corpus_key = f"{example['dialogue_id']}_{gold_turn_id}"
            relevant_docs[q_id] = set([corpus_key])

    return examples, queries, corpus, relevant_docs

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

    # 1. Prepare Data
    print("Preparing data...")
    train_examples, _, _, _ = prepare_data(train_data)
    _, val_queries, val_corpus, val_relevant_docs = prepare_data(val_data)

    print(f"Total training pairs: {len(train_examples)}")
    print(f"Total validation queries: {len(val_queries)}")

    # Convert to HuggingFace Dataset format for newer sentence-transformers
    train_dataset = Dataset.from_dict({
        "anchor": [ex.texts[0] for ex in train_examples],
        "positive": [ex.texts[1] for ex in train_examples],
    })

    # 2. Load Model
    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)

    # 3. Setup Loss
    # MultipleNegativesRankingLoss treats all other items in the batch as negative examples
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
        fp16=True if torch.cuda.is_available() else False, # Enable fp16 if on GPU
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES, # Required for MNRL
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # 6. Train!
    print("Starting Fine-tuning...")
    # Using the newer trainer API
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

    # After training on Kaggle, the user can load this fine-tuned model in solution.py:
    # MODEL_NAME = 'bge_finetuned'

if __name__ == "__main__":
    main()
