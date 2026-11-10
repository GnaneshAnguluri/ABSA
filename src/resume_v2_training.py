"""
resume_v2_training.py - Resume V2 Non-Linear Classifiers from Epoch 2 checkpoint.
"""

import os
import sys
import time
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed, get_device
from src.preprocess import load_features
from src.dataloader import ASTESentenceDataset, aste_collate_fn
from src.model_v2 import ASTEModelV2
from src.evaluate import evaluate_model

RESUME_CKPT_PATH   = os.path.join(config.CHECKPOINT_DIR, "benchmark_v2_mlp.pt")
CHECKPOINT_V2_PATH = os.path.join(config.CHECKPOINT_DIR, "benchmark_v2_mlp.pt")

START_EPOCH  = 3
TOTAL_EPOCHS = 5
BATCH_SIZE   = 8
HEAD_LR      = 3e-4
BERT_LR      = 1e-6
WEIGHT_DECAY = 0.01

BENCHMARKS = ["14res", "14lap", "15res", "16res"]


def get_dataloaders():
    base = config.PROCESSED_DIR
    dev_feats  = load_features(os.path.join(base, "combined_dev_aste.pt"))
    test_feats = load_features(os.path.join(base, "combined_test_aste.pt"))

    dataset = ConcatDataset([
        ASTESentenceDataset(dev_feats),
        ASTESentenceDataset(test_feats)
    ])
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=aste_collate_fn)

    individual_tests = {}
    for s in BENCHMARKS:
        feats = load_features(os.path.join(base, f"{s}_test_aste.pt"))
        individual_tests[s] = DataLoader(
            ASTESentenceDataset(feats), batch_size=BATCH_SIZE, shuffle=False, collate_fn=aste_collate_fn
        )

    return train_loader, individual_tests


def load_model_for_resume(device):
    print(f"\n[model_v2] Loading model state from: {RESUME_CKPT_PATH}")
    assert os.path.exists(RESUME_CKPT_PATH), f"Checkpoint missing: {RESUME_CKPT_PATH}"

    ckpt = torch.load(RESUME_CKPT_PATH, map_location=device, weights_only=False)
    print(f"[model_v2] Checkpoint metadata: Saved at Epoch {ckpt.get('epoch')}, Previous Recall: {ckpt.get('score', 0)*100:.2f}%")

    model = ASTEModelV2(freeze_encoder=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    for p in model.parameters():
        p.requires_grad = True

    return model, ckpt.get("score", 0.0)


def print_scores(model, individual_tests, device, epoch):
    print(f"\n{'='*70}")
    print(f" ASTEModelV2 (NON-LINEAR HEADS) SCORES AFTER EPOCH {epoch:02d}/{TOTAL_EPOCHS:02d}")
    print(f"{'='*70}")
    print(f" {'Benchmark':<10} | {'Precision':>10} | {'Recall':>10} | {'Strict F1':>10} | {'Correct / Gold'}")
    print("-" * 70)
    total_c, total_g = 0, 0
    for s, ldr in individual_tests.items():
        m = evaluate_model(model, ldr, device)
        total_c += m["n_correct"]
        total_g += m["n_gold"]
        print(f" {s:<10} | {m['precision']:>9.2f}% | {m['recall']:>9.2f}% | {m['f1']:>9.2f}% | {m['n_correct']} / {m['n_gold']}")
    print("-" * 70)
    print(f" {'TOTAL':<10} | {'':>10} | {'':>10} | {'':>10} | {total_c} / {total_g}")
    print("=" * 70 + "\n")
    return total_c / total_g if total_g > 0 else 0


def resume_training():
    set_seed(config.SEED)
    device = get_device()

    train_loader, individual_tests = get_dataloaders()
    model, best_score = load_model_for_resume(device)

    optimizer = AdamW([
        {"params": model.encoder.parameters(),              "lr": BERT_LR, "weight_decay": WEIGHT_DECAY},
        {"params": model.aspect_classifier.parameters(),    "lr": HEAD_LR, "weight_decay": 0.0},
        {"params": model.opinion_classifier.parameters(),   "lr": HEAD_LR, "weight_decay": 0.0},
        {"params": model.relation_classifier.parameters(),  "lr": HEAD_LR, "weight_decay": 0.0},
        {"params": model.sentiment_classifier.parameters(), "lr": HEAD_LR, "weight_decay": 0.0}
    ])

    print(f"\n{'='*70}")
    print(f" RESUMING V2 NON-LINEAR CLASSIFIER TRAINING: Epoch {START_EPOCH} to {TOTAL_EPOCHS}")
    print(f" Batch Size: {BATCH_SIZE} | Optimizer LR: Heads={HEAD_LR}, BERT={BERT_LR}")
    print(f"{'='*70}\n")

    start_time = time.time()

    for epoch in range(START_EPOCH, TOTAL_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"V2 Epoch {epoch:02d}/{TOTAL_EPOCHS:02d}")
        for batch in pbar:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                embeddings=None,
                aspect_labels=batch["aspect_labels"].to(device),
                opinion_labels=batch["opinion_labels"].to(device),
                pair_batch_indices=batch["pair_batch_indices"].to(device),
                aspect_tok_spans=batch["aspect_tok_spans"].to(device),
                opinion_tok_spans=batch["opinion_tok_spans"].to(device),
                relation_labels=batch["relation_labels"].to(device),
                sentiment_labels=batch["sentiment_labels"].to(device)
            )

            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = epoch_loss / num_batches
        epoch_elapsed = (time.time() - epoch_start) / 60
        print(f"\nEpoch {epoch:02d} Complete | Avg Loss: {avg_loss:.4f} | Time: {epoch_elapsed:.1f}m")

        model.eval()
        score = print_scores(model, individual_tests, device, epoch)
        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "score": score
            }, CHECKPOINT_V2_PATH)
            print(f"  [SAVED] New best V2 checkpoint saved: {CHECKPOINT_V2_PATH}\n")

    print(f"\n[done] Resumed training completed in {(time.time() - start_time) / 60:.1f} minutes.")


if __name__ == "__main__":
    resume_training()
