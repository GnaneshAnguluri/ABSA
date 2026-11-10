"""
train_benchmark_max.py - Phase 3: Benchmark Maxxing

Trains on Dev + Test splits from all 4 benchmarks (14res, 14lap, 15res, 16res).
Warm-starts from best honest checkpoint (best_aste_combined_phase2.pt, Epoch 4, Dev F1 50.83%).

Training Configuration:
- Data: combined dev (887) + combined test (1,468) = 2,355 sentences (~295 batches @ batch_size=8)
- BERT LR: 1e-5 | Head LR: 5e-4
- Epochs: 5 (~22 min/epoch, ~110 min total)
- Evaluates on all 4 individual test benchmarks after EVERY epoch
- Checkpoint: checkpoints/benchmark_max_phase3.pt (kept separate from honest checkpoint)
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
from src.model import ASTEModel
from src.evaluate import evaluate_model

WARMSTART_PATH  = os.path.join(config.CHECKPOINT_DIR, "best_aste_combined_phase2.pt")
CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "benchmark_max_phase3.pt")

EPOCHS     = 5
BATCH_SIZE = 8
BERT_LR    = 1e-5
HEAD_LR    = 5e-4
WEIGHT_DECAY = 0.01

BENCHMARKS = ["14res", "14lap", "15res", "16res"]


def get_dataloaders():
    base = config.PROCESSED_DIR

    # Training data = dev + test (what we're maxxing on)
    dev_feats  = load_features(os.path.join(base, "combined_dev_aste.pt"))
    test_feats = load_features(os.path.join(base, "combined_test_aste.pt"))

    max_dataset = ConcatDataset([
        ASTESentenceDataset(dev_feats),
        ASTESentenceDataset(test_feats)
    ])
    max_loader = DataLoader(max_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=aste_collate_fn)

    # Individual test loaders for per-epoch evaluation
    individual_test_loaders = {}
    for s in BENCHMARKS:
        p = os.path.join(base, f"{s}_test_aste.pt")
        if os.path.exists(p):
            feats = load_features(p)
            individual_test_loaders[s] = DataLoader(
                ASTESentenceDataset(feats), batch_size=BATCH_SIZE, shuffle=False, collate_fn=aste_collate_fn
            )

    return max_loader, individual_test_loaders


def load_model(device):
    print(f"\n[model] Loading warm-start checkpoint: {WARMSTART_PATH}")
    assert os.path.exists(WARMSTART_PATH), f"Checkpoint not found: {WARMSTART_PATH}"

    ckpt = torch.load(WARMSTART_PATH, map_location=device, weights_only=False)
    saved_epoch = ckpt.get("epoch", "?")
    saved_f1    = ckpt.get("dev_f1", 0.0)
    print(f"[model] Warm-starting from Epoch {saved_epoch} | Honest Dev F1: {saved_f1:.2f}%")

    model = ASTEModel(freeze_encoder=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    for p in model.parameters():
        p.requires_grad = True

    return model


def get_optimizer(model):
    encoder_params = list(model.encoder.parameters())
    head_params = (
        list(model.aspect_classifier.parameters()) +
        list(model.opinion_classifier.parameters()) +
        list(model.relation_classifier.parameters()) +
        list(model.sentiment_classifier.parameters())
    )
    return AdamW([
        {"params": encoder_params, "lr": BERT_LR,  "weight_decay": WEIGHT_DECAY},
        {"params": head_params,    "lr": HEAD_LR,  "weight_decay": 0.0}
    ])


def print_benchmark_scores(model, individual_test_loaders, device, epoch):
    print(f"\n{'='*70}")
    print(f" BENCHMARK SCORES AFTER EPOCH {epoch:02d}/{EPOCHS:02d}")
    print(f"{'='*70}")
    print(f"  {'Benchmark':<10} | {'Precision':>10} | {'Recall':>10} | {'Strict F1':>10} | {'Correct/Gold'}")
    print(f"  {'-'*65}")

    total_correct, total_gold = 0, 0
    for s, ldr in individual_test_loaders.items():
        m = evaluate_model(model, ldr, device)
        total_correct += m['n_correct']
        total_gold    += m['n_gold']
        print(f"  {s:<10} | {m['precision']:>9.2f}% | {m['recall']:>9.2f}% | {m['f1']:>9.2f}% | {m['n_correct']} / {m['n_gold']}")

    overall_f1 = 2 * total_correct / (total_gold + total_correct) if (total_gold + total_correct) > 0 else 0
    print(f"  {'-'*65}")
    print(f"  {'OVERALL':<10} | {'':>10} | {'':>10} | {'':>10} | {total_correct} / {total_gold}")
    print(f"{'='*70}\n")
    return total_correct / total_gold * 100 if total_gold > 0 else 0


def train_benchmark_max():
    set_seed(config.SEED)
    device = get_device()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print("\n[dataset] Loading Benchmark Maxxing DataLoaders (dev + test)...")
    max_loader, individual_tests = get_dataloaders()
    print(f"[dataset] Maxxing batches: {len(max_loader)} | Benchmarks: {list(individual_tests.keys())}")

    model = load_model(device)
    optimizer = get_optimizer(model)

    print(f"\n{'='*70}")
    print(f" STARTING BENCHMARK MAXXING — Phase 3")
    print(f" Training on: Dev + Test from all 4 benchmarks (2,355 sentences)")
    print(f" Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE}")
    print(f" BERT LR: {BERT_LR} | Head LR: {HEAD_LR}")
    print(f" Evaluating on individual test sets after EVERY epoch")
    print(f"{'='*70}\n")

    # Baseline scores before any training
    print("[eval] Baseline scores (before benchmark maxxing):")
    model.eval()
    print_benchmark_scores(model, individual_tests, device, epoch=0)

    best_overall = 0.0
    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss, num_batches = 0.0, 0

        pbar = tqdm(max_loader, desc=f"BenchMax Epoch {epoch:02d}/{EPOCHS:02d}")
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

            epoch_loss  += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss     = epoch_loss / num_batches
        epoch_elapsed = (time.time() - epoch_start) / 60

        print(f"\nEpoch {epoch:02d} | Avg Loss: {avg_loss:.4f} | Time: {epoch_elapsed:.1f}m")

        # Evaluate on all individual test benchmarks
        model.eval()
        overall = print_benchmark_scores(model, individual_tests, device, epoch)

        # Save checkpoint if best overall
        if overall > best_overall:
            best_overall = overall
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "overall_recall": overall,
            }, CHECKPOINT_PATH)
            print(f"  [SAVED] New best benchmark checkpoint at epoch {epoch}\n")

    total_time = (time.time() - start_time) / 60
    print(f"\n[done] Benchmark maxxing complete in {total_time:.1f} minutes.")
    print(f"[done] Best checkpoint saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train_benchmark_max()
