"""
train_combined_phase2.py - Phase 2 End-to-End Fine-Tuning on Combined ASTE Benchmarks.

Datasets Combined:
- 14res + 14lap + 15res + 16res
- Train: 3,634 sentences (455 batches @ batch size 8)
- Dev:   887 sentences
- Test:  1,468 sentences (plus individual benchmarks: 14res, 14lap, 15res, 16res)

Training Configuration:
- Encoder: bert-base-uncased (unfrozen, requires_grad=True)
- Heads: Aspect, Opinion, Relation, Sentiment (warm-started from Phase 2 / Phase 1 checkpoint)
- Optimizer: AdamW
  - BERT encoder LR: 2e-5 (weight decay 0.01)
  - ASTE heads LR: 1e-3 (weight decay 0.0)
- Batch size: 8
- Epochs: 5
- Metric: Dev Strict Triplet F1 (end-to-end predicted spans, no gold leakage)
- Best Checkpoint: checkpoints/best_aste_combined_phase2.pt
"""

import os
import sys
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed, get_device
from src.preprocess import load_features
from src.dataloader import ASTESentenceDataset, aste_collate_fn
from src.model import ASTEModel
from src.evaluate import evaluate_model

CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "best_aste_combined_phase2.pt")
WARMSTART_CANDIDATES = [
    os.path.join(config.CHECKPOINT_DIR, "best_aste_phase2.pt"),
    os.path.join(config.CHECKPOINT_DIR, "best_aste_model.pt")
]

EPOCHS     = 5
BATCH_SIZE = 8
BERT_LR    = 2e-5
HEAD_LR    = 1e-3
WEIGHT_DECAY = 0.01


def get_combined_dataloaders():
    """Load combined train/dev/test datasets and individual test sets."""
    base = config.PROCESSED_DIR
    train_feats = load_features(os.path.join(base, "combined_train_aste.pt"))
    dev_feats   = load_features(os.path.join(base, "combined_dev_aste.pt"))
    test_feats  = load_features(os.path.join(base, "combined_test_aste.pt"))

    train_loader = DataLoader(ASTESentenceDataset(train_feats), batch_size=BATCH_SIZE, shuffle=True, collate_fn=aste_collate_fn)
    dev_loader   = DataLoader(ASTESentenceDataset(dev_feats),   batch_size=BATCH_SIZE, shuffle=False, collate_fn=aste_collate_fn)
    test_loader  = DataLoader(ASTESentenceDataset(test_feats),  batch_size=BATCH_SIZE, shuffle=False, collate_fn=aste_collate_fn)

    individual_test_loaders = {}
    for s in ["14res", "14lap", "15res", "16res"]:
        p = os.path.join(base, f"{s}_test_aste.pt")
        if os.path.exists(p):
            feats = load_features(p)
            individual_test_loaders[s] = DataLoader(ASTESentenceDataset(feats), batch_size=BATCH_SIZE, shuffle=False, collate_fn=aste_collate_fn)

    return train_loader, dev_loader, test_loader, individual_test_loaders


def initialize_model(device):
    """Instantiate ASTEModel with trainable BERT and warm-start heads."""
    print(f"\n[model] Initializing ASTEModel with trainable encoder: {config.ENCODER_MODEL}...")
    model = ASTEModel(freeze_encoder=False).to(device)

    # Warm-start heads from existing checkpoint if found
    for ckpt_path in WARMSTART_CANDIDATES:
        if os.path.exists(ckpt_path):
            print(f"[model] Warm-starting heads from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            state = ckpt.get("model_state", ckpt.get("model_state_dict", {}))
            model_state = model.state_dict()
            loaded = []
            for k in state:
                if not k.startswith("encoder.") and k in model_state:
                    if model_state[k].shape == state[k].shape:
                        model_state[k].copy_(state[k])
                        loaded.append(k)
            print(f"[model] Successfully loaded {len(loaded)} head parameter tensors.")
            break
    else:
        print("[model] Initializing all parameters from scratch.")

    for p in model.encoder.parameters():
        p.requires_grad = True

    return model


def get_optimizer(model):
    """Differential learning rates for BERT encoder and classification heads."""
    encoder_params = list(model.encoder.parameters())
    head_params = (
        list(model.aspect_classifier.parameters()) +
        list(model.opinion_classifier.parameters()) +
        list(model.relation_classifier.parameters()) +
        list(model.sentiment_classifier.parameters())
    )
    return AdamW([
        {"params": encoder_params, "lr": BERT_LR, "weight_decay": WEIGHT_DECAY},
        {"params": head_params,    "lr": HEAD_LR, "weight_decay": 0.0}
    ])


def run_smoke_test(model, train_loader, device):
    """Smoke test 1 batch to ensure dynamic BERT gradients and optimizer updates work."""
    print("\n" + "=" * 65)
    print(" SMOKE TEST: 1-Batch Forward/Backward Verification")
    print("=" * 65)

    batch = next(iter(train_loader))
    model.train()

    optimizer = get_optimizer(model)
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
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN/Inf!"
    loss.backward()

    # Check encoder gradients
    enc_grads = [p.grad for name, p in model.encoder.named_parameters() if not name.startswith("pooler.") and p.grad is not None]
    assert len(enc_grads) > 0, "BERT parameters did not receive gradients!"
    print(f"  [OK] BERT Encoder receives active gradients ({len(enc_grads)} tensors)")

    # Check head gradients
    for head_name, head in [("Aspect", model.aspect_classifier), ("Opinion", model.opinion_classifier),
                            ("Relation", model.relation_classifier), ("Sentiment", model.sentiment_classifier)]:
        h_grads = [p.grad for p in head.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        assert len(h_grads) > 0, f"{head_name} head did not receive gradients!"
        print(f"  [OK] {head_name} head receives active gradients")

    optimizer.step()
    print("  >>> SMOKE TEST PASSED! BERT is fully trainable and heads are active.\n")


def train_combined():
    set_seed(config.SEED)
    device = get_device()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    print("\n[dataset] Loading Combined ASTE DataLoaders...")
    train_loader, dev_loader, test_loader, individual_tests = get_combined_dataloaders()
    print(f"[dataset] Train batches: {len(train_loader)} | Dev batches: {len(dev_loader)} | Test batches: {len(test_loader)}")

    model = initialize_model(device)
    run_smoke_test(model, train_loader, device)

    optimizer = get_optimizer(model)
    best_dev_f1 = 0.0
    best_epoch = 0

    print(f"\n{'='*75}")
    print(f" STARTING PHASE 2 FINE-TUNING ON COMBINED DATA")
    print(f" Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Train Sentences: 3,634")
    print(f" BERT LR: {BERT_LR} | Head LR: {HEAD_LR}")
    print(f"{'='*75}\n")

    start_train_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Combined Epoch {epoch:02d}/{EPOCHS:02d}")
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

        avg_train_loss = epoch_loss / num_batches
        epoch_elapsed = time.time() - epoch_start

        # Evaluate on Combined Dev set
        print(f"\n[eval] Evaluating on Combined Dev Set ({len(dev_loader)} batches)...")
        dev_metrics = evaluate_model(model, dev_loader, device)
        dev_f1 = dev_metrics["f1"]

        is_best = dev_f1 > best_dev_f1
        if is_best:
            best_dev_f1 = dev_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "dev_f1": best_dev_f1,
                "dev_metrics": dev_metrics,
                "dataset": "combined_aste"
            }, CHECKPOINT_PATH)

        print(f"\nCombined Epoch {epoch:02d} Summary (Time: {epoch_elapsed/60:.1f}m):")
        print(f"  Train Loss : {avg_train_loss:.4f} | Dev Loss: {dev_metrics['loss']:.4f}")
        print(f"  Dev Triplet: Precision={dev_metrics['precision']:.2f}% | Recall={dev_metrics['recall']:.2f}% | F1={dev_f1:.2f}% "
              f"({dev_metrics['n_correct']}/{dev_metrics['n_gold']}) {'[BEST SAVED]' if is_best else ''}")
        print("-" * 75)

    total_train_time = time.time() - start_train_time
    print(f"\n[train] Training completed in {total_train_time/60:.1f} minutes.")

    # Final Evaluations with best checkpoint
    print(f"\n[eval] Loading best model (Epoch {best_epoch}, Dev F1: {best_dev_f1:.2f}%) for Test Evaluation...")
    best_ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()

    # 1. Combined Test Evaluation
    print("\n" + "=" * 75)
    print(" EVALUATION ON COMBINED TEST SET (1,468 sentences)")
    print("=" * 75)
    comb_test_metrics = evaluate_model(model, test_loader, device)
    print(f"  Strict Precision : {comb_test_metrics['precision']:.2f}%")
    print(f"  Strict Recall    : {comb_test_metrics['recall']:.2f}%")
    print(f"  Strict Triplet F1: {comb_test_metrics['f1']:.2f}%")
    print(f"  Correct Triplets : {comb_test_metrics['n_correct']} / {comb_test_metrics['n_gold']}")

    # 2. Individual Benchmark Breakdowns
    print("\n" + "=" * 75)
    print(" BREAKDOWN ACROSS INDIVIDUAL TEST BENCHMARKS")
    print("=" * 75)
    header = f"{'Benchmark':<12} | {'Precision':<10} | {'Recall':<10} | {'Strict F1':<10} | {'Correct / Gold':<15}"
    print(header)
    print("-" * 75)

    for s, ldr in individual_tests.items():
        m = evaluate_model(model, ldr, device)
        print(f"{s:<12} | {m['precision']:>8.2f}% | {m['recall']:>8.2f}% | {m['f1']:>8.2f}% | {m['n_correct']:>4} / {m['n_gold']:<4}")

    print("=" * 75 + "\n")


if __name__ == "__main__":
    train_combined()
