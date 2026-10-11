"""
train_phase2.py - Phase 2: End-to-End BERT Fine-Tuning for ASTE.

Key Properties:
1. Unfreezes BERT (requires_grad=True).
2. Dynamic computation: Bypasses cached embeddings (use_cached=False) so gradients flow through BERT.
3. Warm-starts classification heads from Phase 1 checkpoint (checkpoints/best_aste_model.pt).
4. Differential Learning Rates:
   - BERT Encoder: 2e-5
   - ASTE Classification Heads: 1e-3
5. Evaluates strict triplet F1 after each epoch on dev set.
6. Saves best checkpoint to checkpoints/best_aste_phase2.pt (does not overwrite Phase 1).
"""

import os
import sys
import torch
from torch.optim import AdamW
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed, get_device
from src.dataloader import get_dataloaders
from src.model import ASTEModel
from src.evaluate import evaluate_model

# Phase 2 Hyperparameters
PHASE2_EPOCHS         = 5
PHASE2_BATCH_SIZE     = 8
PHASE2_BERT_LR        = 2e-5
PHASE2_HEAD_LR        = 1e-3
PHASE2_WEIGHT_DECAY   = 0.01
PHASE2_CHECKPOINT_PATH= os.path.join(config.CHECKPOINT_DIR, "best_aste_phase2.pt")
PHASE1_CHECKPOINT_PATH= os.path.join(config.CHECKPOINT_DIR, "best_aste_model.pt")


def initialize_phase2_model(device):
    """
    Instantiates ASTEModel with trainable BERT (freeze_encoder=False),
    and warm-starts classification heads from Phase 1 checkpoint if compatible.
    """
    print(f"\n[phase2] Initializing model with trainable encoder: {config.ENCODER_MODEL}...")
    model = ASTEModel(freeze_encoder=False).to(device)

    # Warm-start heads from Phase 1 checkpoint if available
    if os.path.exists(PHASE1_CHECKPOINT_PATH):
        print(f"[phase2] Found Phase 1 checkpoint at: {PHASE1_CHECKPOINT_PATH}")
        ckpt = torch.load(PHASE1_CHECKPOINT_PATH, map_location=device, weights_only=False)
        phase1_state = ckpt.get("model_state", {})

        model_state = model.state_dict()
        loaded_heads = []
        for key in phase1_state:
            # Load only the task-specific classification head weights
            if not key.startswith("encoder.") and key in model_state:
                if model_state[key].shape == phase1_state[key].shape:
                    model_state[key].copy_(phase1_state[key])
                    loaded_heads.append(key)
                else:
                    print(f"[phase2] WARNING: Shape mismatch for {key}, skipping warm-start for this parameter.")

        print(f"[phase2] Warm-started {len(loaded_heads)} classification head tensors from Phase 1:")
        for h in loaded_heads:
            print(f"  • {h}")
    else:
        print("[phase2] No Phase 1 checkpoint found. Initializing heads from scratch.")

    # Ensure all BERT parameters require grad
    for param in model.encoder.parameters():
        param.requires_grad = True

    return model


def get_phase2_optimizer(model):
    """
    Constructs differential parameter groups:
    - BERT encoder: 2e-5
    - Classification heads: 1e-3
    """
    encoder_params = list(model.encoder.parameters())
    head_params = (
        list(model.aspect_classifier.parameters()) +
        list(model.opinion_classifier.parameters()) +
        list(model.relation_classifier.parameters()) +
        list(model.sentiment_classifier.parameters())
    )

    optimizer = AdamW([
        {"params": encoder_params, "lr": PHASE2_BERT_LR, "weight_decay": PHASE2_WEIGHT_DECAY},
        {"params": head_params,    "lr": PHASE2_HEAD_LR, "weight_decay": 0.0}
    ])

    return optimizer


def train_phase2():
    set_seed(config.SEED)
    device = get_device()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Phase 2 uses dynamic BERT passes (use_cached=False)
    print("\n[phase2] Loading ASTE DataLoaders (Dynamic BERT pass, use_cached=False)...")
    train_loader, dev_loader, test_loader = get_dataloaders(
        dataset_split=config.DATASET_SPLIT,
        batch_size=PHASE2_BATCH_SIZE,
        use_cached=False
    )

    model = initialize_phase2_model(device)

    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    head_params = sum(p.numel() for p in model.parameters() if p.requires_grad and not any(p is ep for ep in model.encoder.parameters()))

    print(f"\n[phase2] Parameter Breakdown:")
    print(f"  • Total Parameters:          {total_params:,}")
    print(f"  • Trainable BERT Parameters: {encoder_params:,} (LR: {PHASE2_BERT_LR})")
    print(f"  • Trainable Head Parameters: {head_params:,} (LR: {PHASE2_HEAD_LR})")

    optimizer = get_phase2_optimizer(model)
    best_dev_f1 = 0.0

    print(f"\n{'='*75}")
    print(f" Starting Phase 2 End-to-End Fine-Tuning: {PHASE2_EPOCHS} Epochs | Batch Size: {PHASE2_BATCH_SIZE}")
    print(f"{'='*75}\n")

    for epoch in range(1, PHASE2_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Phase 2 Epoch {epoch:02d}/{PHASE2_EPOCHS:02d}")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            aspect_labels   = batch["aspect_labels"].to(device)
            opinion_labels  = batch["opinion_labels"].to(device)

            p_indices = batch["pair_batch_indices"].to(device)
            a_spans   = batch["aspect_tok_spans"].to(device)
            o_spans   = batch["opinion_tok_spans"].to(device)
            r_labels  = batch["relation_labels"].to(device)
            s_labels  = batch["sentiment_labels"].to(device)

            # Note: embeddings is None, triggers dynamic BERT forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                embeddings=None,
                aspect_labels=aspect_labels,
                opinion_labels=opinion_labels,
                pair_batch_indices=p_indices,
                aspect_tok_spans=a_spans,
                opinion_tok_spans=o_spans,
                relation_labels=r_labels,
                sentiment_labels=s_labels
            )

            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / num_batches

        # Evaluate on Dev set using strict triplet extraction
        dev_metrics = evaluate_model(model, dev_loader, device)

        print(f"\nPhase 2 Epoch {epoch:02d} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Dev Loss: {dev_metrics['loss']:.4f}")
        print(f"  Dev Triplet Extraction: Precision={dev_metrics['precision']:.2f}% | "
              f"Recall={dev_metrics['recall']:.2f}% | F1={dev_metrics['f1']:.2f}% "
              f"({dev_metrics['n_correct']}/{dev_metrics['n_gold']} gold)")

        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "dev_f1"     : best_dev_f1,
                "config"     : {
                    "encoder" : config.ENCODER_MODEL,
                    "frozen"  : False,
                    "bert_lr" : PHASE2_BERT_LR,
                    "head_lr" : PHASE2_HEAD_LR
                }
            }, PHASE2_CHECKPOINT_PATH)
            print(f"  >>> Best Phase 2 model saved! (Dev Triplet F1: {best_dev_f1:.2f}%)")

        print("-" * 75)

    # Final evaluation on Test Set using best saved Phase 2 checkpoint
    print("\n[phase2] Loading best Phase 2 model for final evaluation on TEST set...")
    ckpt = torch.load(PHASE2_CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate_model(model, test_loader, device)
    print(f"\n{'='*75}")
    print(f" FINAL TEST EVALUATION (Phase 2: Fine-Tuned BERT)")
    print(f"{'='*75}")
    print(f"  Strict Triplet Precision : {test_metrics['precision']:.2f}%")
    print(f"  Strict Triplet Recall    : {test_metrics['recall']:.2f}%")
    print(f"  Strict Triplet F1        : {test_metrics['f1']:.2f}%")
    print(f"  Correct Triplets         : {test_metrics['n_correct']} / {test_metrics['n_gold']}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    train_phase2()
