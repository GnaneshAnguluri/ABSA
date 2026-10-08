"""
train.py - Step 9: Multi-Task Training Loop for ASTE (Phase 1: Frozen BERT).
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


def train():
    set_seed(config.SEED)
    device = get_device()
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    if config.USE_CACHED_EMBEDDINGS:
        from src.cache_embeddings import build_all_embedding_caches
        build_all_embedding_caches()

    print("\n[train] Loading ASTE DataLoaders...")
    train_loader, dev_loader, test_loader = get_dataloaders()

    print(f"\n[train] Initializing ASTE Model (Encoder: {config.ENCODER_MODEL}, Frozen: {config.FREEZE_ENCODER})...")
    model = ASTEModel().to(device)

    # Filter trainable parameters (only ASTE heads are trained in Phase 1)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"[train] Total Parameters:     {total_params:,}")
    print(f"[train] Trainable Parameters: {num_trainable:,} ({100*num_trainable/total_params:.2f}%)")

    optimizer = AdamW(trainable_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_dev_f1 = 0.0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_aste_model.pt")

    print(f"\n{'='*75}")
    print(f" Starting ASTE Training: {config.EPOCHS} Epochs | Batch Size: {config.BATCH_SIZE}")
    print(f"{'='*75}\n")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{config.EPOCHS:02d}")
        for batch in pbar:
            optimizer.zero_grad()

            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
            embeddings      = batch["embeddings"].to(device) if batch["embeddings"] is not None else None
            aspect_labels   = batch["aspect_labels"].to(device)
            opinion_labels  = batch["opinion_labels"].to(device)

            p_indices = batch["pair_batch_indices"].to(device)
            a_spans   = batch["aspect_tok_spans"].to(device)
            o_spans   = batch["opinion_tok_spans"].to(device)
            r_labels  = batch["relation_labels"].to(device)
            s_labels  = batch["sentiment_labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                embeddings=embeddings,
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
            torch.nn.utils.clip_grad_norm_(trainable_params, config.MAX_GRAD_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = epoch_loss / num_batches

        # Evaluate on Dev set
        dev_metrics = evaluate_model(model, dev_loader, device)

        print(f"\nEpoch {epoch:02d} Summary:")
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
                    "frozen"  : config.FREEZE_ENCODER,
                }
            }, checkpoint_path)
            print(f"  >>> Best model saved! (Dev Triplet F1: {best_dev_f1:.2f}%)")

        print("-" * 75)

    # Final evaluation on Test Set using best saved checkpoint
    print("\n[train] Loading best model for final evaluation on TEST set...")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    test_metrics = evaluate_model(model, test_loader, device)
    print(f"\n{'='*75}")
    print(f" FINAL TEST EVALUATION (Phase 1: Frozen BERT)")
    print(f"{'='*75}")
    print(f"  Strict Triplet Precision : {test_metrics['precision']:.2f}%")
    print(f"  Strict Triplet Recall    : {test_metrics['recall']:.2f}%")
    print(f"  Strict Triplet F1        : {test_metrics['f1']:.2f}%")
    print(f"  Correct Triplets         : {test_metrics['n_correct']} / {test_metrics['n_gold']}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    train()
