"""
dataloader.py - Step 4: PyTorch Dataset & DataLoader for the ABSA pipeline.

Wraps the pre-processed feature dicts into a torch Dataset
and returns batched DataLoaders for train / dev / test.
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocess import load_features


# ─────────────────────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────────────────────

class ABSADataset(Dataset):
    """
    Wraps a list of pre-processed feature dicts into a PyTorch Dataset.

    Each __getitem__ returns a dict of tensors:
        input_ids       : [max_len]   - BERT token IDs
        attention_mask  : [max_len]   - 1=real token, 0=pad
        aspect_labels   : [max_len]   - BIO tag IDs  (IGNORE_INDEX for CLS/SEP/PAD/subwords)
        opinion_labels  : [max_len]   - BIO tag IDs
        sentiment_label : []          - scalar int  (NEG=0, NEU=1, POS=2)
    """

    def __init__(self, features: list):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        return {
            "input_ids"      : feat["input_ids"],                              # [128]
            "attention_mask" : feat["attention_mask"],                         # [128]
            "aspect_labels"  : feat["aspect_labels"],                          # [128]
            "opinion_labels" : feat["opinion_labels"],                         # [128]
            "sentiment_label": torch.tensor(feat["sentiment_label"],
                                            dtype=torch.long),                 # scalar
        }


# ─────────────────────────────────────────────────────────────
#  DataLoader Factory
# ─────────────────────────────────────────────────────────────

def get_dataloaders(dataset_split: str = config.DATASET_SPLIT,
                    batch_size:    int  = config.BATCH_SIZE,
                    num_workers:   int  = 0):
    """
    Load pre-processed .pt files and return three DataLoaders.

    Args:
        dataset_split : one of '14res', '14lap', '15res', '16res'
        batch_size    : number of samples per batch
        num_workers   : parallel data loading workers
                        (keep 0 on Windows to avoid multiprocessing issues)

    Returns:
        train_loader, dev_loader, test_loader
    """
    base = config.PROCESSED_DIR

    # ── Load pre-processed features ──
    train_feats = load_features(os.path.join(base, f"{dataset_split}_train.pt"))
    dev_feats   = load_features(os.path.join(base, f"{dataset_split}_dev.pt"))
    test_feats  = load_features(os.path.join(base, f"{dataset_split}_test.pt"))

    # ── Wrap in Dataset ──
    train_ds = ABSADataset(train_feats)
    dev_ds   = ABSADataset(dev_feats)
    test_ds  = ABSADataset(test_feats)

    # ── Build DataLoaders ──
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,           # shuffle only training data
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"\n[dataloader] Split         : {dataset_split}")
    print(f"[dataloader] Batch size    : {batch_size}")
    print(f"[dataloader] Train batches : {len(train_loader)}  ({len(train_ds)} samples)")
    print(f"[dataloader] Dev batches   : {len(dev_loader)}  ({len(dev_ds)} samples)")
    print(f"[dataloader] Test batches  : {len(test_loader)}  ({len(test_ds)} samples)")

    return train_loader, dev_loader, test_loader


# ─────────────────────────────────────────────────────────────
#  DEBUG — Inspect one batch
# ─────────────────────────────────────────────────────────────

def inspect_batch(batch: dict):
    """Print shapes and a sample of values from one batch."""
    print(f"\n[dataloader] Batch contents:")
    print(f"  input_ids       : {batch['input_ids'].shape}     dtype={batch['input_ids'].dtype}")
    print(f"  attention_mask  : {batch['attention_mask'].shape}     dtype={batch['attention_mask'].dtype}")
    print(f"  aspect_labels   : {batch['aspect_labels'].shape}     dtype={batch['aspect_labels'].dtype}")
    print(f"  opinion_labels  : {batch['opinion_labels'].shape}     dtype={batch['opinion_labels'].dtype}")
    print(f"  sentiment_label : {batch['sentiment_label'].shape}     dtype={batch['sentiment_label'].dtype}")

    print(f"\n  First sample sentiment labels : {batch['sentiment_label'][:8].tolist()}")
    print(f"  First sample aspect tags      : {batch['aspect_labels'][0][:20].tolist()}  ...")
    print(f"  First sample opinion tags     : {batch['opinion_labels'][0][:20].tolist()}  ...")

    # Count non-ignored tokens in first sample
    mask      = batch['aspect_labels'][0] != -100
    real_toks = mask.sum().item()
    asp_toks  = (batch['aspect_labels'][0] > 0).sum().item()   # B-ASP or I-ASP
    opn_toks  = (batch['opinion_labels'][0] > 0).sum().item()  # B-OPN or I-OPN
    print(f"\n  Real (non-PAD) tokens in sample 0 : {real_toks}")
    print(f"  Aspect tokens (B/I-ASP)           : {asp_toks}")
    print(f"  Opinion tokens (B/I-OPN)          : {opn_toks}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.utils import set_seed
    set_seed(config.SEED)

    # Build loaders
    train_loader, dev_loader, test_loader = get_dataloaders()

    # Grab and inspect first batch
    batch = next(iter(train_loader))
    inspect_batch(batch)

    # Verify all batches load without error
    print(f"\n[dataloader] Running full train loader pass...")
    for i, b in enumerate(train_loader):
        assert b["input_ids"].shape      == (config.BATCH_SIZE, config.MAX_SEQ_LEN) or \
               b["input_ids"].shape[0]   <= config.BATCH_SIZE, "Shape mismatch!"
    print(f"[dataloader] All {len(train_loader)} train batches OK.")

    print("\n[dataloader] Step 4 complete!")
