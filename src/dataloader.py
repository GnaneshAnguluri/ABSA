"""
dataloader.py - Step 4: ASTE Sentence-Level Dataset & Collate Function.

Handles:
- Standard sequence inputs: input_ids, attention_mask, aspect_labels, opinion_labels [B, max_len]
- Variable-length candidate pairs via custom collate_fn:
  - pair_batch_indices: [num_pairs] tensor indicating which sentence in the batch this pair belongs to
  - aspect_tok_spans:   [num_pairs, 2] tensor (start_tok, end_tok)
  - opinion_tok_spans:  [num_pairs, 2] tensor (start_tok, end_tok)
  - relation_labels:    [num_pairs] tensor (0=INVALID, 1=VALID)
  - sentiment_labels:   [num_pairs] tensor (0=NEG, 1=NEU, 2=POS, -100=IGNORE)
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocess import load_features


class ASTESentenceDataset(Dataset):
    """Wraps preprocessed sentence-level ASTE feature dictionaries."""

    def __init__(self, features: list):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


def aste_collate_fn(batch):
    """
    Collate function that dynamically aggregates variable numbers of candidate pairs per batch.
    """
    input_ids      = torch.stack([f["input_ids"] for f in batch])
    attention_mask = torch.stack([f["attention_mask"] for f in batch])
    aspect_labels  = torch.stack([f["aspect_labels"] for f in batch])
    opinion_labels = torch.stack([f["opinion_labels"] for f in batch])

    pair_batch_indices = []
    aspect_tok_spans   = []
    opinion_tok_spans  = []
    relation_labels    = []
    sentiment_labels   = []
    pair_metadata      = []

    for b_idx, f in enumerate(batch):
        for cp in f["candidate_pairs"]:
            pair_batch_indices.append(b_idx)
            aspect_tok_spans.append(cp["aspect_tok_span"])
            opinion_tok_spans.append(cp["opinion_tok_span"])
            relation_labels.append(cp["relation_label"])
            sentiment_labels.append(cp["sentiment_label"])
            pair_metadata.append({
                "batch_idx"        : b_idx,
                "sentence"         : f["sentence"],
                "aspect_word_span" : cp["aspect_word_span"],
                "opinion_word_span": cp["opinion_word_span"],
            })

    if len(pair_batch_indices) > 0:
        pair_batch_indices = torch.tensor(pair_batch_indices, dtype=torch.long)
        aspect_tok_spans   = torch.tensor(aspect_tok_spans, dtype=torch.long)
        opinion_tok_spans  = torch.tensor(opinion_tok_spans, dtype=torch.long)
        relation_labels    = torch.tensor(relation_labels, dtype=torch.long)
        sentiment_labels   = torch.tensor(sentiment_labels, dtype=torch.long)
    else:
        pair_batch_indices = torch.empty(0, dtype=torch.long)
        aspect_tok_spans   = torch.empty((0, 2), dtype=torch.long)
        opinion_tok_spans  = torch.empty((0, 2), dtype=torch.long)
        relation_labels    = torch.empty(0, dtype=torch.long)
        sentiment_labels   = torch.empty(0, dtype=torch.long)

    gold_triples = [f["gold_triples"] for f in batch]
    raw_sentences = [f["sentence"] for f in batch]
    raw_words     = [f["words"] for f in batch]
    raw_word_ids  = [f["word_ids"] for f in batch]

    return {
        "input_ids"         : input_ids,           # [B, max_len]
        "attention_mask"    : attention_mask,      # [B, max_len]
        "aspect_labels"     : aspect_labels,       # [B, max_len]
        "opinion_labels"    : opinion_labels,      # [B, max_len]
        "pair_batch_indices": pair_batch_indices,  # [num_pairs]
        "aspect_tok_spans"  : aspect_tok_spans,    # [num_pairs, 2]
        "opinion_tok_spans" : opinion_tok_spans,   # [num_pairs, 2]
        "relation_labels"   : relation_labels,     # [num_pairs]
        "sentiment_labels"  : sentiment_labels,    # [num_pairs]
        "gold_triples"      : gold_triples,        # list of length B
        "sentences"         : raw_sentences,
        "words"             : raw_words,
        "word_ids"          : raw_word_ids,
        "pair_metadata"     : pair_metadata
    }


def get_dataloaders(dataset_split: str = config.DATASET_SPLIT,
                    batch_size:    int  = config.BATCH_SIZE,
                    num_workers:   int  = 0):
    """Returns train, dev, and test DataLoader objects."""
    base = config.PROCESSED_DIR

    train_feats = load_features(os.path.join(base, f"{dataset_split}_train_aste.pt"))
    dev_feats   = load_features(os.path.join(base, f"{dataset_split}_dev_aste.pt"))
    test_feats  = load_features(os.path.join(base, f"{dataset_split}_test_aste.pt"))

    train_ds = ASTESentenceDataset(train_feats)
    dev_ds   = ASTESentenceDataset(dev_feats)
    test_ds  = ASTESentenceDataset(test_feats)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=aste_collate_fn
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=aste_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=aste_collate_fn
    )

    print(f"\n[dataloader] Split: {dataset_split} | Batch size: {batch_size}")
    print(f"[dataloader] Train sentences: {len(train_ds)} ({len(train_loader)} batches)")
    print(f"[dataloader] Dev sentences:   {len(dev_ds)} ({len(dev_loader)} batches)")
    print(f"[dataloader] Test sentences:  {len(test_ds)} ({len(test_loader)} batches)")

    return train_loader, dev_loader, test_loader
