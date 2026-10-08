"""
cache_embeddings.py - Precompute and cache frozen BERT embeddings for CPU acceleration.

Runs bert-base-uncased with torch.no_grad() and model.eval() ONCE per sample.
Converts embeddings to float16 to save RAM/disk.
Reuses existing cache if available.
"""

import os
import sys
import torch
from transformers import AutoModel
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocess import load_features, save_features


def get_cached_filepath(split_name: str, dataset_split: str = config.DATASET_SPLIT) -> str:
    """Returns the path to the cached embedding file."""
    return os.path.join(config.PROCESSED_DIR, f"{dataset_split}_{split_name}_cached.pt")


def cache_split_embeddings(split_name: str, encoder, batch_size: int = 8, force_recompute: bool = False):
    """
    Computes frozen BERT embeddings for a given split and saves to disk.
    """
    cache_path = get_cached_filepath(split_name)
    if os.path.exists(cache_path) and not force_recompute:
        print(f"[cache] Embeddings already cached for {split_name} -> {cache_path}")
        return cache_path

    raw_feat_path = os.path.join(config.PROCESSED_DIR, f"{config.DATASET_SPLIT}_{split_name}_aste.pt")
    if not os.path.exists(raw_feat_path):
        raise FileNotFoundError(f"Source features not found at {raw_feat_path}. Run src/preprocess.py first.")

    features = load_features(raw_feat_path)
    print(f"\n[cache] Computing BERT embeddings for {split_name} ({len(features)} sentences)...")

    cached_features = []
    n_samples = len(features)

    for i in tqdm(range(0, n_samples, batch_size), desc=f"Caching {split_name}"):
        batch_slice = features[i : i + batch_size]
        input_ids = torch.stack([f["input_ids"] for f in batch_slice])
        attention_mask = torch.stack([f["attention_mask"] for f in batch_slice])

        with torch.no_grad():
            outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
            # [B, 128, 768] -> float16
            last_hidden = outputs.last_hidden_state.to(torch.float16).cpu()

        for idx, f in enumerate(batch_slice):
            new_f = dict(f)
            new_f["embedding"] = last_hidden[idx]  # [128, 768] float16
            cached_features.append(new_f)

    save_features(cached_features, cache_path)
    file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"[cache] Saved {len(cached_features)} cached items -> {cache_path} ({file_size_mb:.1f} MB)")
    return cache_path


def build_all_embedding_caches(force_recompute: bool = False):
    """
    Precomputes and saves caches for train, dev, and test splits.
    """
    splits = ["train", "dev", "test"]
    all_exist = all(os.path.exists(get_cached_filepath(s)) for s in splits)

    if all_exist and not force_recompute:
        print("[cache] All embedding caches exist. Skipping computation.")
        return

    print(f"\n[cache] Loading frozen encoder: {config.ENCODER_MODEL} for caching...")
    encoder = AutoModel.from_pretrained(config.ENCODER_MODEL)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    for s in splits:
        cache_split_embeddings(s, encoder, batch_size=8, force_recompute=force_recompute)

    print("\n[cache] All embedding caches successfully generated!")


if __name__ == "__main__":
    build_all_embedding_caches()
