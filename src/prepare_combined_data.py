"""
prepare_combined_data.py - Merge all 4 ASTE benchmark splits and preprocess features.

Splits:
- 14res (1266 train, 310 dev, 492 test)
- 14lap (906 train, 219 dev, 328 test)
- 15res (605 train, 148 dev, 322 test)
- 16res (857 train, 210 dev, 326 test)

Combined:
- 3,634 Train sentences
- 887 Dev sentences
- 1,468 Test sentences
"""

import os
import sys
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import load_aste_dataset
from src.preprocess import preprocess_split, save_features


def prepare_combined():
    print("=" * 70)
    print(" PREPARING COMBINED ASTE DATASET (14res + 14lap + 15res + 16res)")
    print("=" * 70)

    splits = ["14res", "14lap", "15res", "16res"]
    combined_train_raw = []
    combined_dev_raw   = []
    combined_test_raw  = []
    individual_tests   = {}

    for s in splits:
        tr, dv, te = load_aste_dataset(s)
        combined_train_raw.extend(tr)
        combined_dev_raw.extend(dv)
        combined_test_raw.extend(te)
        individual_tests[s] = te

    print("\n[dataset] Combined Summary:")
    print(f"  • Total Train: {len(combined_train_raw)} sentences")
    print(f"  • Total Dev:   {len(combined_dev_raw)} sentences")
    print(f"  • Total Test:  {len(combined_test_raw)} sentences")

    print(f"\n[preprocess] Loading tokenizer: {config.ENCODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.ENCODER_MODEL)

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)

    # 1. Preprocess combined splits
    print("\n[preprocess] Encoding combined train split...")
    comb_train_feats = preprocess_split(combined_train_raw, tokenizer, "comb_train")
    save_features(comb_train_feats, os.path.join(config.PROCESSED_DIR, "combined_train_aste.pt"))

    print("\n[preprocess] Encoding combined dev split...")
    comb_dev_feats = preprocess_split(combined_dev_raw, tokenizer, "comb_dev")
    save_features(comb_dev_feats, os.path.join(config.PROCESSED_DIR, "combined_dev_aste.pt"))

    print("\n[preprocess] Encoding combined test split...")
    comb_test_feats = preprocess_split(combined_test_raw, tokenizer, "comb_test")
    save_features(comb_test_feats, os.path.join(config.PROCESSED_DIR, "combined_test_aste.pt"))

    # 2. Preprocess individual test splits for benchmark breakdown
    for s, te_data in individual_tests.items():
        out_path = os.path.join(config.PROCESSED_DIR, f"{s}_test_aste.pt")
        if not os.path.exists(out_path):
            print(f"\n[preprocess] Encoding individual test split: {s}...")
            te_feats = preprocess_split(te_data, tokenizer, f"{s}_test")
            save_features(te_feats, out_path)
        else:
            print(f"[preprocess] Already exists -> {out_path}")

    print("\n[prepare] All combined and individual features successfully prepared!")


if __name__ == "__main__":
    prepare_combined()
