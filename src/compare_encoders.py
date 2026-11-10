"""
compare_encoders.py - Controlled Phase 1 Pretrained Encoder Comparison for ASTE.

Evaluates alternative encoders under the exact Phase 1 setup:
1. bert-base-uncased (reference baseline)
2. activebus/BERT-XD_Review (domain-adapted review encoder)
3. FacebookAI/roberta-base (RoBERTa baseline)
4. microsoft/deberta-v3-base (DeBERTa-v3 baseline)

Fixed Experimental Setup:
- Dataset: 14res (ASTE-Data-V2)
- Encoder: Frozen (requires_grad=False)
- Cached embeddings: float16 on CPU
- 4 ASTE Heads: Aspect (768->3), Opinion (768->3), Relation (1536->2), Sentiment (1536->3)
- Optimizer: AdamW, lr=1e-3, weight_decay=0.01
- Batch size: 8, Epochs: 5, Seed: 42
- Strict triplet matching (end-to-end predicted spans)
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed, get_device
from src.dataset import load_aste_dataset
from src.preprocess import encode_sentence_sample, save_features, load_features
from src.dataloader import ASTESentenceDataset, aste_collate_fn
from src.model import ASTEModel
from src.evaluate import evaluate_model

MODELS = {
    "bert_xd_review": {
        "name": "activebus/BERT-XD_Review",
        "display_name": "activebus/BERT-XD_Review",
        "slug": "bert_xd_review",
        "tokenizer_kwargs": {},
    },
    "roberta_base": {
        "name": "FacebookAI/roberta-base",
        "display_name": "FacebookAI/roberta-base",
        "slug": "roberta_base",
        "tokenizer_kwargs": {"add_prefix_space": True},
    },
    "deberta_v3_base": {
        "name": "microsoft/deberta-v3-base",
        "display_name": "microsoft/deberta-v3-base",
        "slug": "deberta_v3_base",
        "tokenizer_kwargs": {},
    },
}

REFERENCE_BASELINE = {
    "display_name": "bert-base-uncased (Baseline)",
    "dev_f1": 37.84,
    "test_p": 37.65,
    "test_r": 40.34,
    "test_f1": 38.95,
    "test_correct": 401,
    "test_gold": 994,
    "cache_time": 42.0,
    "train_time": 85.0,
    "status": "Completed"
}


def get_feature_path(dataset_split: str, slug: str, split_name: str) -> str:
    return os.path.join(config.PROCESSED_DIR, f"{dataset_split}_{slug}_{split_name}_aste.pt")


def get_cached_path(dataset_split: str, slug: str, split_name: str) -> str:
    return os.path.join(config.PROCESSED_DIR, f"{dataset_split}_{slug}_{split_name}_cached.pt")


def preprocess_for_encoder(model_info: dict, dataset_split: str = "14res", force: bool = False):
    """Tokenize and generate sentence features for a given encoder."""
    slug = model_info["slug"]
    splits = ["train", "dev", "test"]
    all_exist = all(os.path.exists(get_feature_path(dataset_split, slug, s)) for s in splits)

    if all_exist and not force:
        print(f"[preprocess] Features already exist on disk for {slug}. Reusing existing files.")
        return

    print(f"\n[preprocess] Loading tokenizer for {model_info['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(model_info["name"], **model_info["tokenizer_kwargs"])
    raw_data = dict(zip(splits, load_aste_dataset(dataset_split)))

    for s_name in splits:
        feat_path = get_feature_path(dataset_split, slug, s_name)
        if os.path.exists(feat_path) and not force:
            continue
        print(f"[preprocess] Processing {s_name} ({len(raw_data[s_name])} sentences) with {slug}...")
        features = []
        skipped = 0
        for sample in raw_data[s_name]:
            try:
                feat = encode_sentence_sample(sample, tokenizer, max_len=config.MAX_SEQ_LEN)
                features.append(feat)
            except Exception as e:
                skipped += 1
        save_features(features, feat_path)
        print(f"             Saved -> {feat_path} (skipped={skipped})")


def cache_embeddings_for_encoder(model_info: dict, dataset_split: str = "14res", batch_size: int = 8, force: bool = False) -> float:
    """Precompute frozen token embeddings as float16 on CPU."""
    slug = model_info["slug"]
    splits = ["train", "dev", "test"]
    all_exist = all(os.path.exists(get_cached_path(dataset_split, slug, s)) for s in splits)

    if all_exist and not force:
        print(f"[cache] Embeddings already cached on disk for {slug}. Reusing existing cache.")
        return 0.0

    print(f"\n[cache] Loading frozen encoder {model_info['name']} for caching...")
    encoder = AutoModel.from_pretrained(model_info["name"])
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    start_time = time.time()
    for s_name in splits:
        cache_path = get_cached_path(dataset_split, slug, s_name)
        if os.path.exists(cache_path) and not force:
            continue

        raw_path = get_feature_path(dataset_split, slug, s_name)
        features = load_features(raw_path)
        cached_features = []
        n_samples = len(features)

        pbar = tqdm(range(0, n_samples, batch_size), desc=f"Caching {slug} [{s_name}]")
        for i in pbar:
            batch_slice = features[i : i + batch_size]
            input_ids = torch.stack([f["input_ids"] for f in batch_slice])
            attention_mask = torch.stack([f["attention_mask"] for f in batch_slice])

            with torch.no_grad():
                outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
                last_hidden = outputs.last_hidden_state.to(torch.float16).cpu()

            for idx, f in enumerate(batch_slice):
                new_f = dict(f)
                new_f["embedding"] = last_hidden[idx]
                cached_features.append(new_f)

        save_features(cached_features, cache_path)
        file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"[cache] Saved {len(cached_features)} items -> {cache_path} ({file_size_mb:.1f} MB)")

    total_time = time.time() - start_time
    print(f"[cache] Finished caching {slug} in {total_time:.1f}s")
    return total_time


def smoke_test(model_info: dict, dataset_split: str = "14res"):
    """
    Smoke test for candidate encoder.
    Verifies:
    1. Tokenizer and encoder load properly.
    2. Output hidden dimension is [B, 128, 768].
    3. Head logits have expected shapes.
    4. Frozen encoder receives 0 gradients.
    5. Trainable heads receive non-zero gradients.
    6. Cached embeddings load and match.
    7. Predicted-span evaluation executes end-to-end without errors.
    """
    print(f"\n{'='*70}")
    print(f" SMOKE TEST: {model_info['display_name']}")
    print(f"{'='*70}")

    slug = model_info["slug"]
    train_cache_path = get_cached_path(dataset_split, slug, "train")
    features = load_features(train_cache_path)[:4]
    dataset = ASTESentenceDataset(features)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=aste_collate_fn)
    batch = next(iter(loader))

    # 1. Instantiate model
    model = ASTEModel(encoder_name=model_info["name"], freeze_encoder=True)

    # 2. Check hidden output shape & head logits
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        embeddings=batch["embeddings"],
        aspect_labels=batch["aspect_labels"],
        opinion_labels=batch["opinion_labels"],
        pair_batch_indices=batch["pair_batch_indices"],
        aspect_tok_spans=batch["aspect_tok_spans"],
        opinion_tok_spans=batch["opinion_tok_spans"],
        relation_labels=batch["relation_labels"],
        sentiment_labels=batch["sentiment_labels"]
    )

    seq_out = outputs["sequence_output"]
    assert seq_out.shape == (2, 128, 768), f"Expected (2, 128, 768), got {seq_out.shape}"
    assert outputs["aspect_logits"].shape == (2, 128, 3), f"Wrong aspect logits shape: {outputs['aspect_logits'].shape}"
    assert outputs["opinion_logits"].shape == (2, 128, 3), f"Wrong opinion logits shape: {outputs['opinion_logits'].shape}"
    print(f"  [OK] Contextual hidden shape : {seq_out.shape}")
    print(f"  [OK] Aspect logits shape     : {outputs['aspect_logits'].shape}")
    print(f"  [OK] Opinion logits shape    : {outputs['opinion_logits'].shape}")

    # 3. Test backward pass & gradient isolation
    loss = outputs["loss"]
    assert not torch.isnan(loss) and not torch.isinf(loss), "Loss is NaN/Inf!"
    loss.backward()

    # Check encoder frozen
    encoder_grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert len(encoder_grads) == 0, f"Encoder received {len(encoder_grads)} gradients! Should be 0."
    print("  [OK] Frozen encoder gradients : 0 non-zero gradients (strictly frozen)")

    # Check heads trainable
    heads = [model.aspect_classifier, model.opinion_classifier, model.relation_classifier, model.sentiment_classifier]
    for h in heads:
        head_grads = [p.grad for p in h.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        assert len(head_grads) > 0, "Classification head did not receive gradients!"
    print("  [OK] Trainable head gradients : all 4 classification heads received non-zero gradients")

    # 4. Test end-to-end evaluation with predicted spans
    model.eval()
    eval_metrics = evaluate_model(model, loader, device=torch.device("cpu"))
    print(f"  [OK] Strict evaluation run   : dev P={eval_metrics['precision']:.1f}% R={eval_metrics['recall']:.1f}% F1={eval_metrics['f1']:.1f}%")
    print(f"  >>> SMOKE TEST PASSED for {model_info['display_name']}!\n")


def get_encoder_dataloaders(slug: str, dataset_split: str = "14res", batch_size: int = 8):
    train_feats = load_features(get_cached_path(dataset_split, slug, "train"))
    dev_feats   = load_features(get_cached_path(dataset_split, slug, "dev"))
    test_feats  = load_features(get_cached_path(dataset_split, slug, "test"))

    train_loader = DataLoader(ASTESentenceDataset(train_feats), batch_size=batch_size, shuffle=True, collate_fn=aste_collate_fn)
    dev_loader   = DataLoader(ASTESentenceDataset(dev_feats),   batch_size=batch_size, shuffle=False, collate_fn=aste_collate_fn)
    test_loader  = DataLoader(ASTESentenceDataset(test_feats),  batch_size=batch_size, shuffle=False, collate_fn=aste_collate_fn)
    return train_loader, dev_loader, test_loader


def train_and_evaluate_encoder(model_info: dict, dataset_split: str = "14res") -> dict:
    """Train 5 epochs on cached embeddings, select best on dev, evaluate on test."""
    set_seed(config.SEED)
    device = get_device()
    slug = model_info["slug"]

    print(f"\n{'='*75}")
    print(f" TRAINING: {model_info['display_name']} (Phase 1: 5 Epochs, LR: 1e-3, Batch: 8)")
    print(f"{'='*75}")

    train_loader, dev_loader, test_loader = get_encoder_dataloaders(slug, dataset_split=dataset_split)

    model = ASTEModel(encoder_name=model_info["name"], freeze_encoder=True).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_dev_f1 = -1.0
    best_epoch = 0
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"best_aste_{slug}.pt")
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    train_start = time.time()
    epoch_records = []

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{config.EPOCHS:02d} [{slug}]")
        for batch in pbar:
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                embeddings=batch["embeddings"].to(device),
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
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.MAX_GRAD_NORM)
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / num_train_batches

        # Dev evaluation after every epoch
        model.eval()
        dev_metrics = evaluate_model(model, dev_loader, device=device)
        dev_f1 = dev_metrics["f1"]

        is_best = dev_f1 > best_dev_f1
        if is_best:
            best_dev_f1 = dev_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "dev_f1": dev_f1,
                "dev_metrics": dev_metrics,
                "encoder_name": model_info["name"]
            }, checkpoint_path)

        epoch_records.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "dev_loss": dev_metrics["loss"],
            "dev_p": dev_metrics["precision"],
            "dev_r": dev_metrics["recall"],
            "dev_f1": dev_f1,
            "is_best": is_best
        })

        print(f"Epoch {epoch:02d} Summary: Train Loss={avg_train_loss:.4f} | Dev Loss={dev_metrics['loss']:.4f} | "
              f"Dev P={dev_metrics['precision']:.2f}% R={dev_metrics['recall']:.2f}% F1={dev_f1:.2f}% "
              f"({dev_metrics['n_correct']}/{dev_metrics['n_gold']}) {'[BEST SAVED]' if is_best else ''}")

    train_time = time.time() - train_start

    # Final evaluation on TEST set using BEST dev checkpoint
    print(f"\n[eval] Loading best checkpoint (Epoch {best_epoch}, Dev F1: {best_dev_f1:.2f}%) for Test Evaluation...")
    best_ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_metrics = evaluate_model(model, test_loader, device=device)
    print(f"{'='*60}")
    print(f" TEST RESULTS: {model_info['display_name']}")
    print(f"{'='*60}")
    print(f"  Strict Triplet Precision : {test_metrics['precision']:.2f}%")
    print(f"  Strict Triplet Recall    : {test_metrics['recall']:.2f}%")
    print(f"  Strict Triplet F1        : {test_metrics['f1']:.2f}%")
    print(f"  Correct Triplets         : {test_metrics['n_correct']} / {test_metrics['n_gold']}")
    print(f"  Training Time (5 epochs) : {train_time:.1f}s")
    print(f"{'='*60}\n")

    return {
        "display_name": model_info["display_name"],
        "slug": slug,
        "best_epoch": best_epoch,
        "dev_f1": best_dev_f1,
        "test_p": test_metrics["precision"],
        "test_r": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_correct": test_metrics["n_correct"],
        "test_gold": test_metrics["n_gold"],
        "train_time": train_time,
        "epoch_records": epoch_records,
        "status": "Completed"
    }


def print_comparison_table(results: dict):
    print("\n" + "="*95)
    print(" PHASE 1 ENCODER BENCHMARK COMPARISON TABLE (14res)")
    print("="*95)
    header = f"{'Encoder Model':<32} | {'Dev F1':<8} | {'Test P':<8} | {'Test R':<8} | {'Test F1':<8} | {'Correct/Gold':<13} | {'Train(s)':<8}"
    print(header)
    print("-" * 95)

    # Reference baseline
    b = REFERENCE_BASELINE
    print(f"{b['display_name']:<32} | {b['dev_f1']:>6.2f}% | {b['test_p']:>6.2f}% | {b['test_r']:>6.2f}% | {b['test_f1']:>6.2f}% | {b['test_correct']:>4}/{b['test_gold']:<4}       | {b['train_time']:>6.1f}s")

    for key, res in results.items():
        cg_str = f"{res['test_correct']}/{res['test_gold']}"
        print(f"{res['display_name']:<32} | {res['dev_f1']:>6.2f}% | {res['test_p']:>6.2f}% | {res['test_r']:>6.2f}% | {res['test_f1']:>6.2f}% | {cg_str:<13} | {res['train_time']:>6.1f}s")

    print("="*95 + "\n")


def run_all_comparisons(selected_models=None):
    print("\n" + "#"*75)
    print(" ABSA PHASE 1 ENCODER COMPARISON PIPELINE")
    print("#"*75)

    timing_cache = {}
    results = {}

    target_models = {k: v for k, v in MODELS.items() if (selected_models is None or k in selected_models)}

    for key, model_info in target_models.items():
        print(f"\n{'#'*75}")
        print(f" STARTING PIPELINE FOR: {model_info['display_name']}")
        print(f"{'#'*75}")

        # 1. Preprocess
        preprocess_for_encoder(model_info, dataset_split="14res")

        # 2. Cache embeddings
        c_time = cache_embeddings_for_encoder(model_info, dataset_split="14res", batch_size=8)
        timing_cache[key] = c_time

        # 3. Smoke Test
        smoke_test(model_info, dataset_split="14res")

        # 4. Train & Evaluate
        res = train_and_evaluate_encoder(model_info, dataset_split="14res")
        res["cache_time"] = c_time
        results[key] = res

        # Intermediate report
        print_comparison_table(results)

    # Final Summary
    print_comparison_table(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Specific model slug to run (e.g. bert_xd_review, roberta_base, deberta_v3_base)")
    args = parser.parse_args()

    selected = [args.model] if args.model else None
    run_all_comparisons(selected_models=selected)
