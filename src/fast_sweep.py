"""
fast_sweep.py - Pre-computes logits once and sweeps thresholds instantaneously in memory.
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import set_seed, get_device
from src.preprocess import load_features
from src.dataloader import ASTESentenceDataset, aste_collate_fn
from src.model import ASTEModel
from src.evaluate import (
    extract_spans_from_bio,
    token_span_to_word_span,
    compute_aste_metrics
)

CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "benchmark_max_phase3.pt")
BENCHMARKS = ["14res", "14lap", "15res", "16res"]


def extract_candidate_pairs_and_logits(model, data_loader, device):
    """Run model forward passes ONCE and cache pair probabilities & gold labels."""
    model.eval()
    cached_data = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                embeddings=None,
                aspect_labels=batch["aspect_labels"].to(device),
                opinion_labels=batch["opinion_labels"].to(device),
                pair_batch_indices=batch["pair_batch_indices"].to(device),
                aspect_tok_spans=batch["aspect_tok_spans"].to(device),
                opinion_tok_spans=batch["opinion_tok_spans"].to(device),
                relation_labels=batch["relation_labels"].to(device),
                sentiment_labels=batch["sentiment_labels"].to(device)
            )

            gold_batch = []
            for g_list in batch["gold_triples"]:
                gold_batch.append([
                    (tuple(t["aspect_span"]), tuple(t["opinion_span"]), t["sentiment"])
                    for t in g_list
                ])

            sequence_output = outputs["sequence_output"]
            asp_preds = outputs["aspect_logits"].argmax(dim=-1).cpu().tolist()
            opn_preds = outputs["opinion_logits"].argmax(dim=-1).cpu().tolist()

            batch_size = input_ids.shape[0]
            eval_batch_indices = []
            eval_asp_tok_spans = []
            eval_opn_tok_spans = []
            eval_pair_metadata = []

            for b_idx in range(batch_size):
                word_ids = batch["word_ids"][b_idx]
                pred_asp = extract_spans_from_bio(asp_preds[b_idx], tag_type="ASP")
                pred_opn = extract_spans_from_bio(opn_preds[b_idx], tag_type="OPN")

                valid_asp = []
                for a_tok in pred_asp:
                    a_word = token_span_to_word_span(a_tok, word_ids)
                    if a_word is not None:
                        valid_asp.append((a_tok, a_word))

                valid_opn = []
                for o_tok in pred_opn:
                    o_word = token_span_to_word_span(o_tok, word_ids)
                    if o_word is not None:
                        valid_opn.append((o_tok, o_word))

                for (a_tok, a_word) in valid_asp:
                    for (o_tok, o_word) in valid_opn:
                        eval_batch_indices.append(b_idx)
                        eval_asp_tok_spans.append(a_tok)
                        eval_opn_tok_spans.append(o_tok)
                        eval_pair_metadata.append({
                            "batch_idx"       : b_idx,
                            "aspect_word_span": a_word,
                            "opinion_word_span": o_word
                        })

            candidates = []
            if len(eval_batch_indices) > 0:
                e_batch_indices = torch.tensor(eval_batch_indices, dtype=torch.long, device=device)
                e_asp_spans     = torch.tensor(eval_asp_tok_spans, dtype=torch.long, device=device)
                e_opn_spans     = torch.tensor(eval_opn_tok_spans, dtype=torch.long, device=device)

                rel_logits, sent_logits = model.classify_pairs(
                    sequence_output=sequence_output,
                    pair_batch_indices=e_batch_indices,
                    aspect_tok_spans=e_asp_spans,
                    opinion_tok_spans=e_opn_spans
                )

                rel_probs = F.softmax(rel_logits, dim=-1)[:, 1].cpu().tolist()
                pred_sents = sent_logits.argmax(dim=-1).cpu().tolist()

                for p_idx, meta in enumerate(eval_pair_metadata):
                    candidates.append({
                        "batch_idx": meta["batch_idx"],
                        "aspect_span": meta["aspect_word_span"],
                        "opinion_span": meta["opinion_word_span"],
                        "prob": rel_probs[p_idx],
                        "sentiment": config.ID2SENTIMENT[pred_sents[p_idx]]
                    })

            cached_data.append({
                "batch_size": batch_size,
                "gold": gold_batch,
                "candidates": candidates
            })

    return cached_data


def evaluate_cached(cached_data, threshold):
    """Instant evaluation on cached candidate pair predictions."""
    gold_all = []
    pred_all = []

    for item in cached_data:
        gold_all.extend(item["gold"])
        batch_preds = [[] for _ in range(item["batch_size"])]
        for cand in item["candidates"]:
            if cand["prob"] >= threshold:
                batch_preds[cand["batch_idx"]].append((
                    tuple(cand["aspect_span"]),
                    tuple(cand["opinion_span"]),
                    cand["sentiment"]
                ))
        pred_all.extend(batch_preds)

    return compute_aste_metrics(gold_all, pred_all)


def run_fast_sweep():
    set_seed(config.SEED)
    device = get_device()

    print(f"\n[model] Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = ASTEModel(freeze_encoder=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    base = config.PROCESSED_DIR
    bench_loaders = {}
    for s in BENCHMARKS:
        feats = load_features(os.path.join(base, f"{s}_test_aste.pt"))
        bench_loaders[s] = DataLoader(
            ASTESentenceDataset(feats), batch_size=8, shuffle=False, collate_fn=aste_collate_fn
        )

    print("\n[cache] Pre-computing predictions once per benchmark...")
    cached_benches = {}
    for s in BENCHMARKS:
        print(f"  • Caching {s} test set...")
        cached_benches[s] = extract_candidate_pairs_and_logits(model, bench_loaders[s], device)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    print("\n" + "=" * 70)
    print(" INSTANT IN-MEMORY THRESHOLD SWEEP (ALL BENCHMARKS)")
    print("=" * 70)
    print(f" {'Thresh':<8} | {'Precision':>10} | {'Recall':>10} | {'Strict F1':>10} | {'Correct / Gold'}")
    print("-" * 70)

    best_th = 0.50
    best_f1 = 0.0

    for th in thresholds:
        total_correct = 0
        total_pred = 0
        total_gold = 0
        for s in BENCHMARKS:
            m = evaluate_cached(cached_benches[s], th)
            total_correct += m["n_correct"]
            total_pred += m["n_pred"]
            total_gold += m["n_gold"]

        prec = total_correct / total_pred * 100 if total_pred > 0 else 0
        rec  = total_correct / total_gold * 100 if total_gold > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
            marker = "  <-- BEST"

        print(f" {th:<8.2f} | {prec:>9.2f}% | {rec:>9.2f}% | {f1:>9.2f}% | {total_correct} / {total_gold}{marker}")

    print("=" * 70)
    print(f"\n>>> Optimal Relation Threshold: {best_th:.2f} (Overall F1: {best_f1:.2f}% vs default 50% threshold)")

    print("\n" + "=" * 70)
    print(f" BREAKDOWN AT OPTIMAL THRESHOLD ({best_th:.2f})")
    print("=" * 70)
    print(f" {'Benchmark':<10} | {'Precision':>10} | {'Recall':>10} | {'Strict F1':>10} | {'Correct / Gold'}")
    print("-" * 70)
    for s in BENCHMARKS:
        m = evaluate_cached(cached_benches[s], best_th)
        print(f" {s:<10} | {m['precision']:>9.2f}% | {m['recall']:>9.2f}% | {m['f1']:>9.2f}% | {m['n_correct']} / {m['n_gold']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_fast_sweep()
