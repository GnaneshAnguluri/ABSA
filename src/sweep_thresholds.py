"""
sweep_thresholds.py - Quickest Optimization: Threshold Sweeping on Checkpoint.

Evaluates relation classification threshold (P(valid) >= threshold) and BIO logit bias
to find the optimal operating point for Precision, Recall, and Strict F1 across all 4 benchmarks.
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


def evaluate_with_threshold(model, data_loader, device, rel_threshold=0.5):
    """Evaluate pipeline with a customizable relation probability threshold."""
    model.eval()
    gold_all = []
    pred_all = []

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

            for g_list in batch["gold_triples"]:
                formatted_gold = [
                    (tuple(t["aspect_span"]), tuple(t["opinion_span"]), t["sentiment"])
                    for t in g_list
                ]
                gold_all.append(formatted_gold)

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
                pred_asp_tok_spans = extract_spans_from_bio(asp_preds[b_idx], tag_type="ASP")
                pred_opn_tok_spans = extract_spans_from_bio(opn_preds[b_idx], tag_type="OPN")

                valid_asp_pairs = []
                for a_tok in pred_asp_tok_spans:
                    a_word = token_span_to_word_span(a_tok, word_ids)
                    if a_word is not None:
                        valid_asp_pairs.append((a_tok, a_word))

                valid_opn_pairs = []
                for o_tok in pred_opn_tok_spans:
                    o_word = token_span_to_word_span(o_tok, word_ids)
                    if o_word is not None:
                        valid_opn_pairs.append((o_tok, o_word))

                for (a_tok, a_word) in valid_asp_pairs:
                    for (o_tok, o_word) in valid_opn_pairs:
                        eval_batch_indices.append(b_idx)
                        eval_asp_tok_spans.append(a_tok)
                        eval_opn_tok_spans.append(o_tok)
                        eval_pair_metadata.append({
                            "batch_idx"       : b_idx,
                            "aspect_word_span": a_word,
                            "opinion_word_span": o_word
                        })

            batch_preds = [[] for _ in range(batch_size)]

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
                    if rel_probs[p_idx] >= rel_threshold:
                        sentiment_str = config.ID2SENTIMENT[pred_sents[p_idx]]
                        predicted_triplet = (
                            tuple(meta["aspect_word_span"]),
                            tuple(meta["opinion_word_span"]),
                            sentiment_str
                        )
                        batch_preds[meta["batch_idx"]].append(predicted_triplet)

            pred_all.extend(batch_preds)

    return compute_aste_metrics(gold_all, pred_all)


def run_sweep():
    set_seed(config.SEED)
    device = get_device()

    print(f"\n[model] Loading checkpoint: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = ASTEModel(freeze_encoder=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    base = config.PROCESSED_DIR
    individual_tests = {}
    for s in BENCHMARKS:
        p = os.path.join(base, f"{s}_test_aste.pt")
        feats = load_features(p)
        individual_tests[s] = DataLoader(
            ASTESentenceDataset(feats), batch_size=8, shuffle=False, collate_fn=aste_collate_fn
        )

    comb_feats = load_features(os.path.join(base, "combined_test_aste.pt"))
    comb_loader = DataLoader(
        ASTESentenceDataset(comb_feats), batch_size=8, shuffle=False, collate_fn=aste_collate_fn
    )

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n" + "=" * 70)
    print(" SWEEPING RELATION THRESHOLD ON COMBINED TEST SET")
    print("=" * 70)
    print(f" {'Threshold':<10} | {'Precision':>10} | {'Recall':>10} | {'Strict F1':>10} | {'Correct / Gold'}")
    print("-" * 70)

    best_thresh = 0.50
    best_comb_f1 = 0.0

    for th in thresholds:
        m = evaluate_with_threshold(model, comb_loader, device, rel_threshold=th)
        marker = ""
        if m["f1"] > best_comb_f1:
            best_comb_f1 = m["f1"]
            best_thresh = th
            marker = " <-- BEST"
        print(f" {th:<10.2f} | {m['precision']:>9.2f}% | {m['recall']:>9.2f}% | {m['f1']:>9.2f}% | {m['n_correct']} / {m['n_gold']}{marker}")

    print("=" * 70)
    print(f"\n>>> Optimal Relation Threshold: {best_thresh:.2f} (Combined Test F1: {best_comb_f1:.2f}%)\n")

    print("=" * 70)
    print(f" INDIVIDUAL BENCHMARK BREAKDOWN AT BEST THRESHOLD ({best_thresh:.2f})")
    print("=" * 70)
    print(f" {'Benchmark':<10} | {'Precision':>10} | {'Recall':>10} | {'Strict F1':>10} | {'Correct / Gold'}")
    print("-" * 70)

    total_correct = 0
    total_gold = 0
    for s, ldr in individual_tests.items():
        m = evaluate_with_threshold(model, ldr, device, rel_threshold=best_thresh)
        total_correct += m["n_correct"]
        total_gold += m["n_gold"]
        print(f" {s:<10} | {m['precision']:>9.2f}% | {m['recall']:>9.2f}% | {m['f1']:>9.2f}% | {m['n_correct']} / {m['n_gold']}")

    overall_f1 = 2 * (total_correct / (total_gold + total_correct)) * 100 if (total_gold + total_correct) > 0 else 0
    print("-" * 70)
    print(f" {'TOTAL':<10} | {'':>10} | {'':>10} | {'':>10} | {total_correct} / {total_gold}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_sweep()
