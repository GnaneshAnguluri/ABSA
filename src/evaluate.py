"""
evaluate.py - Step 10: Strict ASTE Triplet Evaluation (Precision, Recall, F1).

A predicted triplet (Aspect, Opinion, Sentiment) is counted as CORRECT if and only if:
1. Aspect word span matches gold exactly
2. Opinion word span matches gold exactly
3. Sentiment polarity matches gold exactly
"""

import os
import sys
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def extract_spans_from_bio(bio_ids: list, tag_type: str = "ASP"):
    """
    Extract token/word spans from a sequence of BIO tag IDs.
    Returns: list of (start_idx, end_idx) inclusive.
    """
    spans = []
    current_span = None

    b_id = config.ASPECT_TAGS[f"B-{tag_type}"] if tag_type == "ASP" else config.OPINION_TAGS[f"B-{tag_type}"]
    i_id = config.ASPECT_TAGS[f"I-{tag_type}"] if tag_type == "ASP" else config.OPINION_TAGS[f"I-{tag_type}"]

    for idx, tag in enumerate(bio_ids):
        if tag == b_id:
            if current_span is not None:
                spans.append(current_span)
            current_span = [idx, idx]
        elif tag == i_id:
            if current_span is not None:
                current_span[1] = idx
        else:
            if current_span is not None:
                spans.append(current_span)
                current_span = None

    if current_span is not None:
        spans.append(current_span)

    return spans


def compute_aste_metrics(gold_triplets_all: list, pred_triplets_all: list):
    """
    Compute strict Triplet Precision, Recall, and F1.

    Each triplet is represented as: ( (a_start, a_end), (o_start, o_end), sentiment )
    """
    n_gold = sum(len(g) for g in gold_triplets_all)
    n_pred = sum(len(p) for p in pred_triplets_all)
    n_correct = 0

    for gold_list, pred_list in zip(gold_triplets_all, pred_triplets_all):
        gold_set = set(gold_list)
        pred_set = set(pred_list)
        n_correct += len(gold_set.intersection(pred_set))

    precision = n_correct / n_pred if n_pred > 0 else 0.0
    recall    = n_correct / n_gold if n_gold > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision * 100,
        "recall"   : recall * 100,
        "f1"       : f1 * 100,
        "n_gold"   : n_gold,
        "n_pred"   : n_pred,
        "n_correct": n_correct
    }


def evaluate_model(model, data_loader, device):
    """
    Run evaluation over a DataLoader.
    Computes loss and strict ASTE triplet extraction metrics.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    gold_all = []
    pred_all = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids       = batch["input_ids"].to(device)
            attention_mask  = batch["attention_mask"].to(device)
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
                aspect_labels=aspect_labels,
                opinion_labels=opinion_labels,
                pair_batch_indices=p_indices,
                aspect_tok_spans=a_spans,
                opinion_tok_spans=o_spans,
                relation_labels=r_labels,
                sentiment_labels=s_labels
            )

            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
            num_batches += 1

            # Format gold triplets for each sentence in batch
            batch_gold = batch["gold_triples"]
            for g_list in batch_gold:
                formatted_gold = [
                    (tuple(t["aspect_span"]), tuple(t["opinion_span"]), t["sentiment"])
                    for t in g_list
                ]
                gold_all.append(formatted_gold)

            # Extract predicted triplets from relation & sentiment logits
            batch_size = input_ids.shape[0]
            batch_preds = [[] for _ in range(batch_size)]

            if outputs["relation_logits"] is not None and len(outputs["relation_logits"]) > 0:
                rel_preds  = outputs["relation_logits"].argmax(dim=-1).cpu().tolist()
                sent_preds = outputs["sentiment_logits"].argmax(dim=-1).cpu().tolist()

                for p_idx, meta in enumerate(batch["pair_metadata"]):
                    is_valid = rel_preds[p_idx] == 1
                    if is_valid:
                        pred_sentiment = config.ID2SENTIMENT[sent_preds[p_idx]]
                        pred_triplet = (
                            tuple(meta["aspect_word_span"]),
                            tuple(meta["opinion_word_span"]),
                            pred_sentiment
                        )
                        batch_preds[meta["batch_idx"]].append(pred_triplet)

            pred_all.extend(batch_preds)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = compute_aste_metrics(gold_all, pred_all)
    metrics["loss"] = avg_loss
    return metrics
