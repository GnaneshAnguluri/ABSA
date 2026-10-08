"""
evaluate.py - End-to-End Strict ASTE Triplet Evaluation.

Evaluation Pipeline:
1. Sentence tokens -> BERT -> Aspect BIO logits & Opinion BIO logits
2. argmax BIO predictions -> extract predicted aspect token spans & opinion token spans
3. Map predicted subword token spans back to word spans using word_ids
4. Cartesian product of predicted aspects x predicted opinions
5. Pool candidate pair representations from BERT contextual embeddings
6. Relation classifier -> keep only pairs predicted VALID (label 1)
7. Sentiment classifier -> assign POS / NEG / NEU
8. Compare predicted (Aspect, Opinion, Sentiment) triplets against gold triplets with strict exact matching.
"""

import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def extract_spans_from_bio(bio_ids: list, tag_type: str = "ASP"):
    """
    Extract token spans [start_idx, end_idx] (inclusive) from a sequence of BIO tag IDs.
    Handles B-TAG followed by consecutive I-TAGs.
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


def token_span_to_word_span(token_span: list, word_ids: list):
    """
    Maps a BERT subword token span [tok_start, tok_end] (inclusive)
    back to the original word-level span [w_start, w_end] (inclusive).

    Returns:
        [w_start, w_end] or None if the token span contains only special/padding tokens.
    """
    tok_start, tok_end = token_span
    mapped_word_indices = []

    for t_idx in range(tok_start, tok_end + 1):
        if t_idx < len(word_ids):
            w_id = word_ids[t_idx]
            if w_id is not None:
                mapped_word_indices.append(w_id)

    if len(mapped_word_indices) == 0:
        return None

    return [min(mapped_word_indices), max(mapped_word_indices)]


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
    Run end-to-end evaluation over a DataLoader.
    Extracts triplets strictly from predicted BIO spans and model classification heads.
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

            # 1. Forward pass for loss calculation (on gold pairs for consistent loss reporting)
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

            # Format gold triplets
            for g_list in batch["gold_triples"]:
                formatted_gold = [
                    (tuple(t["aspect_span"]), tuple(t["opinion_span"]), t["sentiment"])
                    for t in g_list
                ]
                gold_all.append(formatted_gold)

            # 2. End-to-End Extraction from Model's Predicted BIO Logits
            sequence_output = outputs["sequence_output"]  # [B, SeqLen, 768]
            asp_preds = outputs["aspect_logits"].argmax(dim=-1).cpu().tolist()   # [B, SeqLen]
            opn_preds = outputs["opinion_logits"].argmax(dim=-1).cpu().tolist()  # [B, SeqLen]

            batch_size = input_ids.shape[0]

            # Build candidate pairs from predicted spans per sentence
            eval_batch_indices = []
            eval_asp_tok_spans = []
            eval_opn_tok_spans = []
            eval_pair_metadata = []

            for b_idx in range(batch_size):
                word_ids = batch["word_ids"][b_idx]

                # Extract predicted token spans from BIO output
                pred_asp_tok_spans = extract_spans_from_bio(asp_preds[b_idx], tag_type="ASP")
                pred_opn_tok_spans = extract_spans_from_bio(opn_preds[b_idx], tag_type="OPN")

                # Map token spans to word spans
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

                # Cartesian product of predicted aspects x predicted opinions
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

            # Run relation and sentiment classification on predicted pairs
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

                pred_rels  = rel_logits.argmax(dim=-1).cpu().tolist()
                pred_sents = sent_logits.argmax(dim=-1).cpu().tolist()

                for p_idx, meta in enumerate(eval_pair_metadata):
                    # Keep only pairs classified as VALID (label 1)
                    if pred_rels[p_idx] == 1:
                        sentiment_str = config.ID2SENTIMENT[pred_sents[p_idx]]
                        predicted_triplet = (
                            tuple(meta["aspect_word_span"]),
                            tuple(meta["opinion_word_span"]),
                            sentiment_str
                        )
                        batch_preds[meta["batch_idx"]].append(predicted_triplet)

            pred_all.extend(batch_preds)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    metrics = compute_aste_metrics(gold_all, pred_all)
    metrics["loss"] = avg_loss
    return metrics
