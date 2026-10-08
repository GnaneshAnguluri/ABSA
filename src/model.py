"""
model.py - Context-Aware End-to-End ASTE Model.

Architecture:
1. Transformer Backbone (BERT) - frozen in Phase 1 (requires_grad=False)
2. Parallel BIO Heads:
   - Aspect BIO Classifier: Linear(768, 3) -> O, B-ASP, I-ASP
   - Opinion BIO Classifier: Linear(768, 3) -> O, B-OPN, I-OPN
3. Span Pooling:
   - Mean pooling over token embeddings for aspect and opinion spans
4. Pair Representation:
   - [h_aspect ; h_opinion] -> 1536-dim vector
5. Pair Relation Classifier:
   - Linear(1536, 2) -> INVALID (0) vs. VALID (1)
6. Pair Sentiment Classifier:
   - Linear(1536, 3) -> NEG (0), NEU (1), POS (2)
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ASTEModel(nn.Module):
    def __init__(self,
                 encoder_name: str = config.ENCODER_MODEL,
                 hidden_size: int = config.HIDDEN_SIZE,
                 pair_hidden_size: int = config.PAIR_HIDDEN_SIZE,
                 freeze_encoder: bool = config.FREEZE_ENCODER):
        super(ASTEModel, self).__init__()

        # 1. BERT Backbone
        self.encoder = AutoModel.from_pretrained(encoder_name)
        if freeze_encoder:
            print("[model] Phase 1: Freezing BERT encoder parameters (requires_grad=False)")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print("[model] Phase 2: Fine-tuning BERT encoder (requires_grad=True)")

        self.dropout = nn.Dropout(0.1)

        # 2. BIO Sequence Labeling Heads
        self.aspect_classifier = nn.Linear(hidden_size, config.NUM_ASPECT_LABELS)
        self.opinion_classifier = nn.Linear(hidden_size, config.NUM_OPINION_LABELS)

        # 3. Pair Relation Classifier: 1536 -> 2 (INVALID, VALID)
        self.relation_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(pair_hidden_size, config.NUM_RELATION_LABELS)
        )

        # 4. Pair Sentiment Classifier: 1536 -> 3 (NEG, NEU, POS)
        self.sentiment_classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(pair_hidden_size, config.NUM_SENTIMENT_LABELS)
        )

        # Loss Functions
        self.bio_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.relation_loss_fn = nn.CrossEntropyLoss()
        self.sentiment_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def pool_span(self, token_embeddings, start_tok, end_tok):
        """
        Mean-pool contextual token representations over span [start_tok, end_tok] (inclusive).
        """
        span_reps = token_embeddings[start_tok : end_tok + 1]
        return span_reps.mean(dim=0)

    def forward(self,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                embeddings: torch.Tensor = None,
                aspect_labels: torch.Tensor = None,
                opinion_labels: torch.Tensor = None,
                pair_batch_indices: torch.Tensor = None,
                aspect_tok_spans: torch.Tensor = None,
                opinion_tok_spans: torch.Tensor = None,
                relation_labels: torch.Tensor = None,
                sentiment_labels: torch.Tensor = None):
        """
        Forward pass for ASTE multi-task pipeline.
        Supports precomputed cached embeddings to bypass BERT on CPU.
        """
        # Step 1: Transformer Contextual Token Embeddings
        if embeddings is not None:
            # Cast float16 -> float32 for downstream linear layers
            sequence_output = embeddings.to(torch.float32)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state   # [Batch, SeqLen, 768]

        sequence_output = self.dropout(sequence_output)

        # Step 2: Parallel BIO Logits
        aspect_logits = self.aspect_classifier(sequence_output)   # [Batch, SeqLen, 3]
        opinion_logits = self.opinion_classifier(sequence_output) # [Batch, SeqLen, 3]

        # Step 3: Candidate Pair Representations
        relation_logits = None
        sentiment_logits = None
        total_loss = None
        loss_dict = {}

        if pair_batch_indices is not None and len(pair_batch_indices) > 0:
            relation_logits, sentiment_logits = self.classify_pairs(
                sequence_output=sequence_output,
                pair_batch_indices=pair_batch_indices,
                aspect_tok_spans=aspect_tok_spans,
                opinion_tok_spans=opinion_tok_spans
            )

        # Step 5: Multi-Task Loss Calculation (if targets are supplied)
        if aspect_labels is not None and opinion_labels is not None:
            # Aspect BIO loss
            loss_aspect = self.bio_loss_fn(
                aspect_logits.view(-1, config.NUM_ASPECT_LABELS),
                aspect_labels.view(-1)
            )
            # Opinion BIO loss
            loss_opinion = self.bio_loss_fn(
                opinion_logits.view(-1, config.NUM_OPINION_LABELS),
                opinion_labels.view(-1)
            )

            # Relation and Sentiment losses
            if relation_logits is not None and relation_labels is not None:
                loss_relation = self.relation_loss_fn(relation_logits, relation_labels)
            else:
                loss_relation = torch.tensor(0.0, device=input_ids.device)

            if sentiment_logits is not None and sentiment_labels is not None:
                # Note: -100 is ignored, so INVALID pairs don't penalize sentiment loss
                loss_sentiment = self.sentiment_loss_fn(sentiment_logits, sentiment_labels)
            else:
                loss_sentiment = torch.tensor(0.0, device=input_ids.device)

            total_loss = (
                config.LAMBDA_ASPECT * loss_aspect +
                config.LAMBDA_OPINION * loss_opinion +
                config.LAMBDA_RELATION * loss_relation +
                config.LAMBDA_SENTIMENT * loss_sentiment
            )

            loss_dict = {
                "loss_total"    : total_loss.item(),
                "loss_aspect"   : loss_aspect.item(),
                "loss_opinion"  : loss_opinion.item(),
                "loss_relation" : loss_relation.item() if isinstance(loss_relation, torch.Tensor) else loss_relation,
                "loss_sentiment": loss_sentiment.item() if isinstance(loss_sentiment, torch.Tensor) else loss_sentiment
            }

        return {
            "loss"            : total_loss,
            "loss_dict"       : loss_dict,
            "sequence_output" : sequence_output,
            "aspect_logits"   : aspect_logits,
            "opinion_logits"  : opinion_logits,
            "relation_logits" : relation_logits,
            "sentiment_logits": sentiment_logits
        }

    def classify_pairs(self, sequence_output: torch.Tensor, pair_batch_indices: torch.Tensor,
                       aspect_tok_spans: torch.Tensor, opinion_tok_spans: torch.Tensor):
        """
        Takes arbitrary candidate pairs and predicts relation and sentiment logits.
        Used during evaluation on model-predicted spans.
        """
        if len(pair_batch_indices) == 0:
            return None, None

        pair_vectors = []
        for idx in range(len(pair_batch_indices)):
            b_idx = pair_batch_indices[idx].item()
            asp_s, asp_e = aspect_tok_spans[idx][0].item(), aspect_tok_spans[idx][1].item()
            opn_s, opn_e = opinion_tok_spans[idx][0].item(), opinion_tok_spans[idx][1].item()

            sent_tokens = sequence_output[b_idx]
            h_asp = self.pool_span(sent_tokens, asp_s, asp_e)
            h_opn = self.pool_span(sent_tokens, opn_s, opn_e)
            h_pair = torch.cat([h_asp, h_opn], dim=-1)
            pair_vectors.append(h_pair)

        pair_reps = torch.stack(pair_vectors, dim=0)
        rel_logits = self.relation_classifier(pair_reps)
        sent_logits = self.sentiment_classifier(pair_reps)
        return rel_logits, sent_logits
