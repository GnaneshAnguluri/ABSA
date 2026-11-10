"""
model_v2.py - Upgraded ASTE Model Architecture with Advanced Non-Linear Classifiers.

Enhancements:
1. BIO Heads: 2-Layer MLP with LayerNorm, GELU, and Dropout.
2. Pair Feature Engineering:
   - [h_asp ; h_opn ; |h_asp - h_opn| ; h_asp * h_opn] -> 3072-dimensional representation
3. Deep Non-Linear Relation Head:
   - Linear(3072 -> 512) -> LayerNorm -> GELU -> Dropout(0.2) -> Linear(512 -> 2)
4. Deep Non-Linear Sentiment Head:
   - Linear(3072 -> 512) -> LayerNorm -> GELU -> Dropout(0.2) -> Linear(512 -> 3)
"""

import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ASTEModelV2(nn.Module):
    def __init__(self,
                 encoder_name: str = config.ENCODER_MODEL,
                 hidden_size: int = config.HIDDEN_SIZE,
                 freeze_encoder: bool = False):
        super(ASTEModelV2, self).__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        if freeze_encoder:
            print("[model_v2] Freezing BERT encoder parameters")
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            print("[model_v2] Fine-tuning BERT encoder (requires_grad=True)")

        self.dropout = nn.Dropout(0.1)

        # 1. Upgraded 2-Layer BIO Heads
        self.aspect_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.NUM_ASPECT_LABELS)
        )

        self.opinion_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.NUM_OPINION_LABELS)
        )

        # 2. Richer Pair Representation: 4 x 768 = 3072 dims
        rich_pair_dim = hidden_size * 4

        # 3. Deep Non-Linear Pair Relation Classifier
        self.relation_classifier = nn.Sequential(
            nn.Linear(rich_pair_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.NUM_RELATION_LABELS)
        )

        # 4. Deep Non-Linear Pair Sentiment Classifier
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(rich_pair_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, config.NUM_SENTIMENT_LABELS)
        )

        # Loss Functions
        self.bio_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        self.relation_loss_fn = nn.CrossEntropyLoss()
        self.sentiment_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def pool_span(self, token_embeddings, start_tok, end_tok):
        """Mean-pool contextual token representations over span [start_tok, end_tok]."""
        span_reps = token_embeddings[start_tok : end_tok + 1]
        return span_reps.mean(dim=0)

    def construct_pair_rep(self, h_asp, h_opn):
        """[h_asp ; h_opn ; |h_asp - h_opn| ; h_asp * h_opn]"""
        diff = torch.abs(h_asp - h_opn)
        prod = h_asp * h_opn
        return torch.cat([h_asp, h_opn, diff, prod], dim=-1)

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

        if embeddings is not None:
            sequence_output = embeddings.to(torch.float32)
        else:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state

        sequence_output = self.dropout(sequence_output)

        aspect_logits = self.aspect_classifier(sequence_output)
        opinion_logits = self.opinion_classifier(sequence_output)

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

        if aspect_labels is not None and opinion_labels is not None:
            loss_aspect = self.bio_loss_fn(
                aspect_logits.view(-1, config.NUM_ASPECT_LABELS),
                aspect_labels.view(-1)
            )
            loss_opinion = self.bio_loss_fn(
                opinion_logits.view(-1, config.NUM_OPINION_LABELS),
                opinion_labels.view(-1)
            )

            if relation_logits is not None and relation_labels is not None:
                loss_relation = self.relation_loss_fn(relation_logits, relation_labels)
            else:
                loss_relation = torch.tensor(0.0, device=input_ids.device)

            if sentiment_logits is not None and sentiment_labels is not None:
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
            h_pair = self.construct_pair_rep(h_asp, h_opn)
            pair_vectors.append(h_pair)

        pair_reps = torch.stack(pair_vectors, dim=0)
        rel_logits = self.relation_classifier(pair_reps)
        sent_logits = self.sentiment_classifier(pair_reps)
        return rel_logits, sent_logits
