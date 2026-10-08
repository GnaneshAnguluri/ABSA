# Context-Aware End-to-End ABSA / ASTE Pipeline
### Branch: `Gnanesh` | Contributor: Gnanesh Anguluri

---

## 1. Project Overview & ASTE Architecture

This project implements an **End-to-End Aspect Sentiment Triplet Extraction (ASTE)** pipeline. Unlike traditional sequence classification that assigns a single sentiment to an entire review sentence, ASTE extracts fine-grained, structured sentiment triplets:
$$\text{(Aspect Term, Opinion Term, Sentiment Polarity)}$$

Example:
> *"The food was exceptional, but the service was terrible."*  
> **Output Triplets:**  
> `[("food", "exceptional", POS), ("service", "terrible", NEG)]`

---

## 2. Model Architecture

```text
                     Input Review Text
                            │
                            ▼
               BERT Encoder (bert-base-uncased) ❄️
                  [Phase 1: requires_grad=False]
                            │
              Contextual Embeddings [SeqLen, 768]
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    Aspect BIO Classifier       Opinion BIO Classifier
        Linear(768, 3)              Linear(768, 3)
      [O, B-ASP, I-ASP]           [O, B-OPN, I-OPN]
              │                           │
              └─────────────┬─────────────┘
                            ▼
                  Candidate Pair Pooling
                    h_aspect  ∈ R^768
                    h_opinion ∈ R^768
                            │
                            ▼
               Concatenated Pair Representation
               [h_aspect ; h_opinion] ∈ R^1536
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   Pair Relation Classifier    Pair Sentiment Classifier
        Linear(1536, 2)             Linear(1536, 3)
      [INVALID: 0, VALID: 1]       [NEG: 0, NEU: 1, POS: 2]
```

---

## 3. Key Design Choices & Upgrades

### A. Phase 1: Frozen BERT (`requires_grad=False`)
- The pre-trained `bert-base-uncased` backbone parameters are frozen in Phase 1 (`FREEZE_ENCODER = True`).
- Only ASTE-specific heads (Aspect Linear, Opinion Linear, Relation Classifier, Sentiment Classifier) are trained.
- Phase 2 will unfreeze BERT for end-to-end fine-tuning to compare performance.

### B. Candidate Pairing & 1536-Dimensional Pair Representations
- For each sentence, all detected/gold aspect spans and opinion spans are combined into Cartesian candidate pairs:
  $$\{(A_1, O_1), (A_1, O_2), (A_2, O_1), (A_2, O_2), \dots\}$$
- Token representations across each span are mean-pooled into $h_{\text{aspect}} \in \mathbb{R}^{768}$ and $h_{\text{opinion}} \in \mathbb{R}^{768}$.
- Concatenated vector: $[h_{\text{aspect}}; h_{\text{opinion}}] \in \mathbb{R}^{1536}$.

### C. Binary Pair Relation Classifier
- A trainable linear classifier maps $1536 \rightarrow 2$:
  - `VALID (1)`: The pair is a true opinion-aspect association.
  - `INVALID (0)`: The pair is an incorrect combination.

### D. Pair-Level Sentiment Classifier
- A trainable linear classifier maps $1536 \rightarrow 3$:
  - Classes: `NEG (0)`, `NEU (1)`, `POS (2)`.
  - Pairs marked as `INVALID` have sentiment label `-100` (ignored during sentiment cross-entropy calculation).

### E. Multi-Task Loss Function
The model optimizes all subtasks jointly:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{asp}} \mathcal{L}_{\text{asp}} + \lambda_{\text{opn}} \mathcal{L}_{\text{opn}} + \lambda_{\text{rel}} \mathcal{L}_{\text{rel}} + \lambda_{\text{sent}} \mathcal{L}_{\text{sent}}$$
Initially balanced with equal weights:
$$\lambda_{\text{asp}} = 1.0, \quad \lambda_{\text{opn}} = 1.0, \quad \lambda_{\text{rel}} = 1.0, \quad \lambda_{\text{sent}} = 1.0$$

### F. Strict ASTE Triplet Evaluation Metric
Performance is evaluated on **full triplet matching**:
- A predicted triplet $(A, O, S)$ is correct if and only if:
  1. Aspect word span matches gold exactly
  2. Opinion word span matches gold exactly
  3. Sentiment polarity matches gold exactly
- Metrics computed: **Precision**, **Recall**, and **F1-Score**.

---

## 4. File Structure

```text
ABSA/
├── config.py             # Hyperparameters, tag IDs, FREEZE_ENCODER flag
├── requirements.txt      # PyTorch, transformers, etc.
├── README.md             # Project documentation
│
└── src/
    ├── __init__.py
    ├── utils.py          # Random seed & device utilities
    ├── dataset.py        # Ingestion & parsing of ASTE-Data-V2
    ├── preprocess.py     # Sentence-level BIO tagging & candidate pair generation
    ├── dataloader.py     # PyTorch Dataset & dynamic batch collate_fn
    ├── model.py          # ASTEModel (Frozen BERT, BIO heads, relation & sentiment heads)
    ├── evaluate.py       # Strict Triplet Precision, Recall, F1 evaluation
    └── train.py          # Multi-task training loop & checkpointing
```

---

## 5. How to Run

### Step 1: Preprocess Data
```bash
python src/preprocess.py
```

## 6. Experimental Results

### Phase 1: Frozen BERT Baseline (`FREEZE_ENCODER = True`)
- **Backbone:** `bert-base-uncased` (Frozen, zero gradient updates)
- **Trainable Parameters:** Only ASTE-specific linear heads
- **Hardware:** Local Intel CPU (Optimized with float16 cached token embeddings)
- **Epochs:** 5 | **Batch Size:** 8 | **Learning Rate:** 1e-3

| Metric | Dev Set | Test Set |
|---|---|---|
| **Strict Triplet Precision** | 37.39% | **37.65%** |
| **Strict Triplet Recall** | 38.30% | **40.34%** |
| **Strict Triplet F1-Score** | **37.84%** | **38.95%** |
| **Correct Triplets Extracted** | 221 / 577 | **401 / 994** |

> Note: All metrics use strict exact triplet matching on predicted spans (no gold span leakage). Phase 2 will unfreeze BERT for end-to-end fine-tuning to compare performance improvements.