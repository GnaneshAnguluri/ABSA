# End-to-End Aspect Sentiment Triplet Extraction (ASTE) with Non-Linear Deep Interaction Classifiers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![HuggingFace Transformers](https://img.shields.io/badge/Transformers-4.35+-yellow.svg)](https://huggingface.co/)
[![Branch](https://img.shields.io/badge/branch-Gnanesh-success.svg)]()

> **Project Author:** Gnanesh Anguluri  
> **Task:** End-to-End Aspect Sentiment Triplet Extraction (ASTE) on SemEval Benchmark Datasets

---

## 1. Project Overview

Aspect Sentiment Triplet Extraction (ASTE) is the most fine-grained task in Aspect-Based Sentiment Analysis (ABSA). Instead of predicting a single coarse sentiment for a sentence or detecting aspect terms in isolation, ASTE requires jointly extracting fine-grained structured triplets:

$$\text{Triplet} = \big(\text{Aspect Term},\, \text{Opinion Term},\, \text{Sentiment Polarity}\big)$$

### Example
> *"The **food** was **exceptional**, but the **service** was **terrible**."*  
> **Extracted Triplets:**  
> 1. `("food", "exceptional", POSITIVE)`  
> 2. `("service", "terrible", NEGATIVE)`

Evaluation strictly enforces the **Strict Exact Match** standard: an extracted triplet is considered correct if and only if the aspect boundary, opinion boundary, and sentiment polarity all match the ground-truth annotations simultaneously.

---

## 2. Model Architecture: `ASTEModelV2`

Our final high-performance model, **`ASTEModelV2`**, resolves the representational limitations of standard linear classification heads through deep non-linear multi-layer perceptrons and explicit span-interaction vector engineering:

```text
                             Input Review Sentence
                                      │
                                      ▼
                        Fine-Tuned BERT Backbone
                           (bert-base-uncased)
                                      │
                         Hidden States h_t ∈ R^768
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
      2-Layer Aspect MLP Head               2-Layer Opinion MLP Head
        Linear(768 -> 512)                    Linear(768 -> 512)
      LayerNorm + GELU + Drop               LayerNorm + GELU + Drop
        Linear(512 -> 3)                      Linear(512 -> 3)
      [O, B-ASP, I-ASP]                     [O, B-OPN, I-OPN]
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                           Span Boundary Detection
                           Mean Pooling Span Tokens
                          h_asp ∈ R^768, h_opn ∈ R^768
                                      │
                                      ▼
                   3,072-Dimensional Interaction Feature
          [ h_asp || h_opn || |h_asp - h_opn| || (h_asp ⊙ h_opn) ]
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        Deep Relation Classifier              Deep Sentiment Classifier
          Linear(3072 -> 512)                   Linear(3072 -> 512)
        LayerNorm + GELU + Drop               LayerNorm + GELU + Drop
          Linear(512 -> 2)                      Linear(512 -> 3)
         [INVALID: 0, VALID: 1]                [NEG: 0, NEU: 1, POS: 2]
```

### Key Architectural Enhancements
1. **2-Layer MLP Boundary Taggers:** Replaced single linear projections with 2-layer MLPs featuring Layer Normalization, GELU activations, and Dropout (0.2), stabilizing boundary predictions for multi-word aspect and opinion spans.
2. **3,072-Dimensional Interaction Feature Vector:**
   $$\mathbf{h}_{\text{pair}} = \big[\,\mathbf{h}_{\text{asp}} \;\big\|\; \mathbf{h}_{\text{opn}} \;\big\|\; |\mathbf{h}_{\text{asp}} - \mathbf{h}_{\text{opn}}| \;\big\|\; (\mathbf{h}_{\text{asp}} \odot \mathbf{h}_{\text{opn}})\,\big] \in \mathbb{R}^{3072}$$
   - Concatenation $[\mathbf{h}_{\text{asp}} \,\|\, \mathbf{h}_{\text{opn}}]$ preserves individual semantics.
   - Absolute difference $|\mathbf{h}_{\text{asp}} - \mathbf{h}_{\text{opn}}|$ captures contextual distance.
   - Hadamard product $(\mathbf{h}_{\text{asp}} \odot \mathbf{h}_{\text{opn}})$ models fine-grained semantic interaction.
3. **Deep Non-Linear Classifiers:** Both relation validity and sentiment polarity classifiers leverage projection to 512 dimensions, Layer Normalization, and GELU non-linearities.

---

## 3. Experimental Milestones & Results

We progressed through three distinct experimental phases:

### Milestone Progression Overview

| Stage | Model Configuration | Precision | Recall | Strict F1 | Extracted Gold Triplets |
|:---|:---|:---:|:---:|:---:|:---:|
| **Phase 1** | Frozen `bert-base-uncased` + Linear Heads | ~37.2% | ~38.5% | **~38.0%** | ~930 / 2,536 |
| **Phase 2** | Full Corpus Fine-Tuned BERT + Linear Heads | 68.70% | 73.66% | **71.09%** | 1,868 / 2,536 |
| **Phase 3 (Final)** | **Fine-Tuned BERT + 3072-dim Non-Linear MLPs** | **70.69%** | **77.13%** | **73.77%** | **1,956 / 2,536** |

---

### Detailed Benchmark Results (Phase 3: `ASTEModelV2`)

Evaluated on all four canonical SemEval benchmark test splits under strict exact match:

| Benchmark Dataset | Domain | Precision | Recall | Strict F1 | Gold Triplets Extracted | Phase 2 -> Phase 3 Gain |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **SemEval 2014 Restaurant (`14res`)** | Restaurant | **71.10%** | **77.97%** | **74.38%** | **775 / 994** | **+8.01% F1** |
| **SemEval 2014 Laptop (`14lap`)** | Hardware/Laptop | **77.05%** | **71.09%** | **73.95%** | **386 / 543** | +3.44% Prec. |
| **SemEval 2015 Restaurant (`15res`)** | Restaurant | **68.28%** | **78.56%** | **73.06%** | **381 / 485** | +3.71% Recall |
| **SemEval 2016 Restaurant (`16res`)** | Restaurant | **66.35%** | **80.54%** | **72.76%** | **414 / 514** | **+0.99% F1** |
| **Overall Micro-Average** | **Multi-Domain** | **70.69%** | **77.13%** | **73.77%** | **1,956 / 2,536** | **+2.68% F1** |

> Complete experimental analysis, encoder comparison details, and ablation studies are available in [experiments_report.md](experiments_report.md) and the accompanying [experiments_report.pdf](experiments_report.pdf).

---

## 4. Repository Structure

```text
ABSA/
├── config.py                     # Global hyperparameters, label maps, and settings
├── requirements.txt              # Environment dependencies
├── README.md                     # Comprehensive project documentation
├── experiments_report.md         # Full technical experiments report
├── experiments_report.pdf        # Formatted presentation-ready PDF report
│
├── data/
│   ├── raw/                      # Raw ASTE SemEval datasets (14res, 14lap, 15res, 16res)
│   └── processed/                # Tokenized & BIO-tagged binary datasets (git-ignored)
│
├── src/
│   ├── model_v2.py               # ASTEModelV2 (3072-dim Non-Linear Interaction MLP Architecture)
│   ├── model.py                  # Baseline ASTEModel with linear heads
│   ├── evaluate.py               # Strict exact-match Triplet Precision, Recall, and F1 evaluator
│   ├── dataloader.py             # PyTorch Dataset and dynamic collator
│   ├── preprocess.py             # Sentence BIO tagging and candidate span pairing
│   ├── prepare_combined_data.py  # Multi-domain dataset merger (14res, 14lap, 15res, 16res)
│   ├── train_v2_classifiers.py   # Training script for ASTEModelV2
│   ├── resume_v2_training.py     # Checkpoint resumption & high-epoch fine-tuning
│   ├── train_benchmark_max.py    # Phase 2 full benchmark corpus training
│   ├── train_combined_phase2.py  # Cross-domain training script
│   ├── compare_encoders.py       # Encoder evaluation script (BERT vs. RoBERTa vs. BERT-XD)
│   ├── fast_sweep.py             # High-speed in-memory threshold optimizer
│   ├── make_pdf.py               # Headless report compilation utility
│   └── utils.py                  # Seeding, device allocation, and helpers
│
└── checkpoints/                  # Saved PyTorch model checkpoints (git-ignored)
```

---

## 5. Quick Start & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone -b Gnanesh https://github.com/GnaneshAnguluri/ABSA.git
cd ABSA
pip install -r requirements.txt
```

### 2. Data Preparation
Prepare the combined multi-domain benchmark dataset across all 4 SemEval splits:
```bash
python src/prepare_combined_data.py
```

### 3. Evaluate the Best Model (`ASTEModelV2`)
Run strict exact-match evaluation on the test split:
```bash
python -c "import torch; from src.evaluate import evaluate_triplets; from src.dataloader import get_aste_dataloaders; from src.model_v2 import ASTEModelV2; device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); _, _, test_loader = get_aste_dataloaders('data/processed/combined', batch_size=8); model = ASTEModelV2(); checkpoint = torch.load('checkpoints/benchmark_v2_mlp.pt', map_location=device); model.load_state_dict(checkpoint['model_state_dict']); model.to(device); metrics = evaluate_triplets(model, test_loader, device); print(metrics)"
```

### 4. Train `ASTEModelV2`
To train the non-linear interaction model:
```bash
python src/train_v2_classifiers.py
```

---

## 6. Citations & References
- SemEval ASTE Datasets: Peng et al. (2020), Xu et al. (2020).
- BERT Backbone: Devlin et al. (2019) *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.
