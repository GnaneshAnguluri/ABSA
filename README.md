# ABSA Pipeline — Data Pipeline Module
### Branch: `Gnanesh` | Contributor: Gnanesh Anguluri

---

## What This Branch Contains

This branch contains the **complete data pipeline** for an Aspect-Based Sentiment Analysis (ABSA) system — specifically the **Aspect Sentiment Triplet Extraction (ASTE)** task.

This is the foundational 1/3 of the full pipeline, responsible for:
- Acquiring and loading the dataset
- Converting raw data into model-ready tensors
- Batching data for training

---

## Project Architecture

```
Input Review Text
       │
       ▼
  dataset.py      ← Download & parse ASTE-Data-V2 from GitHub
       │
       ▼
  preprocess.py   ← BIO tag conversion + BERT subword alignment
       │
       ▼
  dataloader.py   ← Batch tensors into PyTorch DataLoaders
       │
       ▼
   [ Model ]      ← (Steps 5–10, separate branches)
```

---

## Files in This Branch

```
ABSA/
├── config.py              ← All hyperparameters & settings
├── requirements.txt       ← Python dependencies
├── README.md              ← This file
└── src/
    ├── __init__.py
    ├── utils.py           ← Seed & device helpers
    ├── dataset.py         ← Dataset loading
    ├── preprocess.py      ← BIO tag conversion
    └── dataloader.py      ← PyTorch DataLoader
```

---

## Dataset — ASTE-Data-V2 (14res)

**Source:** [xuuuluuu/SemEval-Triplet-data](https://github.com/xuuuluuu/SemEval-Triplet-data)  
**Domain:** Restaurant reviews (SemEval 2014)  
**Task:** Aspect Sentiment Triplet Extraction

Each sample in the dataset is a review sentence annotated with triplets:

```
Sentence : "The food was great but the service was terrible."
Triplets : [
    (food,    great,    POS),   ← aspect + opinion + sentiment
    (service, terrible, NEG)
]
```

### Dataset Statistics (14res split)

| Split | Sentences | Triplets |
|---|---|---|
| Train | 1,266 | 2,338 |
| Dev   | 310   | 577   |
| Test  | 492   | 994   |

**Sentiment distribution (train):**
- POS: 72.4%
- NEG: 20.5%
- NEU: 7.1%

---

## Module Details

### `config.py` — Central Configuration

All hyperparameters defined in one place:

```python
ENCODER_MODEL = "bert-base-uncased"
MAX_SEQ_LEN   = 128
BATCH_SIZE    = 16
LEARNING_RATE = 2e-5
EPOCHS        = 20

ASPECT_TAGS   = {"O": 0, "B-ASP": 1, "I-ASP": 2}
OPINION_TAGS  = {"O": 0, "B-OPN": 1, "I-OPN": 2}
SENTIMENT_MAP = {"NEG": 0, "NEU": 1, "POS": 2}
```

---

### `src/dataset.py` — Dataset Loading

Downloads the ASTE-Data-V2 `.txt` files directly from GitHub
(avoids the `pyarrow`/HuggingFace datasets dependency).

**Raw file format:**
```
But the staff was so horrible to us .####[([2], [5], 'NEG')]
```
- Left of `####` → the sentence
- Right of `####` → list of triplets as `(aspect_indices, opinion_indices, sentiment)`

**Key functions:**
- `download_raw_files()` — fetches train/dev/test `.txt` files
- `parse_file()` — parses each line into structured dicts
- `load_aste_dataset()` — full pipeline, returns train/dev/test lists
- `compute_statistics()` — dataset analysis
- `save_raw_csv()` — saves human-readable CSVs

---

### `src/preprocess.py` — BIO Tag Conversion

This is the most critical module. It converts word-level span annotations into subword-level BIO label tensors aligned with BERT's WordPiece tokenizer.

**The BIO Tagging Scheme:**
```
O     → this token is not part of an aspect or opinion
B-ASP → this token BEGINS an aspect term
I-ASP → this token is INSIDE (continues) an aspect term
B-OPN → this token BEGINS an opinion term
I-OPN → this token is INSIDE (continues) an opinion term
```

**The Subword Alignment Problem:**

BERT splits words into subwords. Labels must be aligned carefully:

```
Words :  [ The,  battery,  life,   is,  great  ]
         aspect span = [1, 2]        opinion span = [4, 4]

BERT  :  [CLS], The, bat, ##tery, life, is, great, [SEP], [PAD]...
word_id: [None,  0,   1,    1,     2,   3,    4,   None,  None]

Aspect:  [ -100,  O, B-ASP, -100, I-ASP, O,   O,   -100,  -100]
Opinion: [ -100,  O,  O,    -100,   O,   O,  B-OPN, -100, -100]
```

**Rules applied:**
- `[CLS]`, `[SEP]`, `[PAD]` → `-100` (ignored in loss)
- First subword of a word → gets the actual BIO tag
- Remaining subwords (`##tery`) → `-100` (ignored in loss)

**Why `-100`?** PyTorch's `CrossEntropyLoss` ignores positions labelled `-100` automatically. This means subword tokens never contribute to the loss — only the first subword of each word does.

**Key functions:**
- `words_to_bio_tags()` — builds word-level BIO sequences from spans
- `align_tags_to_subwords()` — aligns word tags to BERT subword tokens using `word_ids()`
- `encode_sample()` — encodes one full sample into tensors
- `preprocess_split()` — processes all samples in a split
- `save_features()` / `load_features()` — persist processed tensors

**Output tensor shapes (per feature):**
```
input_ids       : [128]   ← BERT token IDs
attention_mask  : [128]   ← 1=real token, 0=padding
aspect_labels   : [128]   ← BIO tag IDs for aspects
opinion_labels  : [128]   ← BIO tag IDs for opinions
sentiment_label : int     ← 0=NEG, 1=NEU, 2=POS
```

---

### `src/dataloader.py` — PyTorch DataLoader

Wraps processed features into a PyTorch `Dataset` and returns
`DataLoader` objects for model training.

```python
train_loader, dev_loader, test_loader = get_dataloaders()
```

**Batch shape:**
```
input_ids       : [16, 128]
attention_mask  : [16, 128]
aspect_labels   : [16, 128]
opinion_labels  : [16, 128]
sentiment_label : [16]
```

| Loader | Batches | Shuffle |
|---|---|---|
| Train | 147 | ✅ Yes |
| Dev   | 37  | ❌ No  |
| Test  | 63  | ❌ No  |

> Train is shuffled to prevent the model memorising example order.
> Dev/Test are never shuffled so evaluation is reproducible.

---

## How to Run

### 1. Install dependencies
```bash
pip install torch transformers pandas tqdm scikit-learn matplotlib seaborn tensorboard
```

### 2. Download & explore dataset
```bash
python src/dataset.py
```

### 3. Preprocess (BIO tagging + tokenization)
```bash
python src/preprocess.py
```

### 4. Verify DataLoader
```bash
python src/dataloader.py
```

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| ASTE-Data-V2 over SemEval-2014 alone | ASTE has opinion spans — needed for the opinion extractor head |
| `bert-base-uncased` | Strong baseline, widely benchmarked, fits on CPU |
| `MAX_SEQ_LEN = 128` | 99%+ of restaurant reviews fit in 128 tokens |
| `BATCH_SIZE = 16` | Balances memory usage and gradient stability |
| Direct GitHub download | Avoids `pyarrow` DLL conflict on Windows |
| `-100` for subwords | PyTorch CrossEntropy natively ignores this index |

---

## Dependencies

```
torch>=2.0.0
transformers>=4.35.0
pandas>=2.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
tqdm>=4.65.0
```

---

*Part of the ABSA minor project — Context-Aware End-to-End Pipeline using BERT/RoBERTa*