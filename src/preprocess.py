"""
preprocess.py - Step 3: Convert raw ASTE samples into BERT-tokenized,
                BIO-tagged tensors ready for the model.

The core challenge:
  Word-level:   [ The,   food,  was,  great ]
  BERT tokens:  [ [CLS], The,   food, was,   great, [SEP] ]
  BIO tags:     [ -100,  O,     B-ASP, O,    O,     -100  ]

Multi-word aspects / sub-word tokens:
  Word:         [ battery,  life ]        <- aspect span [0,1]
  BERT tokens:  [ bat, ##tery, life ]
  BIO tags:     [ B-ASP, -100, I-ASP ]   <- only FIRST subword per word gets label
                                             remaining subwords get -100 (ignored in loss)
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import load_aste_dataset


# ─────────────────────────────────────────────────────────────
#  TAG ID MAPPINGS  (from config)
# ─────────────────────────────────────────────────────────────

ASPECT_TAG2ID  = config.ASPECT_TAGS    # {"O":0, "B-ASP":1, "I-ASP":2}
OPINION_TAG2ID = config.OPINION_TAGS   # {"O":0, "B-OPN":1, "I-OPN":2}
SENTIMENT2ID   = config.SENTIMENT_MAP  # {"NEG":0, "NEU":1, "POS":2}

IGNORE_INDEX   = -100   # PyTorch CrossEntropyLoss ignores this index


# ─────────────────────────────────────────────────────────────
#  CORE BIO CONVERSION — word level -> token level
# ─────────────────────────────────────────────────────────────

def words_to_bio_tags(words: list, asp_span: list, opn_span: list):
    """
    Build word-level BIO tag sequences for aspect and opinion.

    Args:
        words    : list of str  e.g. ['The', 'food', 'was', 'great']
        asp_span : [start, end] inclusive  e.g. [1, 1]
        opn_span : [start, end] inclusive  e.g. [3, 3]

    Returns:
        aspect_tags  : list of str  e.g. ['O', 'B-ASP', 'O', 'O']
        opinion_tags : list of str  e.g. ['O', 'O', 'O', 'B-OPN']
    """
    n = len(words)
    asp_start, asp_end = asp_span
    opn_start, opn_end = opn_span

    aspect_tags  = []
    opinion_tags = []

    for i in range(n):
        # ── Aspect BIO ──
        if i == asp_start:
            aspect_tags.append("B-ASP")
        elif asp_start < i <= asp_end:
            aspect_tags.append("I-ASP")
        else:
            aspect_tags.append("O")

        # ── Opinion BIO ──
        if i == opn_start:
            opinion_tags.append("B-OPN")
        elif opn_start < i <= opn_end:
            opinion_tags.append("I-OPN")
        else:
            opinion_tags.append("O")

    return aspect_tags, opinion_tags


def align_tags_to_subwords(words: list, word_tags: list, tokenizer) -> list:
    """
    Map word-level BIO tags to subword-level token tags.

    Rule:
      - [CLS], [SEP]         -> IGNORE_INDEX
      - First subword of word -> word's tag (as ID)
      - Other subwords        -> IGNORE_INDEX  (not penalised in loss)

    Args:
        words     : list of str (original words)
        word_tags : list of str (BIO tags per word)
        tokenizer : HuggingFace tokenizer

    Returns:
        token_tag_ids : list of int (one per token including CLS/SEP)
    """
    token_tag_ids = [IGNORE_INDEX]   # for [CLS]

    for word, tag in zip(words, word_tags):
        subwords = tokenizer.tokenize(word)
        if len(subwords) == 0:
            continue   # rare edge case (unknown character)

        # First subword -> actual tag
        if tag in ASPECT_TAG2ID:
            token_tag_ids.append(ASPECT_TAG2ID[tag])
        else:
            token_tag_ids.append(ASPECT_TAG2ID["O"])   # fallback

        # Remaining subwords -> IGNORE
        for _ in subwords[1:]:
            token_tag_ids.append(IGNORE_INDEX)

    token_tag_ids.append(IGNORE_INDEX)   # for [SEP]
    return token_tag_ids


# ─────────────────────────────────────────────────────────────
#  ENCODE A SINGLE SAMPLE
# ─────────────────────────────────────────────────────────────

def encode_sample(sample: dict, tokenizer, max_len: int = config.MAX_SEQ_LEN) -> list:
    """
    Convert ONE raw sample into a list of encoded feature dicts
    (one dict per triplet in the sample).

    Each feature dict contains:
        input_ids       : tensor [max_len]       - BERT token IDs
        attention_mask  : tensor [max_len]       - 1 for real tokens, 0 for padding
        aspect_labels   : tensor [max_len]       - BIO tag IDs for aspects
        opinion_labels  : tensor [max_len]       - BIO tag IDs for opinions
        sentiment_label : int                    - 0/1/2 for NEG/NEU/POS
        aspect_span     : [start, end]           - word-level span (for pairing module)
        opinion_span    : [start, end]           - word-level span (for pairing module)
        sentence        : str                    - original sentence (for debugging)

    Args:
        sample  : dict with keys 'words', 'triples', 'sentence'
        tokenizer: HuggingFace tokenizer
        max_len : maximum token length (pad/truncate to this)

    Returns:
        List of feature dicts (one per triplet)
    """
    words   = sample["words"]
    triples = sample["triples"]
    features = []

    for triplet in triples:
        asp_span  = triplet["aspect_span"]   # [start, end]
        opn_span  = triplet["opinion_span"]  # [start, end]
        sentiment = triplet["sentiment"]     # 'POS' | 'NEG' | 'NEU'

        # ── 1. Build word-level BIO tags ──
        word_asp_tags, word_opn_tags = words_to_bio_tags(words, asp_span, opn_span)

        # ── 2. BERT full encoding (handles CLS, SEP, padding, attention mask) ──
        encoding = tokenizer(
            words,
            is_split_into_words=True,      # words already split
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids      = encoding["input_ids"].squeeze(0)       # [max_len]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [max_len]

        # ── 3. Align BIO tags to subword tokens ──
        # We build manually using word_ids() from the encoding
        word_ids = encoding.word_ids(batch_index=0)
        # word_ids: list of ints or None (None = CLS, SEP, PAD)
        #   e.g. [None, 0, 1, 1, 2, 3, None, None, None ...]

        asp_label_ids = []
        opn_label_ids = []
        prev_word_id  = None

        for word_id in word_ids:
            if word_id is None:
                # CLS, SEP, or PAD token -> ignore
                asp_label_ids.append(IGNORE_INDEX)
                opn_label_ids.append(IGNORE_INDEX)
            elif word_id != prev_word_id:
                # First subword of this word -> use the actual tag
                asp_label_ids.append(ASPECT_TAG2ID[word_asp_tags[word_id]])
                opn_label_ids.append(OPINION_TAG2ID[word_opn_tags[word_id]])
            else:
                # Subsequent subword -> ignore in loss
                asp_label_ids.append(IGNORE_INDEX)
                opn_label_ids.append(IGNORE_INDEX)

            prev_word_id = word_id

        # ── 4. Convert to tensors ──
        asp_labels = torch.tensor(asp_label_ids, dtype=torch.long)  # [max_len]
        opn_labels = torch.tensor(opn_label_ids, dtype=torch.long)  # [max_len]
        sent_label = SENTIMENT2ID[sentiment]                          # int

        features.append({
            "input_ids"      : input_ids,
            "attention_mask" : attention_mask,
            "aspect_labels"  : asp_labels,
            "opinion_labels" : opn_labels,
            "sentiment_label": sent_label,
            "aspect_span"    : asp_span,
            "opinion_span"   : opn_span,
            "sentence"       : sample["sentence"],
        })

    return features


# ─────────────────────────────────────────────────────────────
#  PROCESS FULL SPLIT
# ─────────────────────────────────────────────────────────────

def preprocess_split(data: list, tokenizer, split_name: str = "train") -> list:
    """
    Process an entire dataset split (train / dev / test).

    Returns:
        List of all feature dicts across all samples & triplets.
    """
    all_features = []
    skipped      = 0

    for sample in data:
        try:
            feats = encode_sample(sample, tokenizer)
            all_features.extend(feats)
        except Exception as e:
            skipped += 1
            print(f"[preprocess] WARNING: skipped sample - {e}")

    print(f"[preprocess] {split_name:5s} -> {len(all_features)} features "
          f"from {len(data)} sentences  (skipped={skipped})")
    return all_features


# ─────────────────────────────────────────────────────────────
#  SAVE / LOAD PROCESSED FEATURES
# ─────────────────────────────────────────────────────────────

def save_features(features: list, path: str):
    """Save processed features to disk using torch.save."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(features, path)
    print(f"[preprocess] Saved {len(features)} features -> {path}")


def load_features(path: str) -> list:
    """Load pre-processed features from disk."""
    features = torch.load(path, weights_only=False)
    print(f"[preprocess] Loaded {len(features)} features <- {path}")
    return features


# ─────────────────────────────────────────────────────────────
#  DEBUG — Print a single encoded feature
# ─────────────────────────────────────────────────────────────

def inspect_feature(feat: dict, tokenizer):
    """Pretty-print a single encoded feature to verify alignment."""
    tokens    = tokenizer.convert_ids_to_tokens(feat["input_ids"])
    asp_ids   = feat["aspect_labels"].tolist()
    opn_ids   = feat["opinion_labels"].tolist()
    id2asp    = {v: k for k, v in ASPECT_TAG2ID.items()}
    id2opn    = {v: k for k, v in OPINION_TAG2ID.items()}
    id2sent   = config.ID2SENTIMENT

    print(f"\n{'='*80}")
    print(f" Sentence : {feat['sentence']}")
    print(f" Sentiment: {id2sent[feat['sentiment_label']]}")
    print(f" Aspect span  (word-level): {feat['aspect_span']}")
    print(f" Opinion span (word-level): {feat['opinion_span']}")
    print(f"\n {'Token':<18} {'Aspect':>8}  {'Opinion':>8}")
    print(f" {'-'*40}")

    for tok, a, o in zip(tokens, asp_ids, opn_ids):
        a_str = id2asp.get(a, "IGN") if a != IGNORE_INDEX else "IGN"
        o_str = id2opn.get(o, "IGN") if o != IGNORE_INDEX else "IGN"
        marker = " <--" if a_str != "O" and a_str != "IGN" else (
                 " ***" if o_str != "O" and o_str != "IGN" else "")
        print(f" {tok:<18} {a_str:>8}  {o_str:>8} {marker}")
    print(f"{'='*80}")


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Load tokenizer ──
    print(f"\n[preprocess] Loading tokenizer: {config.ENCODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.ENCODER_MODEL)
    print(f"[preprocess] Vocab size: {tokenizer.vocab_size}")

    # ── Load raw data ──
    train_raw, dev_raw, test_raw = load_aste_dataset(config.DATASET_SPLIT)

    # ── Process each split ──
    print("\n[preprocess] Processing splits...")
    train_feats = preprocess_split(train_raw, tokenizer, "train")
    dev_feats   = preprocess_split(dev_raw,   tokenizer, "dev")
    test_feats  = preprocess_split(test_raw,  tokenizer, "test")

    # ── Inspect first 2 features ──
    print("\n[preprocess] Sample feature inspection:")
    inspect_feature(train_feats[0], tokenizer)
    inspect_feature(train_feats[1], tokenizer)

    # ── Save processed features ──
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    split = config.DATASET_SPLIT
    save_features(train_feats, os.path.join(config.PROCESSED_DIR, f"{split}_train.pt"))
    save_features(dev_feats,   os.path.join(config.PROCESSED_DIR, f"{split}_dev.pt"))
    save_features(test_feats,  os.path.join(config.PROCESSED_DIR, f"{split}_test.pt"))

    # ── Tensor shape verification ──
    f = train_feats[0]
    print(f"\n[preprocess] Tensor shapes for one feature:")
    print(f"  input_ids      : {f['input_ids'].shape}")
    print(f"  attention_mask : {f['attention_mask'].shape}")
    print(f"  aspect_labels  : {f['aspect_labels'].shape}")
    print(f"  opinion_labels : {f['opinion_labels'].shape}")
    print(f"  sentiment_label: {f['sentiment_label']}  (int)")

    print("\n[preprocess] Step 3 complete!")
