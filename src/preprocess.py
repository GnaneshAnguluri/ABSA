"""
preprocess.py - Step 3: Sentence-level ASTE Preprocessing & Candidate Pairing.

Key changes:
1. Sentence-level (one sample per review sentence, not duplicated per triplet).
2. Maintains complete sentence BIO tags for ALL aspects and opinions.
3. Generates all (Aspect, Opinion) candidate pairs:
   - Positive pairs (in gold triplets) -> relation = 1 (VALID), sentiment in {0, 1, 2}
   - Negative pairs (not in gold triplets) -> relation = 0 (INVALID), sentiment = -100 (ignored in loss)
4. Preserves word-to-subword alignment with word_ids() and -100 masking.
"""

import os
import sys
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.dataset import load_aste_dataset

IGNORE_INDEX = -100


def build_sentence_bio_tags(words: list, triples: list):
    """
    Construct sentence-level BIO tags across all triplets in the sentence.

    Args:
        words   : list of word tokens
        triples : list of {'aspect_span': [s, e], 'opinion_span': [s, e], 'sentiment': str}

    Returns:
        word_asp_tags: list of str ('O', 'B-ASP', 'I-ASP')
        word_opn_tags: list of str ('O', 'B-OPN', 'I-OPN')
        unique_aspects: list of [start, end]
        unique_opinions: list of [start, end]
    """
    n = len(words)
    word_asp_tags = ["O"] * n
    word_opn_tags = ["O"] * n

    unique_aspects = []
    unique_opinions = []

    for t in triples:
        a_s, a_e = t["aspect_span"]
        o_s, o_e = t["opinion_span"]

        if [a_s, a_e] not in unique_aspects:
            unique_aspects.append([a_s, a_e])
        if [o_s, o_e] not in unique_opinions:
            unique_opinions.append([o_s, o_e])

        # Tag aspect span
        if 0 <= a_s < n and 0 <= a_e < n:
            word_asp_tags[a_s] = "B-ASP"
            for i in range(a_s + 1, a_e + 1):
                word_asp_tags[i] = "I-ASP"

        # Tag opinion span
        if 0 <= o_s < n and 0 <= o_e < n:
            word_opn_tags[o_s] = "B-OPN"
            for i in range(o_s + 1, o_e + 1):
                word_opn_tags[i] = "I-OPN"

    return word_asp_tags, word_opn_tags, unique_aspects, unique_opinions


def encode_sentence_sample(sample: dict, tokenizer, max_len: int = config.MAX_SEQ_LEN) -> dict:
    """
    Encode a single review sentence into ASTE features with pair-level targets.
    """
    words   = sample["words"]
    triples = sample["triples"]

    # 1. Build word-level BIO tags and unique spans
    word_asp_tags, word_opn_tags, unique_aspects, unique_opinions = build_sentence_bio_tags(words, triples)

    # 2. Tokenize with BERT WordPiece
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids      = encoding["input_ids"].squeeze(0)
    attention_mask = encoding["attention_mask"].squeeze(0)
    word_ids       = encoding.word_ids(batch_index=0)

    # Map word index to subword token span (start_token, end_token)
    word_to_token_spans = {}
    for tok_idx, w_id in enumerate(word_ids):
        if w_id is not None:
            if w_id not in word_to_token_spans:
                word_to_token_spans[w_id] = [tok_idx, tok_idx]
            else:
                word_to_token_spans[w_id][1] = tok_idx

    # 3. Align BIO tags to subwords
    asp_label_ids = []
    opn_label_ids = []
    prev_word_id  = None

    for word_id in word_ids:
        if word_id is None:
            asp_label_ids.append(IGNORE_INDEX)
            opn_label_ids.append(IGNORE_INDEX)
        elif word_id != prev_word_id:
            # First subword
            asp_label_ids.append(config.ASPECT_TAGS[word_asp_tags[word_id]])
            opn_label_ids.append(config.OPINION_TAGS[word_opn_tags[word_id]])
        else:
            # Continuation subword
            asp_label_ids.append(IGNORE_INDEX)
            opn_label_ids.append(IGNORE_INDEX)
        prev_word_id = word_id

    # 4. Build Candidate Pairs: Cartesian product of all aspects x opinions
    # Map gold triplets for O(1) lookup
    gold_triplets_dict = {}
    for t in triples:
        key = (tuple(t["aspect_span"]), tuple(t["opinion_span"]))
        gold_triplets_dict[key] = config.SENTIMENT_MAP[t["sentiment"]]

    candidate_pairs = []
    for asp in unique_aspects:
        # Check if aspect words are within tokenizer range
        if asp[0] not in word_to_token_spans or asp[1] not in word_to_token_spans:
            continue
        asp_tok_span = [word_to_token_spans[asp[0]][0], word_to_token_spans[asp[1]][1]]

        for opn in unique_opinions:
            if opn[0] not in word_to_token_spans or opn[1] not in word_to_token_spans:
                continue
            opn_tok_span = [word_to_token_spans[opn[0]][0], word_to_token_spans[opn[1]][1]]

            pair_key = (tuple(asp), tuple(opn))
            if pair_key in gold_triplets_dict:
                relation_label  = 1   # VALID
                sentiment_label = gold_triplets_dict[pair_key]
            else:
                relation_label  = 0   # INVALID
                sentiment_label = IGNORE_INDEX  # Ignored in sentiment loss

            candidate_pairs.append({
                "aspect_word_span" : asp,
                "opinion_word_span": opn,
                "aspect_tok_span"  : asp_tok_span,
                "opinion_tok_span" : opn_tok_span,
                "relation_label"   : relation_label,
                "sentiment_label"  : sentiment_label
            })

    return {
        "input_ids"      : input_ids,
        "attention_mask" : attention_mask,
        "aspect_labels"  : torch.tensor(asp_label_ids, dtype=torch.long),
        "opinion_labels" : torch.tensor(opn_label_ids, dtype=torch.long),
        "candidate_pairs": candidate_pairs,
        "sentence"       : sample["sentence"],
        "words"          : words,
        "gold_triples"   : triples
    }


def preprocess_split(data: list, tokenizer, split_name: str = "train") -> list:
    """Process an entire dataset split at sentence level."""
    features = []
    skipped  = 0
    total_pairs = 0
    valid_pairs = 0

    for sample in data:
        try:
            feat = encode_sentence_sample(sample, tokenizer)
            features.append(feat)
            for cp in feat["candidate_pairs"]:
                total_pairs += 1
                if cp["relation_label"] == 1:
                    valid_pairs += 1
        except Exception as e:
            skipped += 1

    print(f"[preprocess] {split_name:5s} -> {len(features)} sentences (skipped={skipped})")
    print(f"             Total candidate pairs: {total_pairs} | Valid: {valid_pairs} | Invalid: {total_pairs - valid_pairs}")
    return features


def save_features(features: list, path: str):
    """Save processed features using torch.save."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(features, path)
    print(f"[preprocess] Saved -> {path}")


def load_features(path: str) -> list:
    """Load pre-processed features."""
    return torch.load(path, weights_only=False)


if __name__ == "__main__":
    print(f"[preprocess] Loading tokenizer: {config.ENCODER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(config.ENCODER_MODEL)

    train_raw, dev_raw, test_raw = load_aste_dataset(config.DATASET_SPLIT)

    print("\n[preprocess] Processing ASTE sentences & generating candidate pairs...")
    train_feats = preprocess_split(train_raw, tokenizer, "train")
    dev_feats   = preprocess_split(dev_raw,   tokenizer, "dev")
    test_feats  = preprocess_split(test_raw,  tokenizer, "test")

    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    split = config.DATASET_SPLIT
    save_features(train_feats, os.path.join(config.PROCESSED_DIR, f"{split}_train_aste.pt"))
    save_features(dev_feats,   os.path.join(config.PROCESSED_DIR, f"{split}_dev_aste.pt"))
    save_features(test_feats,  os.path.join(config.PROCESSED_DIR, f"{split}_test_aste.pt"))

    # Inspect first multi-pair sentence
    for f in train_feats:
        if len(f["candidate_pairs"]) > 1:
            print("\n" + "="*70)
            print("SAMPLE INSPECTION (Sentence with multiple candidate pairs):")
            print("="*70)
            print(f"Sentence: {f['sentence']}")
            print(f"Number of candidate pairs: {len(f['candidate_pairs'])}")
            for idx, p in enumerate(f["candidate_pairs"]):
                print(f" Pair {idx}: Aspect={p['aspect_word_span']} | Opinion={p['opinion_word_span']} | "
                      f"Relation={'VALID (1)' if p['relation_label'] == 1 else 'INVALID (0)'} | "
                      f"Sentiment={config.ID2SENTIMENT.get(p['sentiment_label'], 'IGNORED (-100)')}")
            break

    print("\n[preprocess] ASTE Preprocessing Complete!")
