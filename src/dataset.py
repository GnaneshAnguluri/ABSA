"""
dataset.py — Step 2: Load and explore ASTE-Data-V2 (via direct GitHub download)
No dependency on HuggingFace `datasets` or pyarrow.
"""

import os
import ast
import urllib.request
import pandas as pd
from collections import Counter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─────────────────────────────────────────────────────────────
#  1.  DOWNLOAD RAW FILES FROM GITHUB
# ─────────────────────────────────────────────────────────────

# Base URL for ASTE-Data-V2 on GitHub
_BASE_URL = (
    "https://raw.githubusercontent.com/xuuuluuu/"
    "SemEval-Triplet-data/master/ASTE-Data-V2-EMNLP2020"
)

_SPLIT_FILES = {
    "train": "train_triplets.txt",
    "dev"  : "dev_triplets.txt",
    "test" : "test_triplets.txt",
}


def download_raw_files(dataset_split: str = config.DATASET_SPLIT):
    """
    Download the three raw .txt files (train / dev / test) from GitHub
    into  data/raw/<dataset_split>/
    Skips download if files already exist.
    """
    save_dir = os.path.join(config.DATA_DIR, dataset_split)
    os.makedirs(save_dir, exist_ok=True)

    for split_name, filename in _SPLIT_FILES.items():
        local_path = os.path.join(save_dir, filename)
        if os.path.exists(local_path):
            print(f"[dataset] Already exists -> {local_path}")
            continue

        url = f"{_BASE_URL}/{dataset_split}/{filename}"
        print(f"[dataset] Downloading {url}")
        urllib.request.urlretrieve(url, local_path)
        print(f"[dataset] Saved -> {local_path}")


# ─────────────────────────────────────────────────────────────
#  2.  PARSE .TXT FILES
# ─────────────────────────────────────────────────────────────
#
#  Each line in the raw file looks like:
#
#    But the staff was so rude to us .####[([4, 4], [6, 6], 'NEG')]
#
#  Format:  sentence####[list of triplets]
#  Each triplet: ([aspect_start, aspect_end], [opinion_start, opinion_end], 'POS'|'NEG'|'NEU')
#  Indices are word-level (0-indexed, inclusive)
# ─────────────────────────────────────────────────────────────

def parse_file(filepath: str) -> list:
    """
    Parse a single ASTE .txt file into a list of sample dicts.

    Returns:
        List of dicts, each with keys:
            'sentence' : str   — raw sentence
            'words'    : list  — tokenized words (split on space)
            'triples'  : list  — [(asp_idx, opn_idx, sentiment), ...]
                          asp_idx / opn_idx : [start, end]  (word-level, inclusive)
                          sentiment         : 'POS' | 'NEG' | 'NEU'
    """
    samples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split("####")
            if len(parts) != 2:
                print(f"[dataset] WARNING: malformed line {line_no}, skipping.")
                continue

            sentence, triples_str = parts[0].strip(), parts[1].strip()
            words = sentence.split()

            try:
                triples_raw = ast.literal_eval(triples_str)
            except Exception as e:
                print(f"[dataset] WARNING: could not parse triples at line {line_no}: {e}")
                continue

            triples = []
            for t in triples_raw:
                asp_indices, opn_indices, sentiment = t
                # Spans are lists of individual word indices e.g. [5,6,7] or [2]
                # Normalize to [start, end] (inclusive)
                asp_span = [min(asp_indices), max(asp_indices)]
                opn_span = [min(opn_indices), max(opn_indices)]
                triples.append({
                    "aspect_span" : asp_span,   # [start, end]
                    "opinion_span": opn_span,   # [start, end]
                    "sentiment"   : sentiment   # 'POS' | 'NEG' | 'NEU'
                })

            samples.append({
                "sentence": sentence,
                "words"   : words,
                "triples" : triples
            })

    return samples


def load_aste_dataset(dataset_split: str = config.DATASET_SPLIT):
    """
    Full pipeline: download (if needed) -> parse -> return splits.

    Returns:
        train_data, dev_data, test_data  — each a list of sample dicts
    """
    print(f"\n[dataset] Loading ASTE-Data-V2  split='{dataset_split}' ...")
    download_raw_files(dataset_split)

    base = os.path.join(config.DATA_DIR, dataset_split)
    train_data = parse_file(os.path.join(base, "train_triplets.txt"))
    dev_data   = parse_file(os.path.join(base, "dev_triplets.txt"))
    test_data  = parse_file(os.path.join(base, "test_triplets.txt"))

    print(f"[dataset]  Train : {len(train_data)} sentences")
    print(f"[dataset]  Dev   : {len(dev_data)} sentences")
    print(f"[dataset]  Test  : {len(test_data)} sentences")

    return train_data, dev_data, test_data


# ─────────────────────────────────────────────────────────────
#  3.  EXPLORE — Pretty-print samples
# ─────────────────────────────────────────────────────────────

def inspect_sample(sample: dict, idx: int = 0):
    """Pretty-print a single sample."""
    print(f"\n{'─'*65}")
    print(f" Sample #{idx}")
    print(f"{'─'*65}")
    print(f" Sentence : {sample['sentence']}")
    print(f" Words    : {sample['words']}")
    print(f" Triplets :")
    for t in sample["triples"]:
        a_s, a_e = t["aspect_span"]
        o_s, o_e = t["opinion_span"]
        aspect   = " ".join(sample["words"][a_s : a_e + 1])
        opinion  = " ".join(sample["words"][o_s : o_e + 1])
        print(f"   • Aspect  [{a_s},{a_e}] : '{aspect}'")
        print(f"     Opinion [{o_s},{o_e}] : '{opinion}'")
        print(f"     Polarity           : {t['sentiment']}")
    print(f"{'─'*65}")


# ─────────────────────────────────────────────────────────────
#  4.  STATISTICS
# ─────────────────────────────────────────────────────────────

def compute_statistics(data: list, split_name: str = "train"):
    """Compute and display dataset statistics."""
    total_triplets   = 0
    sentiment_counts = Counter()
    aspect_lengths   = []
    opinion_lengths  = []
    sentence_lengths = []
    multi_aspect     = 0

    for sample in data:
        sentence_lengths.append(len(sample["words"]))
        n = len(sample["triples"])
        total_triplets += n
        if n > 1:
            multi_aspect += 1
        for t in sample["triples"]:
            sentiment_counts[t["sentiment"]] += 1
            a_s, a_e = t["aspect_span"]
            o_s, o_e = t["opinion_span"]
            aspect_lengths.append(a_e - a_s + 1)
            opinion_lengths.append(o_e - o_s + 1)

    n_sent = len(data)
    print(f"\n{'═'*65}")
    print(f"  Statistics — {split_name.upper()}")
    print(f"{'═'*65}")
    print(f"  Sentences            : {n_sent}")
    print(f"  Total triplets       : {total_triplets}")
    print(f"  Avg triplets/sent    : {total_triplets/n_sent:.2f}")
    print(f"  Multi-aspect sents   : {multi_aspect} ({100*multi_aspect/n_sent:.1f}%)")
    print(f"\n  Sentiment distribution:")
    for s, c in sentiment_counts.most_common():
        print(f"    {s:5s} : {c:4d}  ({100*c/total_triplets:.1f}%)")
    print(f"\n  Avg sentence length  : {sum(sentence_lengths)/n_sent:.1f} words")
    print(f"  Avg aspect length    : {sum(aspect_lengths)/len(aspect_lengths):.2f} words")
    print(f"  Avg opinion length   : {sum(opinion_lengths)/len(opinion_lengths):.2f} words")
    print(f"{'═'*65}")


# ─────────────────────────────────────────────────────────────
#  5.  SAVE AS CSV  (for manual inspection)
# ─────────────────────────────────────────────────────────────

def save_raw_csv(data: list, filename: str):
    """Save flat CSV of all triplets for easy inspection."""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    rows = []
    for sample in data:
        for t in sample["triples"]:
            a_s, a_e = t["aspect_span"]
            o_s, o_e = t["opinion_span"]
            rows.append({
                "sentence"    : sample["sentence"],
                "aspect"      : " ".join(sample["words"][a_s : a_e + 1]),
                "aspect_span" : str(t["aspect_span"]),
                "opinion"     : " ".join(sample["words"][o_s : o_e + 1]),
                "opinion_span": str(t["opinion_span"]),
                "sentiment"   : t["sentiment"],
            })
    df  = pd.DataFrame(rows)
    path = os.path.join(config.DATA_DIR, filename)
    df.to_csv(path, index=False)
    print(f"[dataset] CSV saved -> {path}  ({len(df)} rows)")
    return df


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train, dev, test = load_aste_dataset(config.DATASET_SPLIT)

    # Inspect first 3 training samples
    for i in range(3):
        inspect_sample(train[i], idx=i)

    # Statistics for train & test
    compute_statistics(train, "train")
    compute_statistics(test,  "test")

    # Save CSVs
    save_raw_csv(train, "train_raw.csv")
    save_raw_csv(dev,   "dev_raw.csv")
    save_raw_csv(test,  "test_raw.csv")

    print("\n[dataset] ✅ Step 2 complete — dataset loaded and explored!")
