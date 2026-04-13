"""
Computes exact dataset statistics for the ArXiv split of the
scientific_papers dataset (Cohan et al., 2018).

Fields: article (full paper text), abstract (summary)

Subset sizes used in this project:
    - Train:      first 5,000 examples (computational constraint)
    - Validation: first 500 examples   (computational constraint)
    - Test:       full 6,440 examples  (for final evaluation)

Run with:
    pip install datasets transformers tqdm
    python compute_statistics.py

Output:
    - Prints a statistics table to the terminal
    - Saves exact numbers to dataset_statistics.json
"""

import json
import statistics
from transformers import AutoTokenizer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
# Local jsonlines files from the armanc/scientific_papers dataset
DATA_DIR = "arxiv-dataset"
LOCAL_FILES = {
    "train":      f"{DATA_DIR}/train.txt",
    "validation": f"{DATA_DIR}/val.txt",
    "test":       f"{DATA_DIR}/test.txt",
}
TOKENIZER_NAME = "facebook/bart-base"   # same tokenizer used for fine-tuning
OUTPUT_FILE    = "dataset_statistics.json"

# Subset sizes: None means use the full split
SPLIT_SIZES = {
    "train":      5000,
    "validation": 500,
    "test":       None,   # full 6,440
}
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_words(text: str) -> int:
    return len(text.split())


def count_sentences(text: str) -> int:
    # Paragraphs are separated by \n in this dataset version
    return len([s for s in text.split("\n") if s.strip()])


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def summarise(values: list, n: int) -> dict:
    return {
        "count":  n,
        "mean":   round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "stdev":  round(statistics.stdev(values), 2),
        "min":    min(values),
        "max":    max(values),
    }


def collect_stats(split_name: str, subset_size: int | None) -> dict:
    limit_str = f"first {subset_size:,}" if subset_size else "full"
    filepath = LOCAL_FILES[split_name]
    print(f"\nLoading split: {split_name} ({limit_str} examples) from {filepath}")

    article_tokens, article_words, article_sents = [], [], []
    abstract_tokens, abstract_words, abstract_sents = [], [], []
    skipped = 0
    loaded = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=split_name):
            if subset_size and loaded >= subset_size:
                break
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            # Original armanc dataset uses article_text and abstract_text
            # as lists of sentences — join them into a single string
            art_raw  = example.get("article_text", [])
            abst_raw = example.get("abstract_text", [])

            if isinstance(art_raw, list):
                art  = " ".join(art_raw)
                abst = " ".join(abst_raw)
                n_art_sents  = len(art_raw)
                n_abst_sents = len(abst_raw)
            else:
                art  = art_raw
                abst = abst_raw
                n_art_sents  = count_sentences(art)
                n_abst_sents = count_sentences(abst)

            if not art.strip() or not abst.strip():
                skipped += 1
                continue

            article_tokens.append(count_tokens(art))
            article_words.append(count_words(art))
            article_sents.append(n_art_sents)

            abstract_tokens.append(count_tokens(abst))
            abstract_words.append(count_words(abst))
            abstract_sents.append(n_abst_sents)

            loaded += 1

    n = len(article_tokens)
    print(f"  Valid examples: {n:,}  |  Skipped: {skipped}")

    # Compression ratio: article_tokens / abstract_tokens, per document
    per_doc_ratios = [
        at / ab
        for at, ab in zip(article_tokens, abstract_tokens)
        if ab > 0
    ]

    return {
        "n_examples":          n,
        "n_skipped":           skipped,
        "subset_size_used":    subset_size if subset_size else "full split",
        "article_tokens":      summarise(article_tokens, n),
        "article_words":       summarise(article_words, n),
        "article_sentences":   summarise(article_sents, n),
        "abstract_tokens":     summarise(abstract_tokens, n),
        "abstract_words":      summarise(abstract_words, n),
        "abstract_sentences":  summarise(abstract_sents, n),
        "compression_ratio": {
            "definition": (
                "Per-document ratio of article_tokens / abstract_tokens "
                "(BART tokenizer, no special tokens), averaged across documents."
            ),
            "mean":   round(statistics.mean(per_doc_ratios), 2),
            "median": round(statistics.median(per_doc_ratios), 2),
            "stdev":  round(statistics.stdev(per_doc_ratios), 2),
            "min":    round(min(per_doc_ratios), 2),
            "max":    round(max(per_doc_ratios), 2),
        },
        "truncation_coverage": {
            "description": (
                "Percentage of articles that already fit within the "
                "truncation thresholds used in this project."
            ),
            "pct_within_1024_tokens": round(
                100 * sum(1 for t in article_tokens if t <= 1024) / n, 2
            ),
            "pct_within_3000_words": round(
                100 * sum(1 for w in article_words if w <= 3000) / n, 2
            ),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────
all_results = {}

for split_name, subset_size in SPLIT_SIZES.items():
    all_results[split_name] = collect_stats(split_name, subset_size)

# Save raw JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nRaw results saved → {OUTPUT_FILE}")

# ── Pretty-print summary ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DATASET STATISTICS SUMMARY  (ArXiv, BART tokenizer)")
print("=" * 70)

SUBSET_LABELS = {
    "train":      "subset: 5,000",
    "validation": "subset: 500",
    "test":       "full: 6,440",
}

for split, s in all_results.items():
    print(f"\n── {split.upper()}  ({SUBSET_LABELS[split]}, valid n = {s['n_examples']:,}) ──")
    print(f"  {'':28s} {'Tokens':>10s}  {'Words':>10s}  {'Paragraphs':>10s}")

    for label, tok_key, word_key, sent_key in [
        ("Article  mean",   "article_tokens",  "article_words",  "article_sentences"),
        ("Article  median", "article_tokens",  "article_words",  "article_sentences"),
        ("Article  min",    "article_tokens",  "article_words",  "article_sentences"),
        ("Article  max",    "article_tokens",  "article_words",  "article_sentences"),
        ("Abstract mean",   "abstract_tokens", "abstract_words", "abstract_sentences"),
        ("Abstract median", "abstract_tokens", "abstract_words", "abstract_sentences"),
        ("Abstract min",    "abstract_tokens", "abstract_words", "abstract_sentences"),
        ("Abstract max",    "abstract_tokens", "abstract_words", "abstract_sentences"),
    ]:
        stat = label.split()[-1]
        print(
            f"  {label:28s} "
            f"{s[tok_key][stat]:>10}  "
            f"{s[word_key][stat]:>10}  "
            f"{s[sent_key][stat]:>10}"
        )

    cr = s["compression_ratio"]
    print(f"\n  Compression ratio  mean={cr['mean']}  median={cr['median']}  "
          f"stdev={cr['stdev']}  min={cr['min']}  max={cr['max']}")
    print(f"  Definition: {cr['definition']}")

    tc = s["truncation_coverage"]
    print(f"\n  Articles fitting within 1,024 tokens : {tc['pct_within_1024_tokens']}%")
    print(f"  Articles fitting within 3,000 words  : {tc['pct_within_3000_words']}%")

print("\n" + "=" * 70)
print(f"Done. Exact numbers saved to {OUTPUT_FILE}")