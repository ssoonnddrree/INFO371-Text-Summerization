"""
compute_statistics.py
---------------------
Computes exact dataset statistics for the ArXiv split of the
scientific_papers dataset (Cohan et al., 2018).

Downloads the dataset directly from the official S3 URL used by the
armanc/scientific_papers HuggingFace loading script — no local files
needed, works for all team members, no trust_remote_code issues.

The zip is cached in ~/.cache/scientific_papers/ after the first run
so it is only downloaded once.

Field names (from the loading script):
    article_text  : List[str]  — sentences of the full paper
    abstract_text : List[str]  — sentences of the abstract

Subset sizes used in this project:
    - Train:      first 5,000 examples (computational constraint)
    - Validation: first 500 examples   (computational constraint)
    - Test:       full 6,440 examples  (for final evaluation)

Run with:
    pip install transformers tqdm
    python compute_statistics.py

Output:
    - Prints a statistics table to stdout
    - Saves exact numbers to dataset_statistics.json
"""

import io
import json
import os
import statistics
import urllib.request
import zipfile
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
S3_URL         = "https://s3.amazonaws.com/datasets.huggingface.co/scientific_papers/1.1.1/arxiv-dataset.zip"
CACHE_DIR      = Path.home() / ".cache" / "scientific_papers" / "arxiv"
TOKENIZER_NAME = "facebook/bart-base"
OUTPUT_FILE    = "dataset_statistics.json"

SPLIT_FILES = {
    "train":      "arxiv-dataset/train.txt",
    "validation": "arxiv-dataset/val.txt",
    "test":       "arxiv-dataset/test.txt",
}

SPLIT_SIZES = {
    "train":      5000,
    "validation": 500,
    "test":       None,   # full 6,440
}
# ─────────────────────────────────────────────────────────────────────────────


def ensure_data() -> zipfile.ZipFile:
    """Download the zip once and cache it; return an open ZipFile handle."""
    zip_path = CACHE_DIR / "arxiv-dataset.zip"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Downloading dataset from S3 → {zip_path}")
        print("(This only happens once — subsequent runs use the cache.)\n")

        def reporthook(count, block_size, total_size):
            pct = min(100, count * block_size * 100 // total_size) if total_size > 0 else 0
            print(f"\r  {pct}% downloaded", end="", flush=True)

        urllib.request.urlretrieve(S3_URL, zip_path, reporthook)
        print()  # newline after progress
    else:
        print(f"Using cached dataset at {zip_path}\n")

    return zipfile.ZipFile(zip_path, "r")


def iter_split(zf: zipfile.ZipFile, split_name: str, subset_size: int | None):
    """Yield parsed JSON objects from the requested split inside the zip."""
    inner_path = SPLIT_FILES[split_name]
    with zf.open(inner_path) as raw:
        loaded = 0
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            if subset_size and loaded >= subset_size:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                loaded += 1
            except json.JSONDecodeError:
                continue


print(f"Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def count_words(sentences: list) -> int:
    return sum(len(s.split()) for s in sentences)


def count_tokens(sentences: list) -> int:
    return len(tokenizer.encode(" ".join(sentences), add_special_tokens=False))


def summarise(values: list, n: int) -> dict:
    return {
        "count":  n,
        "mean":   round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "stdev":  round(statistics.stdev(values), 2),
        "min":    min(values),
        "max":    max(values),
    }


def collect_stats(zf: zipfile.ZipFile, split_name: str, subset_size: int | None) -> dict:
    limit_str = f"first {subset_size:,}" if subset_size else "full"
    print(f"\nProcessing split: {split_name} ({limit_str} examples)")

    article_tokens, article_words, article_sents = [], [], []
    abstract_tokens, abstract_words, abstract_sents = [], [], []
    skipped = 0

    # Count lines for tqdm (re-open for counting, then iterate properly)
    total = subset_size  # use subset_size as total hint for progress bar

    for example in tqdm(iter_split(zf, split_name, subset_size),
                        total=total, desc=split_name):
        art  = example.get("article_text", [])
        abst = example.get("abstract_text", [])

        # Strip <S> / </S> tags the same way the loading script does
        abst = [s.replace("<S>", "").replace("</S>", "") for s in abst]

        if not art or not abst:
            skipped += 1
            continue

        article_tokens.append(count_tokens(art))
        article_words.append(count_words(art))
        article_sents.append(len(art))

        abstract_tokens.append(count_tokens(abst))
        abstract_words.append(count_words(abst))
        abstract_sents.append(len(abst))

    n = len(article_tokens)
    print(f"  Valid: {n:,}  |  Skipped: {skipped}")

    per_doc_ratios = [
        at / ab for at, ab in zip(article_tokens, abstract_tokens) if ab > 0
    ]

    return {
        "n_examples":         n,
        "n_skipped":          skipped,
        "subset_size_used":   subset_size if subset_size else "full split",
        "article_tokens":     summarise(article_tokens, n),
        "article_words":      summarise(article_words, n),
        "article_sentences":  summarise(article_sents, n),
        "abstract_tokens":    summarise(abstract_tokens, n),
        "abstract_words":     summarise(abstract_words, n),
        "abstract_sentences": summarise(abstract_sents, n),
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

zf = ensure_data()
with zf:
    for split_name, subset_size in SPLIT_SIZES.items():
        all_results[split_name] = collect_stats(zf, split_name, subset_size)

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