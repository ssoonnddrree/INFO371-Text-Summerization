"""
extractive_baseline.py
----------------------
Extractive summarization baseline using TF-IDF sentence scoring
with a lead bias, evaluated on ROUGE and BERTScore.

This baseline runs on the full 6,440-example test split and saves results to:
    - extractive_baseline_summaries.json  (generated summaries)
    - extractive_baseline_scores.json     (ROUGE + BERTScore results)

Run with:
    pip install scikit-learn rouge-score bert-score tqdm
    python extractive_baseline.py
"""

import io
import json
import math
import urllib.request
import zipfile
from pathlib import Path
from collections import Counter

from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
S3_URL         = "https://s3.amazonaws.com/datasets.huggingface.co/scientific_papers/1.1.1/arxiv-dataset.zip"
CACHE_DIR      = Path.home() / ".cache" / "scientific_papers" / "arxiv"
TEST_FILE      = "arxiv-dataset/test.txt"

# How many sentences to extract as the summary.
# The reference abstracts average ~6 sentences (median=6, from our statistics),
# so we extract 6 sentences to match that target length.
NUM_SENTENCES  = 6

# Lead bias: a position-based multiplier applied to sentence scores.
# Sentences earlier in the document receive a higher weight because
# scientific papers front-load their key contributions in the introduction.
# The multiplier decays as: 1 + LEAD_BIAS / (1 + position)
# The formula gives sentence 0 twice the weight, decaying smoothly so deep sentences can still surface if their TF-IDF score is strong enough.
LEAD_BIAS      = 1.0

# BERTScore model: roberta-large is the standard default for bert_score
# and is confirmed to work reliably. 
# allenai/scibert_scivocab_uncased triggers an internal OverflowError in the bert_score library due to a
# known incompatibility between its tokenizer and the bert_score internals — unrelated to our text lengths. 
# roberta-large is a strong general model
# that is widely used for BERTScore in summarization research, making our
# results directly comparable to published baselines.
BERTSCORE_MODEL = "roberta-large"

SUMMARIES_FILE = "extractive_baseline_summaries.json"
SCORES_FILE    = "extractive_baseline_scores.json"
# ─────────────────────────────────────────────────────────────────────────────


# ── Data loading ────────────────────

def ensure_data() -> zipfile.ZipFile:
    zip_path = CACHE_DIR / "arxiv-dataset.zip"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"Downloading dataset → {zip_path}")
        def reporthook(count, block_size, total_size):
            pct = min(100, count * block_size * 100 // total_size) if total_size > 0 else 0
            print(f"\r  {pct}% downloaded", end="", flush=True)
        urllib.request.urlretrieve(S3_URL, zip_path, reporthook)
        print()
    else:
        print(f"Using cached dataset at {zip_path}")
    return zipfile.ZipFile(zip_path, "r")


def iter_split(zf: zipfile.ZipFile, inner_path: str, limit: int | None = None):
    with zf.open(inner_path) as raw:
        loaded = 0
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            if limit and loaded >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                loaded += 1
            except json.JSONDecodeError:
                continue


# ── TF-IDF implementation ────────────────────────────────────────────────────
# We implement TF-IDF from scratch (no sklearn) so the logic is transparent
# and there are no extra dependencies beyond what we already need.
#
# A sentence's score is the sum of TF-IDF weights of all its words.
# This is a classic unsupervised approach that requires no training data.

def tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace, keep only alphabetic tokens."""
    return [w.lower() for w in text.split() if w.isalpha()]


def compute_tfidf_scores(sentences: list[str]) -> list[float]:
    N = len(sentences)
    tokenized = [tokenize(s) for s in sentences]

    # Document frequency: how many sentences contain each term
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))      # set() so each term counted once per sentence

    sentence_scores = []
    for tokens in tokenized:
        if not tokens:
            sentence_scores.append(0.0)
            continue

        # Term frequency within this sentence
        tf = Counter(tokens)
        total_terms = len(tokens)

        score = 0.0
        for term, count in tf.items():
            tf_val  = count / total_terms
            idf_val = math.log(N / (1 + df[term]))
            score  += tf_val * idf_val

        sentence_scores.append(score)

    return sentence_scores


# ── Lead bias ────────────────────────────────────────────────────────────────
# WHY LEAD BIAS?
# Scientific papers are structured with a clear front-loading convention:
# the abstract, introduction and contributions appear early. A purely
# TF-IDF scorer treats all sentences equally regardless of position,
# which can cause it to extract mid-paper technical details instead of
# the high-level contributions.
#
# The lead bias multiplier rewards sentences that appear earlier in the
# document. The formula 1 + LEAD_BIAS / (1 + position) means:
#   - Sentence 0:  multiplier = 1 + 1.0 / 1 = 2.0  (double weight)
#   - Sentence 5:  multiplier = 1 + 1.0 / 6 ≈ 1.17
#   - Sentence 50: multiplier = 1 + 1.0 / 51 ≈ 1.02 (nearly neutral)
# This is a soft bias: deep content can still surface if its TF-IDF score is high enough.

def apply_lead_bias(scores: list[float], bias: float) -> list[float]:
    return [
        score * (1.0 + bias / (1.0 + pos))
        for pos, score in enumerate(scores)
    ]


# ── Extraction ───────────────────────────────────────────────────────────────

def extract_summary(sentences: list[str], n: int, bias: float) -> str:
    """
    Score sentences with TF-IDF + lead bias, select the top-n by score,
    then reassemble them in their ORIGINAL document order.

    We preserve original order (rather than sorting by score) because
    a summary that reads in document order is more coherent.
    The reader sees the logical flow of the paper rather than a scrambled
    list of important-sounding sentences.
    """
    if len(sentences) <= n:
        return " ".join(sentences)

    scores  = compute_tfidf_scores(sentences)
    scores  = apply_lead_bias(scores, bias)

    # Get indices of top-n scoring sentences
    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:n]

    # Re-sort by position to preserve document order
    top_indices = sorted(top_indices)

    return " ".join(sentences[i] for i in top_indices)


# ── Evaluation ───────────────────────────────────────────────────────────────

def compute_rouge(predictions: list[str], references: list[str]) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L for all prediction/reference pairs."""

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1_scores, r2_scores, rl_scores = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1_scores.append(s["rouge1"].fmeasure)
        r2_scores.append(s["rouge2"].fmeasure)
        rl_scores.append(s["rougeL"].fmeasure)

    def avg(lst): return round(sum(lst) / len(lst), 4)

    return {
        "rouge1": avg(r1_scores),
        "rouge2": avg(r2_scores),
        "rougeL": avg(rl_scores),
    }


def truncate_to_wordcount(text: str, max_words: int = 400) -> str:
    """
    Truncate text to max_words words before passing to BERTScore.
    BERTScore uses BERT-based models with a hard 512-token limit.
    Truncating to 400 words keeps us safely within that
    limit for all inputs, including the longest generated summaries.
    We truncate by words rather than tokens to avoid loading a second
    tokenizer just for preprocessing.
    """
    words = text.split()
    return " ".join(words[:max_words]) if len(words) > max_words else text


def compute_bertscore(predictions: list[str], references: list[str], model: str) -> dict:
    print(f"\nComputing BERTScore with {model} (this may take a few minutes)...")
    predictions_trunc = [truncate_to_wordcount(p) for p in predictions]
    references_trunc  = [truncate_to_wordcount(r) for r in references]
    P, R, F = bert_score(
        predictions_trunc,
        references_trunc,
        model_type=model,
        verbose=True,
    )
    return {
        "precision": round(P.mean().item(), 4),
        "recall":    round(R.mean().item(), 4),
        "f1":        round(F.mean().item(), 4),
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():

    # ── Load or generate summaries ───────────────────────────────────
    # If summaries are already saved from a previous run, reload them directly.
    # This avoids re-running the generation step every time.
    if Path(SUMMARIES_FILE).exists():
        print(f"Loading existing summaries from {SUMMARIES_FILE}...")
        with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
            summaries = json.load(f)
        predictions = [s["generated"]  for s in summaries]
        references  = [s["reference"]  for s in summaries]
        print(f"Loaded {len(predictions)} summaries.")

    else:
        zf = ensure_data()
        print("\nGenerating extractive summaries on test split...")
        predictions = []
        references  = []
        summaries   = []

        with zf:
            for example in tqdm(iter_split(zf, TEST_FILE), desc="test", total=6440):
                art_sentences  = example.get("article_text", [])
                abst_sentences = example.get("abstract_text", [])

                # Clean <S> / </S> tags from abstract (same as loading script)
                abst_sentences = [
                    s.replace("<S>", "").replace("</S>", "")
                    for s in abst_sentences
                ]

                reference = " ".join(abst_sentences)
                summary   = extract_summary(art_sentences, NUM_SENTENCES, LEAD_BIAS)

                predictions.append(summary)
                references.append(reference)
                summaries.append({
                    "article_id": example.get("article_id", ""),
                    "generated":  summary,
                    "reference":  reference,
                })

        print(f"\nGenerated {len(predictions)} summaries.")
        with open(SUMMARIES_FILE, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, ensure_ascii=False)
        print(f"Summaries saved → {SUMMARIES_FILE}")

    # ── ROUGE ───────────────────────────────────────────────
    print("\nComputing ROUGE scores...")
    rouge_results = compute_rouge(predictions, references)
    print(f"  ROUGE-1: {rouge_results['rouge1']}")
    print(f"  ROUGE-2: {rouge_results['rouge2']}")
    print(f"  ROUGE-L: {rouge_results['rougeL']}")

    # ── BERTScore ───────────────────────────────────────────────
    bertscore_results = compute_bertscore(predictions, references, BERTSCORE_MODEL)
    print(f"\n  BERTScore precision : {bertscore_results['precision']}")
    print(f"  BERTScore recall    : {bertscore_results['recall']}")
    print(f"  BERTScore F1        : {bertscore_results['f1']}")

    # ── Save scores ───────────────────────────────────────────────
    scores = {
        "model":        "extractive_tfidf_lead_bias",
        "config": {
            "num_sentences":   NUM_SENTENCES,
            "lead_bias":       LEAD_BIAS,
            "bertscore_model": BERTSCORE_MODEL,
        },
        "n_test_examples": len(predictions),
        "rouge":     rouge_results,
        "bertscore": bertscore_results,
    }
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nScores saved → {SCORES_FILE}")
    print("\nDone.")


if __name__ == "__main__":
    main()