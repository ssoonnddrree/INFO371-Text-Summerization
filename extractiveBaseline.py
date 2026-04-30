"""
extractive_baseline.py
----------------------
Extractive summarization baseline using TF-IDF sentence scoring
with a lead bias, evaluated on ROUGE, BERTScore, Semantic Similarity,
and NLI-based Factual Consistency.

This baseline runs on the full 6,440-example test split and saves results to:
    - extractive_baseline_summaries.json  (generated summaries)
    - extractive_baseline_scores.json     (all evaluation results)

Run with:
    pip install scikit-learn rouge-score bert-score tqdm sentence-transformers transformers==4.44.0
    python extractive_baseline.py
"""

import io
import json
import math
import urllib.request
import zipfile
from pathlib import Path
from collections import Counter

import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

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
# The formula gives sentence 0 twice the weight, decaying smoothly so
# deep sentences can still surface if their TF-IDF score is strong enough.
LEAD_BIAS      = 1.0

# BERTScore model: roberta-large is the standard default for bert_score
# and is confirmed to work reliably.
BERTSCORE_MODEL = "roberta-large"

# Semantic similarity model: all-MiniLM-L6-v2 is a lightweight,
# well-regarded sentence-transformers model for cosine similarity scoring.
SEMANTIC_MODEL = "all-MiniLM-L6-v2"

# NLI model: DeBERTa-v3-small cross-encoder for factual consistency scoring.
# Each (source, summary) pair is scored as entailment/neutral/contradiction
# and mapped to a [0, 1] consistency score.
NLI_MODEL      = "cross-encoder/nli-deberta-v3-small"

# To keep NLI runtime manageable over 6,440 examples, we subsample.
# Set to None to run on all examples (will take ~30-40 min on GPU).
NLI_SAMPLE_SIZE = 500

SUMMARIES_FILE = "extractive_baseline_summaries.json"
SCORES_FILE    = "extractive_baseline_scores.json"
# ─────────────────────────────────────────────────────────────────────────────


# ── Data loading ──────────────────────────────────────────────────────────────

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


# ── TF-IDF implementation ─────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace, keep only alphabetic tokens."""
    return [w.lower() for w in text.split() if w.isalpha()]


def compute_tfidf_scores(sentences: list[str]) -> list[float]:
    N = len(sentences)
    tokenized = [tokenize(s) for s in sentences]

    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    sentence_scores = []
    for tokens in tokenized:
        if not tokens:
            sentence_scores.append(0.0)
            continue

        tf = Counter(tokens)
        total_terms = len(tokens)

        score = 0.0
        for term, count in tf.items():
            tf_val  = count / total_terms
            idf_val = math.log(N / (1 + df[term]))
            score  += tf_val * idf_val

        sentence_scores.append(score)

    return sentence_scores


def apply_lead_bias(scores: list[float], bias: float) -> list[float]:
    return [
        score * (1.0 + bias / (1.0 + pos))
        for pos, score in enumerate(scores)
    ]


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_summary(sentences: list[str], n: int, bias: float) -> str:
    """
    Score sentences with TF-IDF + lead bias, select the top-n by score,
    then reassemble them in their original document order.
    """
    if len(sentences) <= n:
        return " ".join(sentences)

    scores  = compute_tfidf_scores(sentences)
    scores  = apply_lead_bias(scores, bias)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:n]

    top_indices = sorted(top_indices)
    return " ".join(sentences[i] for i in top_indices)


# ── Evaluation ────────────────────────────────────────────────────────────────

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
    """Truncate text to max_words words before passing to BERTScore."""
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


def compute_semantic_similarity(
    predictions: list[str],
    references: list[str],
    model_name: str,
    batch_size: int = 64
) -> dict:
    """
    Compute cosine similarity between sentence embeddings of predictions
    and references using a sentence-transformers model.

    This serves as a semantic similarity metric comparable in purpose to
    BERTScore F1, but using a single-vector-per-text approach rather than
    token-level alignment.
    """
    print(f"\nComputing Semantic Similarity with {model_name}...")
    model = SentenceTransformer(model_name)

    print("  Encoding references...")
    ref_embs  = model.encode(
        references, convert_to_tensor=True,
        show_progress_bar=True, batch_size=batch_size
    )
    print("  Encoding predictions...")
    pred_embs = model.encode(
        predictions, convert_to_tensor=True,
        show_progress_bar=True, batch_size=batch_size
    )

    # Diagonal of cosine similarity matrix = per-pair similarity
    scores = util.cos_sim(ref_embs, pred_embs).diagonal().cpu().numpy()

    return {
        "mean":   round(float(scores.mean()), 4),
        "std":    round(float(scores.std()),  4),
        "scores": scores.tolist(),
    }


def compute_nli_consistency(
    sources: list[str],
    summaries: list[str],
    model_name: str,
    batch_size: int = 8,
    src_max_chars: int = 1500,
    sum_max_chars: int = 500,
) -> dict:
    """
    Compute factual consistency using an NLI cross-encoder.

    For each (source, summary) pair:
      - ENTAILMENT → consistency score = model confidence
      - CONTRADICTION → consistency score = 1 - model confidence
      - NEUTRAL → consistency score = 0.5

    Sources are truncated to src_max_chars to stay within the model's
    token limit while still providing enough context for NLI judgment.
    """
    print(f"\nComputing NLI Consistency with {model_name}...")
    nli = pipeline(
        "text-classification",
        model=model_name,
        device=0  # use GPU if available; set to -1 for CPU
    )

    scores = []
    for i in tqdm(range(0, len(sources), batch_size), desc="  NLI scoring"):
        batch_src = sources[i:i+batch_size]
        batch_sum = summaries[i:i+batch_size]

        pairs = [
            {"text": src[:src_max_chars], "text_pair": summ[:sum_max_chars]}
            for src, summ in zip(batch_src, batch_sum)
        ]
        preds = nli(pairs, truncation=True)

        for pred in preds:
            label = pred["label"].lower()
            if label == "entailment":
                scores.append(pred["score"])
            elif label == "contradiction":
                scores.append(1.0 - pred["score"])
            else:  # neutral
                scores.append(0.5)

    scores = np.array(scores)
    return {
        "mean":   round(float(scores.mean()), 4),
        "std":    round(float(scores.std()),  4),
        "n":      len(scores),
        "scores": scores.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # ── Load or generate summaries ────────────────────────────────────────────
    if Path(SUMMARIES_FILE).exists():
        print(f"Loading existing summaries from {SUMMARIES_FILE}...")
        with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
            summaries = json.load(f)
        predictions = [s["generated"] for s in summaries]
        references  = [s["reference"] for s in summaries]
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

    # ── ROUGE ─────────────────────────────────────────────────────────────────
    print("\nComputing ROUGE scores...")
    rouge_results = compute_rouge(predictions, references)
    print(f"  ROUGE-1: {rouge_results['rouge1']}")
    print(f"  ROUGE-2: {rouge_results['rouge2']}")
    print(f"  ROUGE-L: {rouge_results['rougeL']}")

    # ── BERTScore ─────────────────────────────────────────────────────────────
    bertscore_results = compute_bertscore(predictions, references, BERTSCORE_MODEL)
    print(f"\n  BERTScore Precision : {bertscore_results['precision']}")
    print(f"  BERTScore Recall    : {bertscore_results['recall']}")
    print(f"  BERTScore F1        : {bertscore_results['f1']}")

    # ── Semantic Similarity ───────────────────────────────────────────────────
    sem_results = compute_semantic_similarity(predictions, references, SEMANTIC_MODEL)
    print(f"\n  Semantic Similarity (MiniLM): {sem_results['mean']} ± {sem_results['std']}")

    # ── NLI Consistency ───────────────────────────────────────────────────────
    # Subsample for speed if NLI_SAMPLE_SIZE is set
    if NLI_SAMPLE_SIZE is not None and NLI_SAMPLE_SIZE < len(predictions):
        print(f"\n  Subsampling {NLI_SAMPLE_SIZE} examples for NLI consistency...")
        import random
        random.seed(42)
        indices = random.sample(range(len(predictions)), NLI_SAMPLE_SIZE)
        nli_preds = [predictions[i] for i in indices]
        nli_refs  = [references[i]  for i in indices]
    else:
        nli_preds = predictions
        nli_refs  = references

    nli_results = compute_nli_consistency(nli_refs, nli_preds, NLI_MODEL)
    print(f"\n  NLI Consistency: {nli_results['mean']} ± {nli_results['std']} (n={nli_results['n']})")

    # ── Print unified summary table ───────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'Metric':<30} {'Score':>10}")
    print("=" * 50)
    print(f"{'ROUGE-1':<30} {rouge_results['rouge1']:>10.4f}")
    print(f"{'ROUGE-2':<30} {rouge_results['rouge2']:>10.4f}")
    print(f"{'ROUGE-L':<30} {rouge_results['rougeL']:>10.4f}")
    print(f"{'BERTScore F1 (roberta-large)':<30} {bertscore_results['f1']:>10.4f}")
    print(f"{'Semantic Sim (MiniLM)':<30} {sem_results['mean']:>10.4f}")
    print(f"{'NLI Consistency (DeBERTa)':<30} {nli_results['mean']:>10.4f}")
    print("=" * 50)

    # ── Save all scores ───────────────────────────────────────────────────────
    scores = {
        "model": "extractive_tfidf_lead_bias",
        "config": {
            "num_sentences":    NUM_SENTENCES,
            "lead_bias":        LEAD_BIAS,
            "bertscore_model":  BERTSCORE_MODEL,
            "semantic_model":   SEMANTIC_MODEL,
            "nli_model":        NLI_MODEL,
            "nli_sample_size":  NLI_SAMPLE_SIZE,
        },
        "n_test_examples": len(predictions),
        "rouge":       rouge_results,
        "bertscore":   bertscore_results,
        "semantic_similarity": {
            "mean": sem_results["mean"],
            "std":  sem_results["std"],
        },
        "nli_consistency": {
            "mean": nli_results["mean"],
            "std":  nli_results["std"],
            "n":    nli_results["n"],
        },
    }
    with open(SCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nScores saved → {SCORES_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()