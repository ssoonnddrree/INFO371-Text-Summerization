# INFO371-Text-Summerization

installed:
'''bash
    pip install datasets transformers tqdm
'''

# Dataset Statistics

## Overview

We use the **ArXiv** split of the Scientific Papers dataset (Cohan et al., 2018), loaded from [`armanc/scientific_papers`](https://huggingface.co/datasets/armanc/scientific_papers). Each example consists of a full scientific paper (`article_text`) paired with its author-written abstract (`abstract_text`). Both fields are stored as lists of sentences in the raw dataset.

Due to computational constraints, we train on a **subset of 5,000 examples** and validate on **500 examples**, while evaluating on the **full test split of 6,440 examples**.

---

## `compute_statistics.py`

The script [`compute_statistics.py`](compute_statistics.py) computes exact statistics over each split used in this project. It reads directly from (`arxiv-dataset/train.txt`, `val.txt`, `test.txt`) and tokenizes all text using the **BART tokenizer** (`facebook/bart-base`) — the same tokenizer used during fine-tuning — so that token counts are internally consistent with the model.

### What it computes

For both articles and abstracts in each split, it reports:
- **Token count** — using the BART tokenizer (no special tokens)
- **Word count** — whitespace-split
- **Sentence count** — number of sentences in the pre-tokenized list

It also computes:
- **Compression ratio** — defined as `article_tokens / abstract_tokens` per document, then averaged. This quantifies how much semantic compression the summarization task requires.
- **Truncation coverage** — the percentage of articles that already fit within the two input-length thresholds used in the project (1,024 tokens for BART fine-tuning; 3,000 words for LLM prompting).

### How to run

```bash
pip install transformers tqdm
python compute_statistics.py
```

Results are printed to the terminal and saved to `dataset_statistics.json`.

---

## Results

### Train split (n = 5,000)

|                    | Tokens    | Words    | Sentences |
|--------------------|-----------|----------|-----------|
| **Article mean**   | 8,470.27  | 5,950.13 | 204.24    |
| **Article median** | 6,770.00  | 4,890.00 | 171.00    |
| **Article min**    | 14        | 4        | 1         |
| **Article max**    | 100,666   | 49,926   | 1,733     |
| **Abstract mean**  | 431.25    | 305.67   | 10.10     |
| **Abstract median**| 240.00    | 179.00   | 6.00      |
| **Abstract min**   | 13        | 9        | 1         |
| **Abstract max**   | 23,085    | 12,563   | 477       |

> **Note:** The train subset shows higher variance in abstract length (stdev = 1,004.79 tokens) compared to the validation and test splits. This is likely due to a small number of outlier examples in the first 5,000 rows with unusually long abstracts (max 23,085 tokens). The validation and test splits, which cover a broader and more representative sample, show much tighter abstract distributions (stdev ≈ 85–89 tokens).

### Validation split (n = 500)

|                    | Tokens   | Words    | Sentences |
|--------------------|----------|----------|-----------|
| **Article mean**   | 8,207.22 | 5,978.50 | 208.60    |
| **Article median** | 6,907.00 | 5,110.50 | 175.50    |
| **Article min**    | 493      | 404      | 9         |
| **Article max**    | 50,637   | 36,459   | 1,483     |
| **Abstract mean**  | 230.52   | 172.13   | 5.60      |
| **Abstract median**| 225.00   | 171.00   | 5.00      |
| **Abstract min**   | 69       | 51       | 1         |
| **Abstract max**   | 655      | 300      | 16        |

### Test split (n = 6,440 — full)

|                    | Tokens   | Words    | Sentences |
|--------------------|----------|----------|-----------|
| **Article mean**   | 8,131.70 | 5,905.87 | 205.68    |
| **Article median** | 6,726.50 | 5,016.50 | 176.00    |
| **Article min**    | 187      | 105      | 1         |
| **Article max**    | 104,441  | 84,895   | 3,045     |
| **Abstract mean**  | 232.99   | 174.51   | 5.69      |
| **Abstract median**| 226.00   | 171.00   | 6.00      |
| **Abstract min**   | 61       | 50       | 1         |
| **Abstract max**   | 667      | 300      | 25        |

---

## Compression Ratio

The compression ratio is defined as:

```
compression_ratio = article_tokens / abstract_tokens
```

computed per document using the BART tokenizer, then averaged across all documents in the split.

| Split      | Mean  | Median | Stdev | Min  | Max      |
|------------|-------|--------|-------|------|----------|
| Train      | 38.38 | 29.32  | 48.15 | 0.00 | 2,147.69 |
| Validation | 39.49 | 32.51  | 31.13 | 1.96 | 237.32   |
| Test       | 38.23 | 30.88  | 29.99 | 1.04 | 430.53   |

The median compression ratio of ~30:1 across all splits highlights the degree of semantic distillation required. The model must reduce an average article of ~6,700 tokens down to an abstract of ~226 tokens while preserving the key contributions.

---

## Truncation Coverage

| Split      | Articles ≤ 1,024 tokens | Articles ≤ 3,000 words |
|------------|------------------------|------------------------|
| Train      | 2.16%                  | 22.94%                 |
| Validation | 1.00%                  | 21.00%                 |
| Test       | 1.09%                  | 18.43%                 |

Only **~1–2% of articles fit within the 1,024-token limit** used for BART fine-tuning, meaning the model sees only the first ~12% of the average document. Even the 3,000-word threshold used for LLM prompting covers only ~20% of documents in full. This confirms that truncation is not a minor preprocessing choice but a fundamental constraint that affects all models, and directly motivates our tiered input-length evaluation strategy (see Methods).