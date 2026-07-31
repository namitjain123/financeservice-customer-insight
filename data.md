# Financial Services Customer Insights — Dataset

An LLM pipeline that turns unstructured consumer complaints into themed, analytics-ready
data. This document covers **how the dataset was built** — where the raw data came from,
every filter applied, and every transformation performed.

The dataset is derived from the [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/),
chosen specifically because it ships with **human-assigned category labels** (`Issue` /
`Sub-issue`). Those labels are held out of the pipeline and used as ground truth to measure
whether the unsupervised clustering actually recovers real categories — which is what
separates this from a pipeline that merely produces plausible-looking output.

---

## Source data

| | |
|---|---|
| Source | CFPB Consumer Complaint Database (bulk download) |
| File | `complaints.csv.zip` |
| Size | 1.33 GB compressed / **8.54 GB** uncompressed |
| Rows | **17,270,511** |
| Columns | 16 |
| Retrieved | 2026-07-27 |
| Licence | Public domain (US Government work) |

The bulk download is the complete database — every complaint since 2011, unfiltered. All
filtering is done locally by the scripts in `scripts/`, so the process is reproducible and
auditable rather than depending on choices made in a web UI.

---

## Pipeline overview

```
complaints.csv.zip              17,270,511 rows   8.54 GB
        │
        │  scripts/extract_from_bulk.py      (streamed in 200k-row chunks)
        ▼
data/cfpb_filtered.csv             781,473 rows   1,019 MB
        │
        │  scripts/prepare_cfpb.py           (clean → canonicalise → stratify)
        ▼
data/input.csv                       3,000 rows   3.3 MB
data/ground_truth.csv                3,000 rows
data/eval_holdout.csv                  200 rows
```

---

## Stage 1 — extraction

`scripts/extract_from_bulk.py` streams the zip in 200,000-row chunks so memory stays flat
regardless of file size. Three filters are applied to every chunk.

### Filter 1 — must have a narrative

```python
chunk = chunk[chunk["Consumer complaint narrative"].notna()]
```

The pipeline's entire input is free text. Most complaints in the database have no narrative
at all — text is only published where the consumer explicitly consented at submission. Rows
without it are unusable here.

### Filter 2 — 2023 onwards

```python
chunk = chunk[dates >= "2023-01-01"]
```

The raw data reaches back to 2011. Three and a half years is ample for monthly trend
analysis and keeps the working pool manageable. The surviving pool spans **43 months**
(2023-01 → 2026-07).

> The bulk file mixes two date formats in the same column — `2023-04-11` and
> `2023-04-11T09:07:47Z`. Parsing requires `format="mixed"`; without it pandas raises on
> the first inconsistent row.

### Filter 3 — credit reporting downsampled (not excluded)

Credit reporting is by far the largest category in the database:

| Product label | Rows |
|---|---:|
| `Credit reporting or other personal consumer reports` | 11,721,597 |
| `Credit reporting, credit repair services, or other personal consumer reports` | 2,163,780 |
| `Credit reporting` | 140,426 |
| **Total** | **14,025,803 — 81.2% of the database** |

Three strings, one category — CFPB renamed it as the taxonomy evolved, and complaints keep
whatever label was current when filed.

It is **randomly downsampled to 3%** of the pool, *not excluded*. The reasoning matters:

- The category is legitimate and the largest real-world one. Dropping it would make the
  dataset unrepresentative and the choice hard to defend.
- Its volume dominance is already handled downstream by the per-product cap in Stage 2.
- Downsampling exists purely to keep `cfpb_filtered.csv` around 1 GB rather than ~4 GB.
  A random subset of a random subset is still random, so the final sample is unaffected.

Measured characteristics of this category (full-database scan, post-2023):

| | |
|---|---|
| Complaints with narratives | 1,896,241 |
| Narrative rate | 17.9% |
| Distinct `Issue` values | 23 |
| Top-2 issue concentration | 77% |
| Median narrative length | 109 words |

It is more concentrated than other products (top-2 at 77% vs ~28% elsewhere), which is a
reason to cap it — not to remove it. Pass `--cr-fraction 1.0` to retain all 1.9M rows, or
`--cr-fraction 0` to exclude the category entirely.

### Stage 1 output

**781,473 rows (4.52% of source) · 1,019 MB · 91 distinct `Issue` values · 43 months**

Narrative length distribution in the pool:

| p10 | p25 | p50 | p75 | p90 | p99 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 36 | 68 | **128** | 242 | 398 | 1,020 | 6,469 |

90% of narratives are under 400 words — much shorter than a small sample from the CFPB web
search UI suggests, because that interface ranks results in a way that surfaces long,
detailed complaints first.

---

## Stage 2 — cleaning and sampling

`scripts/prepare_cfpb.py` turns the pool into the working set.

### Column selection and renaming

| CFPB column | Pipeline column |
|---|---|
| `Date received` | `date` |
| `Product` | `product` |
| `Sub-product` | `sub_product` |
| `Consumer complaint narrative` | `narrative` |
| `Company` | `company` |
| `State` | `state` |
| `Submitted via` | `channel` |
| `Complaint ID` | `complaint_id` |
| `Issue`, `Sub-issue` | *held out as ground truth* |

### Key integrity

```python
df = df[df["complaint_id"].notna()]
df["complaint_id"] = df["complaint_id"].astype("int64")
```

Nulls elsewhere in the bulk file coerce this column to float, which writes ids as
`10298545.0` and silently breaks the join to `ground_truth.csv`.

### Canonical product mapping

CFPB has renamed several product categories over the years. Grouping on the raw strings
treats each historical label as a distinct product, so a category renamed twice receives
double the sample weight. The mapping collapses variants **before** any grouping:

| Raw labels | Canonical |
|---|---|
| `Credit reporting` · `Credit reporting or other personal consumer reports` · `Credit reporting, credit repair services, or other personal consumer reports` | `Credit reporting` |
| `Payday loan` · `Payday loan, title loan, or personal loan` · `Payday loan, title loan, personal loan, or advance loan` | `Payday/title/personal loan` |
| `Credit card` · `Credit card or prepaid card` | `Credit card` |
| `Checking or savings account` · `Bank account or service` | `Checking or savings account` |
| `Vehicle loan or lease` · `Consumer Loan` | `Vehicle or consumer loan` |
| `Money transfer, virtual currency, or money service` · `Money transfers` · `Virtual currency` | `Money transfer or virtual currency` |

Result: **14 raw labels → 11 canonical products.**

`Other`, `Other financial service` and `Non-financial product/service` are dropped — a few
hundred rows, too vague to form a meaningful theme. This exclusion is a separate, explicit
step rather than being folded into a filter, so the modelling decision stays visible.

**Two mappings are judgment calls, not facts:**

- `Credit card or prepaid card` → `Credit card`. The legacy label covers both products with
  no clean way to split; credit card dominates the volume. `Prepaid card` remains separate.
- `Consumer Loan` → `Vehicle or consumer loan`. The old label spanned vehicle, personal,
  title and pawn loans. A defensible approximation, not a correct one.

### Text cleaning

CFPB redacts PII before publication, leaving heavy artefact tokens. Left raw, these dominate
the embedding space and degrade clustering — in some narratives roughly 9% of tokens are
redaction markers, with runs of 40+ consecutive `XXXX`.

| Pattern | Replacement |
|---|---|
| `{$9800.00}` | `[AMOUNT]` |
| `XX/XX/XXXX`, `XX/XX` | `[DATE]` |
| `XXXX` (2+ consecutive X) | `[REDACTED]` |
| Repeated `[REDACTED]` runs | single `[REDACTED]` |
| Whitespace runs | single space |

### Length bounds

```python
df = df[(df["word_count"] >= 30) & (df["word_count"] <= 3000)]
```

Under 30 words there is nothing to extract topics from; over 3,000 the text is typically a
pasted legal dossier rather than a complaint. Applied **after** cleaning, since collapsing
redaction runs shortens narratives and pushes some below the floor.

**781,473 → 726,023 usable rows.**

### Stratified sampling

```
3,000 target ÷ 11 canonical products ≈ 272 per product
```

Every product reaches its cap, giving a balanced sample.

### Truncation

First 400 words retained. Consumers state their core problem up front, and this bounds
step-1 token cost. Only affects the ~10% of narratives above that length.

### Ground-truth split

`Issue` and `Sub-issue` are written to a separate file keyed by `complaint_id`. **They never
appear in `input.csv`**, so there is no path for labels to leak into a prompt — verified
after every run.

---

## Output files

### `data/input.csv` — 3,000 rows

What the pipeline reads. Contains no labels.

| Column | Type | Notes |
|---|---|---|
| `complaint_id` | int64 | join key |
| `date` | str | `YYYY-MM-DD` |
| `product` | str | canonical, 11 values |
| `sub_product` | str | |
| `company` | str | |
| `state` | str | |
| `channel` | str | constant `Web` — see limitations |
| `narrative` | str | cleaned, ≤400 words |
| `word_count` | int | post-truncation |

Product balance:

| Product | Rows |
|---|---:|
| Money transfer or virtual currency | 275 |
| Checking or savings account | 274 |
| Credit card | 274 |
| Student loan | 273 |
| Mortgage | 272 |
| Debt or credit management | 272 |
| Payday/title/personal loan | 272 |
| Prepaid card | 272 |
| Credit reporting | 272 |
| Vehicle or consumer loan | 272 |
| Debt collection | 272 |

**79 distinct `Issue` labels · 43 months · median 149 words**

### `data/ground_truth.csv` — 3,000 rows

`complaint_id`, `Issue`, `Sub-issue`. The answer key for evaluating clustering.

### `data/eval_holdout.csv` — 200 rows

A labelled subset for scoring topic-extraction quality.

### `data/cfpb_filtered.csv` — 781,473 rows, 1,019 MB

Intermediate pool. Git-ignored (exceeds GitHub's 100 MB limit); regenerate from the zip in
a few minutes. Keep it to re-sample without re-reading 8.54 GB.

---

## Reproducing

```bash
pip install pandas numpy scikit-learn matplotlib openai python-dotenv

# 1. Download complaints.csv.zip from
#    https://www.consumerfinance.gov/data-research/consumer-complaints/

# 2. Extract the working pool (~5 min)
python scripts/extract_from_bulk.py --zip path/to/complaints.csv.zip --out data/cfpb_filtered.csv

# 3. Build the sample
python scripts/prepare_cfpb.py --raw data/cfpb_filtered.csv --out ./data
```

Both scripts use `--seed 42`, so runs are reproducible.

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--cr-fraction` | `0.03` | credit-reporting share of the pool (`1.0` = all, `0` = exclude) |
| `--min-date` | `2023-01-01` | pool start date |
| `--target` | `3000` | working-set size |
| `--holdout` | `200` | eval slice size |
| `--max-words` | `400` | truncation limit |

### `scripts/silhouette_check.py`

Justifies the number of clusters `k` rather than assuming it. Embeds unique topic phrases
once (cached to `.npz`, so a 24-value sweep costs one API call), sweeps k, and reports mean
silhouette under **cosine** distance — Euclidean degrades badly in 1536 dimensions, and
OpenAI embeddings are unit-normalised. Also emits the per-cluster silhouette diagram, which
shows *which* cluster is poor rather than only whether `k` is.

---

## Known limitations

**The sample is balanced, not representative.** Debt collection is ~33% of real complaints
but 9% here. Deliberate — it stops one category dominating the clustering — but "top pain
points" charts built on this reflect category *diversity*, not real-world *frequency*.
Report true frequencies separately from `cfpb_filtered.csv`.

**`channel` is constant.** Every row is `Web`, because narratives require online consent at
submission. The column is unusable as an analysis dimension; use `company` or `state`
instead.

**Ground truth is consumer-selected, not expert-adjudicated.** `Issue` is chosen by the
complainant from a dropdown at filing — not assigned by someone who read the narrative. It
is noisy ground truth. Better than none, but any metric computed against it should carry
that caveat.

**Redaction artefacts survive cleaning.** `[REDACTED]`, `[DATE]` and `[AMOUNT]` remain in
the text by design — collapsed to single tokens rather than removed, so sentence structure
stays intact. They will appear in embeddings.

**Credit reporting is 3% of its true pool volume.** Fine for a balanced sample, but do not
compute population statistics from `cfpb_filtered.csv` without accounting for it.

**Issue coverage is uneven.** 79 labels across 3,000 rows averages ~38 examples each, but
the distribution is skewed and tail issues are thin for evaluation. If that hurts your
metrics, stratify on `Issue` rather than `Product`.
