"""Pick the number of clusters (k) by agreement with real categories, not geometry.

Silhouette can only ask "are these clusters separated?" - and as k grows, that
question gets easier to answer well even when the clustering is meaningless
(the degenerate limit is one cluster per point). It is not a reliable guide to
the right k for text embeddings, where categories overlap rather than sit in
clean, empty space.

This sweeps k and scores each value against the real CFPB Issue labels via
ARI - "which k makes my clusters agree with the true taxonomy?" - which is
the question that actually matters. Silhouette is still reported alongside,
so you can see directly where the two disagree.

Uses the cached embeddings from topic_clustering.py (one vector per complaint,
embedded from its narrative), so sweeping costs zero additional API calls -
each k just re-clusters the same cached vectors.

Two ground-truth granularities to sweep against - pick with a trailing arg:
  issue     (default) the real CFPB Issue field - 60-80 fine-grained categories.
            k has to reach into that range before ARI can rise much, which
            trades away the interpretability of a small, business-readable
            cluster count.
  product   input.csv's `product` column - 11 balanced categories a business
            reader already recognises (Mortgage, Credit card, ...). A fairer
            target for a small k, since it doesn't ask 12 clusters to resolve
            distinctions finer than the k you're actually choosing.

Usage:
    python -m eval.choose_k                       # sweep k=2..30 against Issue
    python -m eval.choose_k 5 40 2                 # sweep k=5..40 step 2 against Issue
    python -m eval.choose_k 2 20 1 product          # sweep k=2..20 against product
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from config import settings
from tools.topic_clustering import _normalise

# Which file + column to treat as ground truth. "product" is deliberately
# sourced from input.csv, not ground_truth.csv - it's descriptive metadata
# about the complaint (what it's about), not part of the answer key the
# pipeline is scored against for topic/sentiment extraction.
TRUTH_SOURCES = {
    "issue": (settings.GROUND_TRUTH_CSV, "Issue"),
    "product": (settings.INPUT_CSV, settings.PRODUCT_COLUMN),
}

# Categories below this many examples can't be recovered by any unsupervised
# method - there's no pattern to learn from a single example - so scoring
# against them only penalises ARI for a task that isn't learnable from this
# data, not for a real clustering failure. Excluded once, up front, so every
# k in the sweep is scored against the same fair set of categories.
MIN_CATEGORY_SUPPORT = 10


def sweep(
    k_values: range,
    truth: str = "issue",
    embedding_cache=settings.EMBEDDING_CACHE,
    min_category_support: int = MIN_CATEGORY_SUPPORT,
) -> pd.DataFrame:
    if not embedding_cache.exists():
        raise FileNotFoundError(
            f"{embedding_cache} not found. Run tools/topic_clustering.py at "
            "least once first so embeddings are cached."
        )

    truth_csv, truth_column = TRUTH_SOURCES[truth]

    cached = np.load(embedding_cache, allow_pickle=True)
    ids = list(cached["keys"])
    embeddings = _normalise(cached["embeddings"])

    truth_df = pd.read_csv(truth_csv)
    truth_by_id = dict(zip(truth_df[settings.ID_COLUMN], truth_df[truth_column]))
    true_labels = pd.Series([truth_by_id.get(i) for i in ids])

    matched_truth = true_labels.dropna()
    category_counts = matched_truth.value_counts()
    sparse = set(category_counts[category_counts < min_category_support].index)
    if sparse:
        n_dropped = int(matched_truth.isin(sparse).sum())
        print(f"  excluding {len(sparse)}/{len(category_counts)} categories with "
              f"< {min_category_support} examples ({n_dropped}/{len(matched_truth)} complaints)\n")

    scorable = true_labels.notna() & ~true_labels.isin(sparse)

    rows = []
    for k in k_values:
        if k >= len(ids):
            continue

        labels = KMeans(
            n_clusters=k, random_state=settings.RANDOM_SEED, n_init=10
        ).fit_predict(embeddings)

        pred_cluster = labels[scorable.to_numpy()]
        true_issue = true_labels[scorable].tolist()

        sil = silhouette_score(embeddings, labels, metric="cosine")
        ari = adjusted_rand_score(true_issue, pred_cluster)
        nmi = normalized_mutual_info_score(true_issue, pred_cluster)

        rows.append({"k": k, "silhouette": round(sil, 4),
                     "ari": round(ari, 4), "nmi": round(nmi, 4)})
        print(f"  k={k:<3} silhouette={sil:+.4f}   ARI={ari:+.4f}   NMI={nmi:.4f}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    truth_name = "issue"
    if args and args[-1] in TRUTH_SOURCES:
        truth_name = args.pop()

    if len(args) >= 2:
        lo, hi = int(args[0]), int(args[1])
        step = int(args[2]) if len(args) > 2 else 1
        k_range = range(lo, hi + 1, step)
    else:
        k_range = range(2, 31)

    truth_csv, truth_column = TRUTH_SOURCES[truth_name]
    print(f"sweeping against '{truth_column}' from {truth_csv.name} "
          f"({pd.read_csv(truth_csv)[truth_column].nunique()} categories)\n")

    results = sweep(k_range, truth=truth_name)

    best_ari = results.loc[results["ari"].idxmax()]
    best_sil = results.loc[results["silhouette"].idxmax()]

    print(f"\nbest k by ARI (agreement with real taxonomy): k={int(best_ari.k)}  "
          f"(ARI={best_ari.ari}, silhouette was {best_ari.silhouette})")
    print(f"best k by silhouette (geometric separation):   k={int(best_sil.k)}  "
          f"(silhouette={best_sil.silhouette}, ARI was {best_sil.ari})")
    if best_ari.k != best_sil.k:
        print("\nThe two disagree - use the ARI-selected k. Silhouette has no "
              "way to know what a 'correct' cluster looks like; ARI does.")
