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

Uses the cached embeddings from topic_clustering.py, so sweeping costs zero
additional API calls.

Usage:
    python -m eval.choose_k              # sweep k = 2..30
    python -m eval.choose_k 5 40 2        # sweep k = 5..40 step 2
"""

from __future__ import annotations

import ast
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from config import settings


def sweep(
    k_values: range,
    topics_csv=settings.TOPICS_CSV,
    ground_truth_csv=settings.GROUND_TRUTH_CSV,
    embedding_cache=settings.EMBEDDING_CACHE,
) -> pd.DataFrame:
    if not embedding_cache.exists():
        raise FileNotFoundError(
            f"{embedding_cache} not found. Run tools/topic_clustering.py at "
            "least once first so embeddings are cached."
        )

    cached = np.load(embedding_cache, allow_pickle=True)
    topics = list(cached["topics"])
    embeddings = cached["embeddings"]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    df = pd.read_csv(topics_csv)
    df["topics_parsed"] = df["all_topics_discussed"].apply(ast.literal_eval)
    truth = pd.read_csv(ground_truth_csv)
    truth_by_id = dict(zip(truth[settings.ID_COLUMN], truth["Issue"]))

    rows = []
    for k in k_values:
        if k >= len(topics):
            continue

        labels = KMeans(n_clusters=k, random_state=settings.RANDOM_SEED, n_init=10).fit_predict(
            embeddings
        )
        topic_to_cluster = {t: int(c) for t, c in zip(topics, labels)}

        pred_cluster, true_issue = [], []
        for _, row in df.iterrows():
            true = truth_by_id.get(row[settings.ID_COLUMN])
            if true is None:
                continue
            ids = [topic_to_cluster[t] for t in row["topics_parsed"]]
            pred_cluster.append(Counter(ids).most_common(1)[0][0])
            true_issue.append(true)

        sil = silhouette_score(embeddings, labels, metric="cosine")
        ari = adjusted_rand_score(true_issue, pred_cluster)
        nmi = normalized_mutual_info_score(true_issue, pred_cluster)

        rows.append({"k": k, "silhouette": round(sil, 4),
                     "ari": round(ari, 4), "nmi": round(nmi, 4)})
        print(f"  k={k:<3} silhouette={sil:+.4f}   ARI={ari:+.4f}   NMI={nmi:.4f}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        lo, hi = int(sys.argv[1]), int(sys.argv[2])
        step = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        k_range = range(lo, hi + 1, step)
    else:
        k_range = range(2, 31)

    results = sweep(k_range)

    best_ari = results.loc[results["ari"].idxmax()]
    best_sil = results.loc[results["silhouette"].idxmax()]

    print(f"\nbest k by ARI (agreement with real taxonomy): k={int(best_ari.k)}  "
          f"(ARI={best_ari.ari}, silhouette was {best_ari.silhouette})")
    print(f"best k by silhouette (geometric separation):   k={int(best_sil.k)}  "
          f"(silhouette={best_sil.silhouette}, ARI was {best_sil.ari})")
    if best_ari.k != best_sil.k:
        print("\nThe two disagree - use the ARI-selected k. Silhouette has no "
              "way to know what a 'correct' cluster looks like; ARI does.")
