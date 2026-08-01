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

Also computes the classic elbow method (inertia/WCSS vs k) alongside ARI and
saves both as one plot to artifacts/plot_elbow_k_selection.png - included for
comparison, not because it's what k gets picked by here. See plot_elbow()'s
docstring for why the elbow is shown but not trusted.

Usage:
    python -m eval.choose_k                       # sweep k=2..30 against Issue
    python -m eval.choose_k 5 40 2                 # sweep k=5..40 step 2 against Issue
    python -m eval.choose_k 2 20 1 product          # sweep k=2..20 against product
"""

from __future__ import annotations

import matplotlib.pyplot as plt
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

        km = KMeans(n_clusters=k, random_state=settings.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(embeddings)

        pred_cluster = labels[scorable.to_numpy()]
        true_issue = true_labels[scorable].tolist()

        sil = silhouette_score(embeddings, labels, metric="cosine")
        ari = adjusted_rand_score(true_issue, pred_cluster)
        nmi = normalized_mutual_info_score(true_issue, pred_cluster)

        rows.append({"k": k, "silhouette": round(sil, 4), "ari": round(ari, 4),
                     "nmi": round(nmi, 4), "inertia": round(float(km.inertia_), 2)})
        print(f"  k={k:<3} silhouette={sil:+.4f}   ARI={ari:+.4f}   "
              f"NMI={nmi:.4f}   inertia={km.inertia_:,.1f}")

    return pd.DataFrame(rows)


def _estimate_elbow(ks: np.ndarray, values: np.ndarray) -> int:
    """Kneedle-style elbow: the k farthest from the straight line joining
    the curve's two endpoints, after scaling both axes to [0, 1] so k's
    small range and inertia's large one don't distort "farthest."

    This is a heuristic, not a formula with one correct answer - a curve
    that declines smoothly with no sharp bend has no clean elbow, and this
    will still return *a* k, just not necessarily a meaningful one. Read it
    alongside the plot, not as a number to trust blindly.
    """
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (values - values.min()) / (values.max() - values.min())
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    # Perpendicular distance from each point to the line through the endpoints.
    num = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    den = np.hypot(y2 - y1, x2 - x1)
    distances = num / den
    return int(ks[np.argmax(distances)])


def plot_elbow(results: pd.DataFrame, output_path=None) -> str:
    """Save the elbow plot (inertia vs k), next to ARI vs k for contrast.

    Elbow method: inertia is the within-cluster sum of squares -
        WCSS(k) = sum over every point x of ||x - centroid(x)||^2
    where centroid(x) is the center of whichever of the k clusters x landed
    in. WCSS always falls as k grows (more clusters can only fit the data
    at least as well), so there's no single "correct" k to read off it -
    only a point where it stops falling fast, the "elbow."

    Plotted next to ARI, not instead of it, because the elbow is purely
    geometric - like silhouette (see this module's docstring), it has no
    way to know whether a cluster corresponds to a real category, only
    whether points sit tightly around their centroid. It can point at a k
    that looks tidy in embedding space while still poorly matching CFPB's
    real categories - which is exactly why ARI, not this, is what
    eval/choose_k.py and eval/cluster_eval.py actually select k by.
    """
    output_path = output_path or (settings.ARTIFACTS / "plot_elbow_k_selection.png")
    ks = results["k"].to_numpy()

    elbow_k = _estimate_elbow(ks, results["inertia"].to_numpy())
    best_ari_k = int(results.loc[results["ari"].idxmax(), "k"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(ks, results["inertia"], marker="o", color="darkorange")
    ax1.axvline(elbow_k, color="crimson", linestyle="--", label=f"elbow k={elbow_k}")
    ax1.set(xlabel="k", ylabel="inertia (within-cluster sum of squares)",
            title="Elbow method - geometric only, no ground truth")
    ax1.legend()

    ax2.plot(ks, results["ari"], marker="o", color="#4C78A8")
    ax2.axvline(best_ari_k, color="crimson", linestyle="--", label=f"best ARI k={best_ari_k}")
    ax2.set(xlabel="k", ylabel="Adjusted Rand Index",
            title="ARI vs real CFPB categories - what this project trusts")
    ax2.legend()

    fig.suptitle("k selection: geometric heuristic vs ground-truth-aware score")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


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
    elbow_k = _estimate_elbow(results["k"].to_numpy(), results["inertia"].to_numpy())

    print(f"\nbest k by ARI (agreement with real taxonomy): k={int(best_ari.k)}  "
          f"(ARI={best_ari.ari}, silhouette was {best_ari.silhouette})")
    print(f"best k by silhouette (geometric separation):   k={int(best_sil.k)}  "
          f"(silhouette={best_sil.silhouette}, ARI was {best_sil.ari})")
    print(f"elbow k by inertia (geometric, no ground truth): k={elbow_k}  "
          f"(ARI at that k was {results.loc[results['k'] == elbow_k, 'ari'].iloc[0]})")
    if best_ari.k != best_sil.k:
        print("\nThe three can disagree - use the ARI-selected k. Silhouette and the "
              "elbow have no way to know what a 'correct' cluster looks like; ARI does.")

    plot_path = plot_elbow(results)
    print(f"\nelbow plot saved to {plot_path}")
