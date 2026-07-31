"""Score clustering quality against real CFPB category labels.

Silhouette (in tools/topic_clustering.py) answers "are my clusters geometrically
separated?" - a question you can answer with no external information at all.
It cannot tell you whether the clusters mean anything.

This module answers a different question: "do my clusters correspond to real,
meaningful categories?" That requires an answer key, which is why ground_truth.csv
exists and why input.csv never contains the Issue column - if it did, there'd be
no way to tell whether the pipeline learned the categories or just copied them.

Two ground-truth granularities to score against - pick with a trailing arg,
same as eval/choose_k.py:
  issue     (default) the real CFPB Issue field - 60-80 fine-grained categories.
  product   input.csv's `product` column - 11 balanced categories.

Usage:
    python -m eval.cluster_eval              # score against Issue
    python -m eval.cluster_eval product       # score against product
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from config import settings
from eval.choose_k import TRUTH_SOURCES

# Categories below this many examples can't be recovered by any unsupervised
# method - there's no pattern to learn from a single example - so scoring
# against them only penalises ARI for a task that isn't learnable from this
# data, not for a real clustering failure. See eval/choose_k.py for the same
# filter applied during k selection.
MIN_CATEGORY_SUPPORT = 10


def _cluster_to_truth_map(merged: pd.DataFrame, truth_column: str) -> pd.DataFrame:
    """For each predicted cluster, what real category values does it contain?

    This is the diagnostic that turns "ARI is 0.31" into something actionable:
    it names which clusters are clean and which are junk drawers spanning many
    unrelated true categories - the failure mode that went undetected in the
    original survey project, where one theme silently absorbed 32% of the data.
    """
    rows = []
    for cluster_id, group in merged.groupby("pred_cluster"):
        counts = group[truth_column].value_counts()
        rows.append({
            "cluster": cluster_id,
            "size": len(group),
            "distinct_true_categories": group[truth_column].nunique(),
            "purity": round(counts.iloc[0] / len(group), 3),
            "dominant_category": counts.index[0],
            "dominant_category_share": round(counts.iloc[0] / len(group), 3),
        })
    return pd.DataFrame(rows).sort_values("purity")


def evaluate_clusters(
    clusters_csv=settings.CLUSTERS_CSV,
    truth: str = "issue",
    min_category_support: int = MIN_CATEGORY_SUPPORT,
) -> dict[str, Any]:
    """Compare predicted clusters against a real CFPB category taxonomy."""
    truth_csv, truth_column = TRUTH_SOURCES[truth]

    pred = pd.read_csv(clusters_csv)
    truth_df = pd.read_csv(truth_csv)[[settings.ID_COLUMN, truth_column]]

    pred["pred_cluster"] = pred["cluster_id"]
    # `product` already exists in pred (pass-through from input.csv upstream)
    # when scoring against product truth - drop it so the merge doesn't
    # collide and suffix into product_x/product_y.
    if truth_column in pred.columns:
        pred = pred.drop(columns=[truth_column])
    merged = pred.merge(truth_df, on=settings.ID_COLUMN, how="inner")

    if merged.empty:
        raise ValueError(
            "No complaint_ids matched between clusters and ground truth. "
            "Did clusters_csv come from a different run of input.csv?"
        )

    n_categories_total = merged[truth_column].nunique()
    category_counts = merged[truth_column].value_counts()
    sparse_categories = category_counts[category_counts < min_category_support]
    scored = merged[~merged[truth_column].isin(sparse_categories.index)]

    ari = adjusted_rand_score(scored[truth_column], scored["pred_cluster"])
    nmi = normalized_mutual_info_score(scored[truth_column], scored["pred_cluster"])

    cluster_map = _cluster_to_truth_map(scored, truth_column)
    mean_purity = float(cluster_map["purity"].mean())

    # A cluster spanning many distinct true categories at low purity is the
    # junk-drawer failure mode - flag it by name, don't bury it in an average.
    #
    # The distinct-category bar has to scale with how many true categories
    # exist. A fixed ">= 5" was fine when ground truth had ~22 issues, but
    # with 54 true issues spread over only 12 clusters, fair division alone
    # puts ~4.5 issues in every cluster - a fixed "5" then fires on nearly
    # everything regardless of quality, and a flag that fires on 10 of 12
    # clusters has stopped telling you anything. Flag only clusters that are
    # unusually diffuse relative to that fair-division baseline, not ones
    # merely at it.
    n_true_categories = scored[truth_column].nunique()
    n_clusters = len(cluster_map)
    diversity_threshold = max(5, round(2 * n_true_categories / n_clusters))
    junk_drawers = cluster_map[
        (cluster_map["distinct_true_categories"] >= diversity_threshold)
        & (cluster_map["purity"] < 0.3)
    ]

    return {
        "truth": truth,
        "truth_column": truth_column,
        "min_category_support": min_category_support,
        "n_categories_total": n_categories_total,
        "n_categories_scored": n_true_categories,
        "n_categories_excluded_as_sparse": len(sparse_categories),
        "n_complaints_total": len(merged),
        "n_complaints_evaluated": len(scored),
        "n_predicted_clusters": scored["pred_cluster"].nunique(),
        "n_true_categories": n_true_categories,
        "adjusted_rand_index": round(float(ari), 4),
        "normalized_mutual_info": round(float(nmi), 4),
        "mean_cluster_purity": round(mean_purity, 4),
        "junk_drawer_clusters": junk_drawers["cluster"].tolist(),
        "cluster_breakdown": cluster_map.to_dict(orient="records"),
    }


if __name__ == "__main__":
    import sys

    truth_name = sys.argv[1] if len(sys.argv) > 1 else "issue"
    result = evaluate_clusters(truth=truth_name)
    breakdown = result.pop("cluster_breakdown")
    print(json.dumps(result, indent=2))
    print("\nworst clusters first (lowest purity = most mixed):")
    print(pd.DataFrame(breakdown).to_string(index=False))
