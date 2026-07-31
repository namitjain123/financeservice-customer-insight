"""Score clustering quality against real CFPB category labels.

Silhouette (in tools/topic_clustering.py) answers "are my clusters geometrically
separated?" - a question you can answer with no external information at all.
It cannot tell you whether the clusters mean anything.

This module answers a different question: "do my clusters correspond to real,
meaningful categories?" That requires an answer key, which is why ground_truth.csv
exists and why input.csv never contains the Issue column - if it did, there'd be
no way to tell whether the pipeline learned the categories or just copied them.

Usage:
    python -m eval.cluster_eval
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from typing import Any

import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from config import settings


def _complaint_level_cluster(topic_cluster_ids: str) -> int:
    """Collapse a complaint's several topic-cluster-ids into one label.

    A complaint can raise 2-5 topics, each independently assigned to a cluster
    in Step 2. ARI/NMI compare one label per complaint against one true Issue
    per complaint, so this picks the complaint's most common cluster - i.e.
    its dominant theme.
    """
    ids = ast.literal_eval(topic_cluster_ids)
    return Counter(ids).most_common(1)[0][0]


def _cluster_to_issue_map(merged: pd.DataFrame) -> pd.DataFrame:
    """For each predicted cluster, what real Issue values does it contain?

    This is the diagnostic that turns "ARI is 0.31" into something actionable:
    it names which clusters are clean and which are junk drawers spanning many
    unrelated true categories - the failure mode that went undetected in the
    original survey project, where one theme silently absorbed 32% of the data.
    """
    rows = []
    for cluster_id, group in merged.groupby("pred_cluster"):
        counts = group["Issue"].value_counts()
        rows.append({
            "cluster": cluster_id,
            "size": len(group),
            "distinct_true_issues": group["Issue"].nunique(),
            "purity": round(counts.iloc[0] / len(group), 3),
            "dominant_issue": counts.index[0],
            "dominant_issue_share": round(counts.iloc[0] / len(group), 3),
        })
    return pd.DataFrame(rows).sort_values("purity")


def evaluate_clusters(
    clusters_csv=settings.CLUSTERS_CSV,
    ground_truth_csv=settings.GROUND_TRUTH_CSV,
) -> dict[str, Any]:
    """Compare predicted clusters against the real CFPB Issue taxonomy."""
    pred = pd.read_csv(clusters_csv)
    truth = pd.read_csv(ground_truth_csv)

    pred["pred_cluster"] = pred["topic_cluster_ids"].apply(_complaint_level_cluster)
    merged = pred.merge(truth, on=settings.ID_COLUMN, how="inner")

    if merged.empty:
        raise ValueError(
            "No complaint_ids matched between clusters and ground truth. "
            "Did clusters_csv come from a different run of input.csv?"
        )

    ari = adjusted_rand_score(merged["Issue"], merged["pred_cluster"])
    nmi = normalized_mutual_info_score(merged["Issue"], merged["pred_cluster"])

    cluster_map = _cluster_to_issue_map(merged)
    mean_purity = float(cluster_map["purity"].mean())

    # A cluster spanning many distinct true issues at low purity is the
    # junk-drawer failure mode - flag it by name, don't bury it in an average.
    #
    # The distinct-issue bar has to scale with how many true categories exist.
    # A fixed ">= 5" was fine when ground truth had ~22 issues, but with 54
    # true issues spread over only 12 clusters, fair division alone puts
    # ~4.5 issues in every cluster - a fixed "5" then fires on nearly
    # everything regardless of quality, and a flag that fires on 10 of 12
    # clusters has stopped telling you anything. Flag only clusters that are
    # unusually diffuse relative to that fair-division baseline, not ones
    # merely at it.
    n_true_issues = merged["Issue"].nunique()
    n_clusters = len(cluster_map)
    diversity_threshold = max(5, round(2 * n_true_issues / n_clusters))
    junk_drawers = cluster_map[
        (cluster_map["distinct_true_issues"] >= diversity_threshold)
        & (cluster_map["purity"] < 0.3)
    ]

    return {
        "n_complaints_evaluated": len(merged),
        "n_predicted_clusters": merged["pred_cluster"].nunique(),
        "n_true_issues": merged["Issue"].nunique(),
        "adjusted_rand_index": round(float(ari), 4),
        "normalized_mutual_info": round(float(nmi), 4),
        "mean_cluster_purity": round(mean_purity, 4),
        "junk_drawer_clusters": junk_drawers["cluster"].tolist(),
        "cluster_breakdown": cluster_map.to_dict(orient="records"),
    }


if __name__ == "__main__":
    result = evaluate_clusters()
    breakdown = result.pop("cluster_breakdown")
    print(json.dumps(result, indent=2))
    print("\nworst clusters first (lowest purity = most mixed):")
    print(pd.DataFrame(breakdown).to_string(index=False))
