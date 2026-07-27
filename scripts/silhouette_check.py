"""Justify the number of clusters (k) used in topic_clustering.py.

Embeds the unique topic phrases once, caches them, then sweeps k and reports
mean silhouette + inertia. Also draws the per-cluster silhouette diagram for
the best k.

Usage:
    set OPENAI_EMBEDDING_API_KEY in .env, then:
    python silhouette_check.py
"""

import ast
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

load_dotenv()

PROJECT = Path(r"e:\customer insights\Customer-Survey-Insight")
INPUT_CSV = PROJECT / "data" / "df_with_topics.csv"
TOPICS_COLUMN = "all_topics_discussed"
CACHE = Path(__file__).parent / "topic_embeddings.npz"
K_RANGE = range(2, 26)
METRIC = "cosine"  # OpenAI embeddings are unit-normalised; cosine is the honest metric


def load_unique_topics() -> list[str]:
    df = pd.read_csv(INPUT_CSV)
    series = df[TOPICS_COLUMN].apply(ast.literal_eval)
    return sorted({t for lst in series for t in lst})


def get_embeddings(topics: list[str]) -> np.ndarray:
    """Embed once, cache to disk. A k-sweep must not cost one API call per k."""
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=True)
        if list(cached["topics"]) == topics:
            print(f"Loaded {len(topics)} cached embeddings from {CACHE.name}")
            return cached["embeddings"]

    client = OpenAI(api_key=os.getenv("OPENAI_EMBEDDING_API_KEY"))
    response = client.embeddings.create(input=topics, model="text-embedding-3-small")
    embeddings = np.array([item.embedding for item in response.data])
    np.savez(CACHE, topics=np.array(topics, dtype=object), embeddings=embeddings)
    print(f"Embedded {len(topics)} topics, cached to {CACHE.name}")
    return embeddings


def sweep(embeddings: np.ndarray) -> tuple[list[float], list[float]]:
    scores, inertias = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, metric=METRIC)
        scores.append(score)
        inertias.append(km.inertia_)
        print(f"k={k:3d}   silhouette={score:+.4f}   inertia={km.inertia_:9.2f}")
    return scores, inertias


def plot_sweep(scores: list[float], inertias: list[float], best_k: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ks = list(K_RANGE)
    ax1.plot(ks, scores, marker="o")
    ax1.axvline(best_k, color="crimson", linestyle="--", label=f"best k = {best_k}")
    ax1.axvline(12, color="gray", linestyle=":", label="current k = 12")
    ax1.set(xlabel="k", ylabel=f"mean silhouette ({METRIC})",
            title="Silhouette score vs k")
    ax1.legend()

    ax2.plot(ks, inertias, marker="o", color="darkorange")
    ax2.set(xlabel="k", ylabel="inertia (within-cluster SSE)",
            title="Elbow curve")

    fig.tight_layout()
    fig.savefig(Path(__file__).parent / "k_sweep.png", dpi=150)
    print("Wrote k_sweep.png")


def plot_silhouette_diagram(embeddings: np.ndarray, k: int) -> None:
    """Per-cluster silhouette bars: shows WHICH cluster is bad, not just whether k is."""
    labels = KMeans(n_clusters=k, random_state=0, n_init=10).fit_predict(embeddings)
    values = silhouette_samples(embeddings, labels, metric=METRIC)
    mean = values.mean()

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(embeddings) / 4 + 2))
    y_lower = 10
    for i in range(k):
        cluster_values = np.sort(values[labels == i])
        size = len(cluster_values)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_values, alpha=0.75)
        ax.text(-0.05, y_lower + size / 2, str(i), va="center", fontsize=9)
        y_lower = y_upper + 10

    ax.axvline(mean, color="crimson", linestyle="--",
               label=f"mean = {mean:+.3f}")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="silhouette coefficient", ylabel="cluster",
           title=f"Per-cluster silhouette, k={k}")
    ax.set_yticks([])
    ax.legend()

    fig.tight_layout()
    fig.savefig(Path(__file__).parent / f"silhouette_k{k}.png", dpi=150)
    print(f"Wrote silhouette_k{k}.png")

    # Flag clusters that are dragging the score down.
    print("\nPer-cluster breakdown:")
    for i in range(k):
        cluster_values = values[labels == i]
        negatives = (cluster_values < 0).sum()
        print(f"  cluster {i:2d}: n={len(cluster_values):3d}  "
              f"mean={cluster_values.mean():+.3f}  negative={negatives}")


def main() -> None:
    topics = load_unique_topics()
    print(f"{len(topics)} unique topic phrases\n")

    embeddings = get_embeddings(topics)
    print()

    scores, inertias = sweep(embeddings)
    best_k = list(K_RANGE)[int(np.argmax(scores))]
    print(f"\nBest k by silhouette: {best_k} (score {max(scores):+.4f})")
    print(f"Current hardcoded k=12 scores {scores[list(K_RANGE).index(12)]:+.4f}")

    plot_sweep(scores, inertias, best_k)
    plot_silhouette_diagram(embeddings, best_k)
    if best_k != 12:
        plot_silhouette_diagram(embeddings, 12)


if __name__ == "__main__":
    main()
