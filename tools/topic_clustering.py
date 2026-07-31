"""Step 2 - group the topic phrases from Step 1 into semantic clusters.

Step 1 produces roughly 8,000 topic phrases across 3,000 complaints, and many of
them say the same thing in different words:

    "Obsolete account still reported"
    "Outdated tradeline not removed"
    "Account past 7-year reporting limit"

Counting those as three separate issues is useless. This step embeds each unique
phrase, then uses KMeans to group phrases whose vectors sit close together.

Differences from the original survey pipeline:
  * embeddings are cached to disk, so re-clustering at a different k is free
    rather than re-billing the entire corpus
  * vectors are L2-normalised, which makes Euclidean KMeans equivalent to cosine
    similarity - the correct metric for text embeddings
  * reports a silhouette score so k is a measured choice, not an assumption
"""

from __future__ import annotations

import ast
import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import settings
from models.llm_client import get_embed_client
from tools.topic_extraction import _is_retryable

# Conservative batch size - well under provider limits, and small enough that a
# transient failure costs one batch rather than the whole corpus.
#
# The embedding endpoint has its OWN quota, separate from the chat model's -
# on a free-tier key this has been observed as "100 embedding units/minute."
# A single 100-phrase batch consumes the entire window in one call, so the
# very next batch always 429s immediately afterward. That is not a rare edge
# case here, it is the expected steady state for any run with >100 unique
# topics - hence the retry loop below, not just a bigger batch size.
EMBED_BATCH = 100
EMBED_RETRIES = 6
EMBED_RETRY_DELAY = 20.0  # seconds; the observed quota window is ~1 minute


def _unique_topics(series: pd.Series) -> list[str]:
    """Flatten the per-row topic lists into one sorted, deduplicated vocabulary."""
    parsed = series.apply(ast.literal_eval)
    return sorted({t for row in parsed for t in row})


def _embed_batch(client, batch: list[str]) -> list[list[float]]:
    """One batch, with retry on the embedding endpoint's own rate limit."""
    kwargs: dict[str, Any] = {"model": settings.EMBED_MODEL, "input": batch}
    if settings.EMBED_DIMENSIONS:
        kwargs["dimensions"] = settings.EMBED_DIMENSIONS

    last_exc: Exception | None = None
    for attempt in range(EMBED_RETRIES):
        try:
            resp = client.embeddings.create(**kwargs)
            return [item.embedding for item in resp.data]
        except Exception as exc:  # noqa: BLE001 - retry loop, re-raised if exhausted
            last_exc = exc
            if not _is_retryable(exc) or attempt == EMBED_RETRIES - 1:
                raise
            print(f"    embedding batch rate-limited, waiting {EMBED_RETRY_DELAY:.0f}s "
                  f"(attempt {attempt + 1}/{EMBED_RETRIES})")
            time.sleep(EMBED_RETRY_DELAY)
    raise last_exc  # unreachable, satisfies type checkers


def _embed(topics: list[str]) -> np.ndarray:
    """Embed every phrase, in batches. Returns an (n_topics, dim) array."""
    client = get_embed_client()
    vectors: list[list[float]] = []

    for start in range(0, len(topics), EMBED_BATCH):
        batch = topics[start : start + EMBED_BATCH]
        vectors.extend(_embed_batch(client, batch))
        print(f"  embedded {min(start + EMBED_BATCH, len(topics)):>6,} / {len(topics):,}")

    return np.asarray(vectors, dtype=np.float32)


def _load_or_embed(topics: list[str]) -> np.ndarray:
    """Reuse cached vectors when the vocabulary is unchanged.

    Without this, sweeping k over 20 values costs 20 full embedding runs.
    """
    cache = settings.EMBEDDING_CACHE
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        if list(stored["topics"]) == topics:
            print(f"  loaded {len(topics):,} cached embeddings from {cache.name}")
            return stored["embeddings"]
        print("  vocabulary changed - re-embedding")

    embeddings = _embed(topics)
    np.savez(cache, topics=np.array(topics, dtype=object), embeddings=embeddings)
    return embeddings


def _normalise(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalise so Euclidean KMeans behaves as cosine clustering.

    KMeans only supports Euclidean distance, but direction - not magnitude -
    carries the meaning in a text embedding. On unit vectors the two orderings
    coincide, so normalising gets cosine behaviour out of a Euclidean algorithm.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def cluster_topics(
    input_csv=settings.TOPICS_CSV,
    output_csv=settings.CLUSTERS_CSV,
    n_clusters: int | None = None,
) -> dict[str, Any]:
    """Assign every topic phrase a cluster id and write the enriched CSV."""
    k = n_clusters or settings.N_CLUSTERS
    df = pd.read_csv(input_csv)

    topics = _unique_topics(df["all_topics_discussed"])
    print(f"{len(topics):,} unique topic phrases from {len(df):,} complaints")

    if len(topics) <= k:
        raise ValueError(
            f"Only {len(topics)} unique topics but k={k}. "
            "Run on more rows, or lower N_CLUSTERS."
        )

    embeddings = _normalise(_load_or_embed(topics))

    labels = KMeans(
        n_clusters=k, random_state=settings.RANDOM_SEED, n_init=10
    ).fit_predict(embeddings)

    # Geometric quality of this k. Not proof it is the right k - eval/choose_k.py
    # sweeps a range and checks alignment with the real CFPB taxonomy.
    silhouette = float(silhouette_score(embeddings, labels, metric="cosine"))

    topic_to_cluster = {t: int(c) for t, c in zip(topics, labels)}
    df["topic_cluster_ids"] = df["all_topics_discussed"].apply(
        lambda raw: [topic_to_cluster[t] for t in ast.literal_eval(raw)]
    )
    df.to_csv(output_csv, index=False)

    sizes = pd.Series(labels).value_counts().sort_index()
    return {
        "status": "success",
        "complaints": len(df),
        "unique_topics": len(topics),
        "clusters": k,
        "silhouette_cosine": round(silhouette, 4),
        "cluster_sizes": sizes.to_dict(),
        "largest_cluster_share": round(float(sizes.max() / sizes.sum()), 3),
        "embedding_model": settings.EMBED_MODEL,
        "dimensions": int(embeddings.shape[1]),
        "output_path": str(output_csv),
    }


if __name__ == "__main__":
    import sys

    k = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(cluster_topics(n_clusters=k), indent=2))
