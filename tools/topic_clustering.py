"""Step 2 - group complaints into semantic clusters by what they discuss.

Each complaint's full narrative is embedded directly (one vector per
complaint) and KMeans clusters those vectors. Step 1's extracted topic
phrases are NOT what gets embedded here - they're still used downstream
(Step 3 samples them to name each cluster; Step 4 uses them for the
per-theme charts), but the clustering signal itself comes from the raw
complaint text.

This is a deliberate tradeoff, not free lunch. An earlier version of this
module embedded the topic PHRASES instead (deliberately abstracted away from
product-specific wording, e.g. "unauthorized charge" rather than "credit
card annual fee dispute"), which kept clusters focused on cross-product
THEMES but capped ARI against CFPB's real Issue/product labels at around
0.10 - too low to trust the resulting report. Narratives carry the specific
vocabulary (mortgage, repossession, escrow, chargeback) that actually
separates CFPB's categories, which is what makes the eval score meaningful -
but it also means clusters will tend to track known product/issue lines
more than they surface novel cross-cutting themes. See eval/cluster_eval.py
and the project README for how to read the resulting ARI honestly given
that tradeoff.

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

# Voyage's free tier caps at 3 requests/minute AND 10K tokens/minute - both
# limits, not just one. A fixed item-count batch can't respect the token cap
# when item length varies a lot: complaint narratives range from ~230 to
# ~2,900 characters, so a "batch of 30" can land anywhere from ~2,000 to
# ~10,800 estimated tokens depending on which 30 narratives happen to fall
# together. An oversized batch is a dead end, not a transient failure - the
# retry loop below can't fix it by waiting, because retrying resends the same
# oversized request. _batch_by_tokens groups by an actual token budget
# instead, so every batch sent is safely under the cap regardless of content.
EMBED_MAX_ITEMS_PER_BATCH = 100
EMBED_MAX_TOKENS_PER_BATCH = 6000  # well under the 10K cap - char/4 is a rough estimate
EMBED_RETRIES = 6
EMBED_RETRY_DELAY = 20.0  # seconds; the observed quota window is ~1 minute

# The RPM cap (not just TPM) means requests must be paced even when each one
# is individually small enough - bursting batches back-to-back trips it
# regardless of size. 21s keeps us under 3/minute with a small margin.
EMBED_PACING_DELAY = 21.0


def _batch_by_tokens(texts: list[str]) -> list[list[str]]:
    """Group texts so each batch stays under both the item and token budget.

    len(text) // 4 is a rough token estimate (English BPE tokenizers average
    roughly 4 chars/token), deliberately conservative given EMBED_MAX_TOKENS_PER_BATCH
    already sits well below the real 10K cap.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for text in texts:
        est_tokens = max(1, len(text) // 4)
        would_exceed = (
            len(current) >= EMBED_MAX_ITEMS_PER_BATCH
            or (current and current_tokens + est_tokens > EMBED_MAX_TOKENS_PER_BATCH)
        )
        if would_exceed:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += est_tokens

    if current:
        batches.append(current)
    return batches


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


def _embed(texts: list[str]) -> np.ndarray:
    """Embed every text, in token-budgeted batches. Returns an (n_texts, dim) array."""
    client = get_embed_client()
    vectors: list[list[float]] = []
    batches = _batch_by_tokens(texts)
    done = 0

    for i, batch in enumerate(batches):
        if i > 0:
            time.sleep(EMBED_PACING_DELAY)  # proactive - don't wait for a 429 to learn the RPM cap
        vectors.extend(_embed_batch(client, batch))
        done += len(batch)
        print(f"  embedded {done:>6,} / {len(texts):,}")

    return np.asarray(vectors, dtype=np.float32)


def _load_or_embed(keys: list, texts: list[str]) -> np.ndarray:
    """Reuse cached vectors when the input is unchanged.

    `keys` is the identity check (complaint_ids) - `texts` is what actually
    gets embedded (narratives). Kept separate because comparing ~1,000 ids
    for cache-hit equality is cheap; comparing the narratives themselves
    would work too but there's no reason to re-hash paragraphs on every run.
    Without this cache, sweeping k over 20 values would cost 20 full
    embedding runs.
    """
    cache = settings.EMBEDDING_CACHE
    if cache.exists():
        stored = np.load(cache, allow_pickle=True)
        if list(stored["keys"]) == keys:
            print(f"  loaded {len(keys):,} cached embeddings from {cache.name}")
            return stored["embeddings"]
        print("  input changed - re-embedding")

    embeddings = _embed(texts)
    np.savez(cache, keys=np.array(keys, dtype=object), embeddings=embeddings)
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
    """Assign every complaint a cluster id and write the enriched CSV."""
    k = n_clusters or settings.N_CLUSTERS
    df = pd.read_csv(input_csv)

    unique_topics = _unique_topics(df["all_topics_discussed"])
    print(f"{len(unique_topics):,} unique topic phrases from {len(df):,} complaints")

    if len(df) <= k:
        raise ValueError(
            f"Only {len(df)} complaints but k={k}. "
            "Run on more rows, or lower N_CLUSTERS."
        )

    ids = df[settings.ID_COLUMN].tolist()
    narratives = df[settings.TEXT_COLUMN].astype(str).tolist()
    embeddings = _normalise(_load_or_embed(ids, narratives))

    labels = KMeans(
        n_clusters=k, random_state=settings.RANDOM_SEED, n_init=10
    ).fit_predict(embeddings)

    # Geometric quality of this k. Not proof it is the right k - eval/choose_k.py
    # sweeps a range and checks alignment with the real CFPB taxonomy.
    silhouette = float(silhouette_score(embeddings, labels, metric="cosine"))

    df["cluster_id"] = labels.astype(int)
    df.to_csv(output_csv, index=False)

    sizes = pd.Series(labels).value_counts().sort_index()
    return {
        "status": "success",
        "complaints": len(df),
        "unique_topics": len(unique_topics),
        "clusters": k,
        "silhouette_cosine": round(silhouette, 4),
        "cluster_sizes": sizes.to_dict(),
        "largest_cluster_share": round(float(sizes.max() / sizes.sum()), 3),
        "embedding_model": settings.EMBED_MODEL,
        "embedding_input": "narrative",
        "dimensions": int(embeddings.shape[1]),
        "output_path": str(output_csv),
    }


if __name__ == "__main__":
    import sys

    k = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(cluster_topics(n_clusters=k), indent=2))
