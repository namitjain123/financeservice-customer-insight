"""Step 3 - turn numbered clusters into business-readable theme names.

Step 2 leaves every complaint tagged with a bare integer, e.g. cluster `4`.
That means nothing to a reader. This step shows each cluster's topic phrases
to the LLM and asks for a short label - "Foreclosure process failures",
"Credit reporting accuracy disputes" - then attaches that label to every
complaint in the cluster.

Differences from the original survey pipeline this replaces:
  * real JSON mode instead of a bare json.loads() on unvalidated content
  * same model-fallback chain as Step 1, so one cluster failing doesn't
    abort labelling for the rest
  * concurrent labelling (there are only a few dozen clusters, so this is
    one batch of calls, not thousands)
  * reports which clusters got a label vs. fell back to a generic name
"""

from __future__ import annotations

import ast
import asyncio
import json
from typing import Any

import pandas as pd
from openai import AsyncOpenAI

from config import settings
from models.llm_client import get_async_client
from tools.topic_extraction import MODEL_CHAIN, _is_retryable

SYSTEM_PROMPT = """You assign a short, business-friendly theme label to a cluster of \
related complaint topics.

You will be given a sample of topic phrases that all belong to the SAME cluster.

Requirements:
- 2-5 words, noun phrase, business-friendly (this will appear on an executive chart).
- Generalise across the listed topics - do not just repeat one of them verbatim.
- Describe the shared problem, not the product or company.
- Return STRICT JSON only: {"label": "<Label>"}

Example:
Topics: Dispute closed without investigation; Requested documents not provided; \
Investigation completed in one day
{"label": "Inadequate dispute investigation"}"""

# More than a handful of examples doesn't sharpen the label further and just
# spends tokens - the LLM needs to see the theme, not read every instance of it.
MAX_SAMPLE_TOPICS = 15


async def _label_one(client: AsyncOpenAI, cluster_id: int, topics: list[str]) -> dict[str, Any]:
    """Ask the model for one cluster's label. Falls back through MODEL_CHAIN."""
    sample = "; ".join(list(dict.fromkeys(topics))[:MAX_SAMPLE_TOPICS])
    last_error = "unknown"

    for model in MODEL_CHAIN:
        for attempt in range(settings.RETRIES_PER_MODEL):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Topics: {sample}"},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                content = resp.choices[0].message.content
                if not content:
                    raise ValueError("model returned empty content")

                data = json.loads(content)
                if isinstance(data, list):
                    data = next((d for d in data if isinstance(d, dict)), None) or {}
                label = str(data.get("label", "")).strip()
                if not label:
                    raise ValueError("no label returned")

                return {"cluster_id": cluster_id, "ok": True, "label": label}
            except Exception as exc:  # noqa: BLE001 - per-cluster isolation
                last_error = f"{model} - {type(exc).__name__}: {exc}"
                if not _is_retryable(exc):
                    break
                if attempt < settings.RETRIES_PER_MODEL - 1:
                    await asyncio.sleep(settings.RETRY_BASE_DELAY * (2**attempt))

    # A cluster without a real label still needs a name downstream - fall
    # back to something identifiable rather than dropping the cluster.
    return {"cluster_id": cluster_id, "ok": False, "label": f"Theme {cluster_id}",
            "error": last_error}


def _cluster_topic_map(df: pd.DataFrame) -> dict[int, list[str]]:
    """cluster_id -> every topic phrase from complaints assigned to it.

    A complaint's cluster_id is one value (Step 2 clusters complaints, not
    individual phrases) - all of that complaint's topics are still useful
    context for the LLM to see what the cluster is about.
    """
    topics_series = df["all_topics_discussed"].apply(ast.literal_eval)

    clusters: dict[int, list[str]] = {}
    for topics, cid in zip(topics_series, df["cluster_id"]):
        clusters.setdefault(int(cid), []).extend(str(t) for t in topics)
    return clusters


def _label_rows(df: pd.DataFrame, labels: dict[int, str]) -> pd.DataFrame:
    """One row per complaint, carrying its cluster's label.

    Step 2 now clusters complaints directly (one cluster_id per complaint),
    so there's no per-topic fan-out to reconstruct here - unlike the earlier
    per-topic-cluster design, a complaint contributes to exactly one theme.
    """
    out = df.copy()
    out["general_topic_l1"] = out["cluster_id"].map(labels)
    out["topic_discussed"] = out["all_topics_discussed"].apply(
        lambda raw: "; ".join(ast.literal_eval(raw))
    )
    return out


async def label_clusters(
    input_csv=settings.CLUSTERS_CSV,
    output_csv=settings.OUTPUT_CSV,
) -> dict[str, Any]:
    """Label every cluster and write the exploded, theme-tagged CSV."""
    df = pd.read_csv(input_csv)
    clusters = _cluster_topic_map(df)
    print(f"labelling {len(clusters)} clusters from {len(df):,} complaints")

    client = get_async_client()
    try:
        results = await asyncio.gather(
            *(_label_one(client, cid, topics) for cid, topics in clusters.items())
        )
    finally:
        await client.close()

    labels = {r["cluster_id"]: r["label"] for r in results}
    fallbacks = [r for r in results if not r["ok"]]

    out = _label_rows(df, labels)
    out.to_csv(output_csv, index=False)

    theme_counts = out["general_topic_l1"].value_counts()
    return {
        "status": "success",
        "input_complaints": len(df),
        "clusters_labelled": len(labels),
        "clusters_using_fallback_name": len(fallbacks),
        "fallback_cluster_ids": [f["cluster_id"] for f in fallbacks],
        "output_rows": len(out),
        "theme_counts": theme_counts.to_dict(),
        "largest_theme_share": round(float(theme_counts.max() / theme_counts.sum()), 3),
        "output_path": str(output_csv),
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(label_clusters()), indent=2))
