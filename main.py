"""CLI entry point. Runs the full pipeline: extract -> cluster -> label -> plot.

The pipeline is a strict, non-branching sequence - step 2 always follows step
1, nothing ever runs conditionally or in parallel - so it's just four awaited
calls in order. No graph, no framework needed to express "do these four
things one after another": that's what Python statement order already means.

Each step's real return value is printed, not a paraphrase of it - this is
the thing the original project's LLM-narrated pipeline got wrong (it reported
"1,000 responses processed (for example)" against a 100-row file).

For the alternative where an Agent decides the step order itself by reading a
system prompt (closer to the original project's design, at the cost of that
same narration risk), see orchestration/agent_pipeline.py.

Usage:
    python main.py            # full run, all rows in data/input.csv
    python main.py 30         # smoke test on the first 30 rows only
"""

from __future__ import annotations

import asyncio
import json

from tools.business_insight import generate_insights
from tools.cluster_labelling import label_clusters
from tools.topic_clustering import cluster_topics
from tools.topic_extraction import extract_topics


async def main(limit: int | None = None) -> None:
    history = [
        {"step": "extract_topics", **await extract_topics(limit=limit)},
        {"step": "cluster_topics", **cluster_topics()},
        {"step": "label_clusters", **await label_clusters()},
        {"step": "generate_insights", **generate_insights()},
    ]

    print("\n=== Pipeline run complete ===")
    for entry in history:
        step = entry.pop("step")
        print(f"\n--- {step} ---")
        print(json.dumps(entry, indent=2, default=str))


if __name__ == "__main__":
    import sys

    row_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(main(limit=row_limit))
