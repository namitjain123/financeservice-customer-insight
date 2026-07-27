"""Step 1 - extract topics, sentiment and a supporting quote from each complaint.

Turns prose into structured fields:

    "I am filing a complaint regarding an obsolete account being reported on my
     credit file. The account is 7 years and 5 months old, which exceeds the
     reporting limit under FCRA Section 605(a)."
                                  |
                                  v
    topics    ["Obsolete account still reported",
               "FCRA reporting time limit exceeded"]
    sentiment "Negative"

Differences from the original survey pipeline this replaces:
  * concurrent, not serial
  * structured JSON output instead of stripping markdown fences by hand
  * one bad row is recorded and skipped, not fatal after paying for 2,999 calls
  * retries on 429/5xx, which the Gemini free tier makes unavoidable
  * falls through to CHAT_MODEL_FALLBACKS when the primary model's quota is
    exhausted, rather than stalling for the rest of the run
  * resumable - already-processed complaints are skipped on a re-run, AND
    checkpointed every CHECKPOINT_EVERY rows within a single run, so a crash
    partway through a 3,000-row pass loses one chunk, not the whole run
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI

from config import settings
from models.llm_client import get_async_client

SYSTEM_PROMPT = """You extract the distinct issues a consumer raises in a financial services complaint.

Return STRICT JSON with these keys:
  topics            array of 2-5 short phrases, 2-6 words each, each a distinct issue
  supporting_quote  <=160 chars, verbatim from the text
  reason            1-2 sentences explaining the extraction
  sentiment         exactly one of: Negative, Neutral, Positive

Rules:
- Each topic must be specific enough to act on. "Bad service" is useless;
  "Repeated failed dispute investigation" is actionable.
- Describe the ISSUE, not the product or company.
- The text contains [REDACTED], [DATE] and [AMOUNT] placeholders where personal
  details were removed before publication. Ignore them; never treat them as topics.
- If the complaint genuinely raises only one issue, return one topic.

Example:
Text: "I disputed three charges on my card. The bank closed the dispute in two days
without contacting me and never sent the documents I asked for."
{"topics": ["Dispute closed without investigation", "Requested documents not provided"],
 "supporting_quote": "The bank closed the dispute in two days without contacting me",
 "reason": "Two distinct failures: inadequate investigation and non-delivery of records.",
 "sentiment": "Negative"}"""

VALID_SENTIMENTS = ("Negative", "Neutral", "Positive")


def _is_retryable(exc: Exception) -> bool:
    """429 (quota), 5xx, and network blips are worth retrying. A 404 never is."""
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    return status == 429 or (status is not None and 500 <= status < 600)


async def _call_model(client: AsyncOpenAI, model: str, text: str) -> dict[str, Any]:
    """One model call plus response validation. Raises on any failure."""
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = resp.choices[0].message.content
    if not content:
        raise ValueError("model returned empty content")

    data = json.loads(content)

    # JSON mode guarantees the response parses, not that it is an object. The
    # model sometimes returns a bare array, or an array wrapping the object.
    if isinstance(data, list):
        data = next((d for d in data if isinstance(d, dict)), None) or {"topics": data}
    if not isinstance(data, dict):
        raise ValueError(f"unexpected JSON shape: {type(data).__name__}")

    # JSON mode guarantees the response parses. It guarantees nothing about
    # whether the contents match what we asked for.
    topics = [str(t).strip() for t in data.get("topics", []) if str(t).strip()]
    topics = list(dict.fromkeys(topics))[: settings.MAX_TOPICS_PER_ROW]
    if not topics:
        raise ValueError("no topics returned")

    sentiment = data.get("sentiment", "Neutral")
    if sentiment not in VALID_SENTIMENTS:
        # An unexpected value would silently add a category to the severity chart.
        sentiment = "Neutral"

    return {
        "ok": True,
        "topics": topics,
        "supporting_quote": (data.get("supporting_quote") or "")[:160],
        "reason": data.get("reason") or "",
        "sentiment": sentiment,
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }


MODEL_CHAIN = [settings.CHAT_MODEL, *settings.CHAT_MODEL_FALLBACKS]


async def _extract_one(
    client: AsyncOpenAI, sem: asyncio.Semaphore, idx: int, text: str
) -> dict[str, Any]:
    """Extract one complaint. Tries each model in MODEL_CHAIN in turn. Never raises."""
    async with sem:
        last_error = "unknown"
        for model in MODEL_CHAIN:
            for attempt in range(settings.RETRIES_PER_MODEL):
                try:
                    result = await _call_model(client, model, text)
                    return {"idx": idx, "model_used": model, **result}
                except Exception as exc:  # noqa: BLE001 - deliberate per-row isolation
                    last_error = f"{model} - {type(exc).__name__}: {exc}"
                    if not _is_retryable(exc):
                        break  # this model won't recover; try the next one now
                    if attempt < settings.RETRIES_PER_MODEL - 1:
                        # Free-tier quotas are per-minute, so the backoff has to
                        # be long enough to actually cross the quota window.
                        await asyncio.sleep(settings.RETRY_BASE_DELAY * (2**attempt))
            print(f"  row {idx}: {model} exhausted, falling through - {last_error[:100]}")
        return {"idx": idx, "ok": False, "error": last_error}


CHECKPOINT_EVERY = 100  # rows per chunk; tune down if a single call is slow


def _already_done(output_csv) -> set:
    """complaint_ids present in a previous run's output (or an earlier checkpoint
    from THIS run, since checkpoints are written to the same file)."""
    if not output_csv.exists():
        return set()
    try:
        return set(pd.read_csv(output_csv)[settings.ID_COLUMN])
    except Exception:  # noqa: BLE001 - a corrupt partial file just means start over
        return set()


def _attach_results(chunk: pd.DataFrame, results: list[dict[str, Any]]) -> pd.DataFrame:
    """Merge model results onto a chunk's rows and drop any that failed outright."""
    by_idx = {r["idx"]: r for r in results}
    n = len(chunk)
    chunk = chunk.copy()
    chunk["all_topics_discussed"] = [by_idx[i].get("topics", []) for i in range(n)]
    chunk["supporting_quote"] = [by_idx[i].get("supporting_quote", "") for i in range(n)]
    chunk["ai_topic_reasoning"] = [by_idx[i].get("reason", "") for i in range(n)]
    chunk["customer_sentiment"] = [by_idx[i].get("sentiment", "Neutral") for i in range(n)]
    chunk["extraction_ok"] = [by_idx[i]["ok"] for i in range(n)]
    chunk["process_time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Records which model in the chain actually served each row - if the
    # primary's daily quota ran out mid-run, this is how you'd see it happened.
    chunk["model_version"] = [by_idx[i].get("model_used", "") for i in range(n)]

    # A row with no topics contributes nothing to the explode in Step 3, and
    # would quietly distort the counts if kept.
    return chunk[chunk["extraction_ok"]].copy()


async def extract_topics(
    input_csv=settings.INPUT_CSV,
    output_csv=settings.TOPICS_CSV,
    limit: int | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Read complaints, extract topics concurrently, write the enriched CSV.

    Processes `todo` in CHECKPOINT_EVERY-row chunks and appends each chunk to
    disk as soon as it completes. A crash on row 2,400 of 3,000 costs at most
    the in-flight chunk (~100 rows), not the preceding 2,300 - and the next
    invocation resumes from whatever made it to disk.
    """
    df = pd.read_csv(input_csv)
    if limit:
        df = df.head(limit).copy()

    if not resume and output_csv.exists():
        output_csv.unlink()  # otherwise the append below would corrupt a stale file

    done = _already_done(output_csv) if resume else set()
    todo = df[~df[settings.ID_COLUMN].isin(done)].copy()

    if done:
        print(f"resuming: {len(done):,} already done, {len(todo):,} remaining")
    if todo.empty:
        return {"status": "success", "rows_in": len(df), "rows_out": len(done),
                "failed": 0, "note": "nothing to do - all rows already processed"}

    client = get_async_client()
    sem = asyncio.Semaphore(settings.EXTRACTION_CONCURRENCY)

    failures: list[dict[str, Any]] = []
    newly_processed = 0
    prompt_tokens = completion_tokens = 0
    file_has_header = output_csv.exists() and bool(done)

    try:
        chunks = [todo.iloc[i : i + CHECKPOINT_EVERY] for i in range(0, len(todo), CHECKPOINT_EVERY)]
        for n, chunk in enumerate(chunks, 1):
            texts = chunk[settings.TEXT_COLUMN].fillna("").astype(str).tolist()
            results = await asyncio.gather(
                *(_extract_one(client, sem, i, t) for i, t in enumerate(texts))
            )

            enriched = _attach_results(chunk, results)
            enriched.to_csv(output_csv, mode="a", index=False, header=not file_has_header)
            file_has_header = True

            newly_processed += len(enriched)
            failures.extend(r for r in results if not r["ok"])
            prompt_tokens += sum(r.get("prompt_tokens", 0) for r in results)
            completion_tokens += sum(r.get("completion_tokens", 0) for r in results)

            print(f"checkpoint {n}/{len(chunks)}: "
                  f"{newly_processed:,}/{len(todo):,} rows written to disk")
    finally:
        await client.close()

    out = pd.read_csv(output_csv)
    return {
        "status": "success",
        "rows_in": len(df),
        "rows_out": len(out),
        "newly_processed": newly_processed,
        "resumed_from": len(done),
        "failed": len(failures),
        "failure_samples": [f["error"][:160] for f in failures[:3]],
        "total_topics": int(out["all_topics_discussed"].apply(
            lambda x: len(x) if isinstance(x, list) else len(eval(x))).sum()),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "output_path": str(output_csv),
        "model_chain": MODEL_CHAIN,
        "model_usage": out["model_version"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(asyncio.run(extract_topics(limit=n)), indent=2))
