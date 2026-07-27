"""Central configuration. Everything tunable lives here, not scattered in tools."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# --- provider ---------------------------------------------------------------
# Google Gemini, reached through its OpenAI-compatible endpoint. This means the
# standard `openai` package works unchanged - only the key, base URL and model
# names differ. Swapping providers later is a three-line change here, not a
# rewrite of every tool.
API_KEY = os.getenv("GEMINI_API_KEY")
BASE_URL = os.getenv(
    "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Verified working against a free-tier key on 2026-07-27.
#   gemini-3.5-flash-lite  <- default: works, cheap, highest free-tier quota
#   gemini-3.1-flash-lite     also verified working
#   gemini-flash-lite-latest  works, but hot-swaps versions (hurts reproducibility)
#   gemini-2.5-flash          404s for keys created after its deprecation
#   gemini-3.6-flash          returns empty content through the OpenAI-compat layer
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.5-flash-lite")

# Free-tier quotas are per-model and tight (reportedly ~15 RPM / ~1,000 RPD as
# of 2026-07 - Google doesn't publish exact figures without an AI Studio login).
# A 3,000-row run can exhaust CHAT_MODEL's daily quota partway through. Rather
# than add an external gateway (Portkey etc.) just to get "try model B if model
# A is exhausted", these are tried in order after CHAT_MODEL - all three
# verified against this key. No new dependency, no new account.
CHAT_MODEL_FALLBACKS = [
    m.strip()
    for m in os.getenv(
        "CHAT_MODEL_FALLBACKS", "gemini-3.1-flash-lite,gemini-flash-lite-latest"
    ).split(",")
    if m.strip()
]
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

# gemini-embedding-001 defaults to 3072 dimensions. 1536 clusters just as well
# for this many topic phrases and halves the memory and distance-computation cost.
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1536"))

# --- pipeline files ---------------------------------------------------------
INPUT_CSV = DATA / "input.csv"
GROUND_TRUTH_CSV = DATA / "ground_truth.csv"
EVAL_HOLDOUT_CSV = DATA / "eval_holdout.csv"

TOPICS_CSV = ARTIFACTS / "df_with_topics.csv"
CLUSTERS_CSV = ARTIFACTS / "df_with_clusters.csv"
OUTPUT_CSV = ARTIFACTS / "output.csv"
EMBEDDING_CACHE = ARTIFACTS / "topic_embeddings.npz"
RUN_MANIFEST = ARTIFACTS / "run_manifest.json"

# --- schema -----------------------------------------------------------------
# The CFPB export uses `narrative`; the original survey project used
# `call_transcrpt`. Pointing old code at this data fails on this column.
TEXT_COLUMN = "narrative"
ID_COLUMN = "complaint_id"
DATE_COLUMN = "date"
PRODUCT_COLUMN = "product"

# --- tunables ---------------------------------------------------------------
MAX_TOPICS_PER_ROW = 5
N_CLUSTERS = 12          # provisional; justify with eval/choose_k.py before trusting
CLUSTER_METRIC = "cosine"  # embeddings are unit-normalised
TOP_N_THEMES = 5
RANDOM_SEED = 42

# The Gemini free tier throttles hard (single-digit requests/minute on some
# models). Concurrency above the quota just generates 429s, so keep it low and
# let the retry logic absorb the rest. Raise this on a paid key.
EXTRACTION_CONCURRENCY = int(os.getenv("EXTRACTION_CONCURRENCY", "4"))
RETRIES_PER_MODEL = 2    # attempts on one model before falling through to the next
RETRY_BASE_DELAY = 5.0   # seconds; doubles each attempt


def require_api_key() -> str:
    """Fail loudly at startup rather than 3,000 rows into a run."""
    if not API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.\n"
            "Get one free at https://aistudio.google.com/apikey"
        )
    return API_KEY
