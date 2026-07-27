"""Turn a raw CFPB export into a clean, stratified working set for the pipeline.

Produces three files:
    data/input.csv        -> what the pipeline sees (NO ground-truth labels)
    data/ground_truth.csv -> Complaint ID -> Issue / Sub-issue, held out for eval
    data/eval_holdout.csv -> a smaller labelled slice for measuring extraction quality

Usage:
    python prepare_cfpb.py --raw "C:/Users/namit/Downloads/complaints-XXXX.csv" --out ./data
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# CFPB export column -> pipeline column
COLUMN_MAP = {
    "Date received": "date",
    "Product": "product",
    "Sub-product": "sub_product",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "State": "state",
    "Submitted via": "channel",
    "Complaint ID": "complaint_id",
}
GROUND_TRUTH_COLS = ["Issue", "Sub-issue"]

# CFPB renamed product categories several times; the same product appears under
# multiple strings depending on when the complaint was filed. Without this map,
# groupby() treats each historical label as a separate product and hands each
# its own quota -- so a category renamed twice gets double the sample weight.
CANONICAL_PRODUCT = {
    "Credit reporting": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Checking or savings account": "Checking or savings account",
    "Bank account or service": "Checking or savings account",
    "Credit card": "Credit card",
    "Credit card or prepaid card": "Credit card",
    "Prepaid card": "Prepaid card",
    "Payday loan": "Payday/title/personal loan",
    "Payday loan, title loan, or personal loan": "Payday/title/personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday/title/personal loan",
    "Vehicle loan or lease": "Vehicle or consumer loan",
    "Consumer Loan": "Vehicle or consumer loan",
    "Money transfer, virtual currency, or money service": "Money transfer or virtual currency",
    "Money transfers": "Money transfer or virtual currency",
    "Virtual currency": "Money transfer or virtual currency",
}

# Too few rows and too vague to form a meaningful theme cluster.
DROP_PRODUCTS = {"Other financial service", "Other", "Non-financial product/service"}

# Narratives are ~9% redaction tokens by volume. Left raw, these dominate the
# embedding space and wreck clustering, so they get collapsed to single markers.
CLEAN_RULES: list[tuple[str, str]] = [
    (r"\{\$[\d,]+\.?\d*\}", " [AMOUNT] "),          # {$9800.00}
    (r"\bXX/XX/(?:XXXX|\d{4})\b", " [DATE] "),      # XX/XX/XXXX
    (r"\bXX/XX\b", " [DATE] "),
    (r"\bX{2,}\b", " [REDACTED] "),                 # XXXX runs
    (r"(?:\[REDACTED\]\s*){2,}", " [REDACTED] "),   # collapse 40-in-a-row
    (r"\s+", " "),
]


def clean_narrative(text: str) -> str:
    for pattern, replacement in CLEAN_RULES:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def load_and_clean(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_path, low_memory=False)

    missing = [c for c in list(COLUMN_MAP) + GROUND_TRUTH_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"Export is missing expected columns: {missing}")

    df = df[list(COLUMN_MAP) + GROUND_TRUTH_COLS].rename(columns=COLUMN_MAP)
    before = len(df)

    df = df[df["narrative"].notna()].copy()

    # Nulls elsewhere in the bulk file coerce this column to float, which would
    # write ids as "10298545.0" and break the join to ground_truth.csv.
    df = df[df["complaint_id"].notna()]
    df["complaint_id"] = df["complaint_id"].astype("int64")

    # Collapse historical label variants before any grouping happens.
    raw_products = df["product"].nunique()
    df["product"] = df["product"].map(lambda p: CANONICAL_PRODUCT.get(p, p))
    df = df[~df["product"].isin(DROP_PRODUCTS)]
    print(f"products: {raw_products} raw labels -> {df['product'].nunique()} canonical")
    # CFPB mixes "2023-04-11" and "2023-04-11T09:07:47Z" in the same column.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True, format="mixed")
    df = df.dropna(subset=["date", "Issue"])

    df["narrative_clean"] = df["narrative"].map(clean_narrative)
    df["word_count"] = df["narrative_clean"].str.split().str.len()

    # Too short = nothing to extract. Absurdly long = usually a pasted dossier.
    df = df[(df["word_count"] >= 30) & (df["word_count"] <= 3000)]

    print(f"loaded {before} rows -> {len(df)} usable after cleaning")
    return df


def stratified_sample(df: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Cap each product so no single category dominates the clustering."""
    products = df["product"].nunique()
    per_product = max(target // products, 1)

    sampled = (
        df.groupby("product", group_keys=False)
          .apply(lambda g: g.sample(min(len(g), per_product), random_state=seed))
    )

    # Top up to target if smaller products were short.
    if len(sampled) < target:
        remainder = df.drop(sampled.index)
        extra = remainder.sample(min(len(remainder), target - len(sampled)), random_state=seed)
        sampled = pd.concat([sampled, extra])

    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def truncate(df: pd.DataFrame, max_words: int) -> pd.DataFrame:
    """People state their core problem up front. Truncating cuts token cost ~3x."""
    df = df.copy()
    df["narrative_clean"] = df["narrative_clean"].apply(
        lambda t: " ".join(t.split()[:max_words])
    )
    df["word_count"] = df["narrative_clean"].str.split().str.len()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path, help="raw CFPB export CSV")
    ap.add_argument("--out", default=Path("./data"), type=Path)
    ap.add_argument("--target", type=int, default=3000, help="working set size")
    ap.add_argument("--holdout", type=int, default=200, help="labelled eval slice")
    ap.add_argument("--max-words", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(args.raw)
    df = stratified_sample(df, args.target, args.seed)
    df = truncate(df, args.max_words)

    # Ground truth lives in its own file, keyed by complaint_id. The pipeline
    # never sees Issue/Sub-issue, so there is no way to leak it into a prompt.
    ground_truth = df[["complaint_id", "Issue", "Sub-issue"]].copy()
    ground_truth.to_csv(args.out / "ground_truth.csv", index=False)

    holdout = ground_truth.sample(min(args.holdout, len(df)), random_state=args.seed)
    holdout.to_csv(args.out / "eval_holdout.csv", index=False)

    pipeline_input = df[[
        "complaint_id", "date", "product", "sub_product",
        "company", "state", "channel", "narrative_clean", "word_count",
    ]].rename(columns={"narrative_clean": "narrative"})
    pipeline_input["date"] = pipeline_input["date"].dt.strftime("%Y-%m-%d")
    pipeline_input.to_csv(args.out / "input.csv", index=False)

    print(f"\nwrote {len(pipeline_input)} rows -> {args.out / 'input.csv'}")
    print(f"wrote ground truth      -> {args.out / 'ground_truth.csv'}")
    print(f"wrote eval holdout ({len(holdout)}) -> {args.out / 'eval_holdout.csv'}")

    print("\nproduct mix:")
    print(pipeline_input["product"].value_counts().to_string())
    print(f"\nmedian words after truncation: {int(pipeline_input['word_count'].median())}")
    print(f"distinct true issues: {ground_truth['Issue'].nunique()}")
    print(f"months covered: {pipeline_input['date'].str[:7].nunique()}")


if __name__ == "__main__":
    main()
