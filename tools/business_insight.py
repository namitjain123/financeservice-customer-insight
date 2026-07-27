"""Step 4 - turn the labelled, exploded data into executive-readable charts.

No LLM calls here - pure pandas + matplotlib over the output of Step 3.

Produces four plots:
  1) Top N Customer Pain Points       - which themes come up most often
  2) Pain Points by Product           - which products are driving which themes
  3) Theme Severity by Sentiment      - how negative each theme skews
  4) Theme Trends Over Time           - are the top themes growing or shrinking

Differences from the original survey pipeline this replaces:
  * column names match the CFPB schema (`date`, lowercase) instead of the
    original survey export's (`Date`)
  * chart 3 explicitly flags a real limitation of this dataset: every row
    here originates from a complaint, so sentiment skews overwhelmingly
    negative - the chart still shows *relative* severity across themes, but
    it will never show a theme that reads as mostly positive
  * returns real per-chart row counts instead of a fixed "for example" claim
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import settings

SENTIMENT_COLORS = {"Negative": "#E45756", "Neutral": "#F2CF5B", "Positive": "#72B7B2"}


def _top_pain_points(df: pd.DataFrame, top_n: int) -> tuple[pd.Series, str]:
    counts = df.groupby("general_topic_l1").size().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.iloc[::-1].plot(kind="barh", ax=ax, color="#4C78A8")
    ax.set_title(f"Top {top_n} Customer Pain Points")
    ax.set_xlabel("Mentions")
    ax.set_ylabel("Theme")
    fig.tight_layout()

    path = str(settings.ARTIFACTS / "plot_top_pain_points.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return counts, path


def _pain_points_by_product(df: pd.DataFrame) -> str:
    cross = df.pivot_table(
        index="general_topic_l1", columns=settings.PRODUCT_COLUMN,
        values="topic_discussed", aggfunc="count", fill_value=0,
    )

    fig, ax = plt.subplots(
        figsize=(max(8, 0.6 * (cross.shape[1] + 4)), max(6, 0.4 * (cross.shape[0] + 4)))
    )
    im = ax.imshow(cross.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(cross.shape[0]))
    ax.set_yticklabels(cross.index.tolist())
    ax.set_xticks(range(cross.shape[1]))
    ax.set_xticklabels(cross.columns.tolist(), rotation=45, ha="right")
    ax.set_title("Pain Points by Product (Counts)")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()

    path = str(settings.ARTIFACTS / "heatmap_pain_points_by_product.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _theme_severity(df: pd.DataFrame, top_themes: list[str]) -> str:
    """100% stacked bar of sentiment share per theme.

    Every row here is drawn from a complaint, so sentiment is structurally
    skewed negative (see the "Known limitations" note in the module
    docstring). The chart is still meaningful for RELATIVE comparison - a
    theme at 95% negative is more urgent than one at 60% - but do not expect
    or claim any theme will read as predominantly positive.
    """
    counts = (
        df.groupby(["general_topic_l1", "customer_sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0)
    )
    shares = counts.div(counts.sum(axis=1), axis=0).fillna(0)
    shares_top = shares.loc[[t for t in shares.index if t in top_themes]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(shares_top))
    for sentiment in ("Negative", "Neutral", "Positive"):
        values = shares_top[sentiment].values
        ax.bar(shares_top.index, values, bottom=bottom, label=sentiment,
               color=SENTIMENT_COLORS[sentiment])
        bottom += values

    ax.set_title("Theme Severity by Sentiment (Share)")
    ax.set_ylabel("Share of Mentions")
    ax.set_xlabel("Theme (Top)")
    ax.legend(title="Sentiment")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()

    path = str(settings.ARTIFACTS / "theme_severity_stacked.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _theme_trends(df: pd.DataFrame, top_themes: list[str]) -> str:
    dft = df.copy()
    dft[settings.DATE_COLUMN] = pd.to_datetime(dft[settings.DATE_COLUMN], errors="coerce")
    dft = dft.dropna(subset=[settings.DATE_COLUMN])
    dft["month"] = dft[settings.DATE_COLUMN].dt.to_period("M").dt.to_timestamp()

    trend = (
        dft[dft["general_topic_l1"].isin(top_themes)]
        .groupby(["month", "general_topic_l1"]).size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for theme in top_themes:
        sub = trend[trend["general_topic_l1"] == theme]
        ax.plot(sub["month"], sub["count"], marker="o", label=theme)

    ax.set_title("Theme Trends Over Time (Top Themes)")
    ax.set_ylabel("Mentions per Month")
    ax.set_xlabel("Month")
    ax.legend(title="Theme", bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()

    path = str(settings.ARTIFACTS / "theme_trends_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_insights(
    input_csv=settings.OUTPUT_CSV,
    top_n: int = settings.TOP_N_THEMES,
) -> dict[str, Any]:
    """Read the labelled, exploded CSV and produce all four charts."""
    df = pd.read_csv(input_csv)
    required = {"general_topic_l1", "topic_discussed", settings.PRODUCT_COLUMN,
                settings.DATE_COLUMN, "customer_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV is missing required columns: {missing}")

    top_counts, p1 = _top_pain_points(df, top_n)
    top_themes = top_counts.index.tolist()

    p2 = _pain_points_by_product(df)
    p3 = _theme_severity(df, top_themes)
    p4 = _theme_trends(df, top_themes)

    return {
        "status": "success",
        "input_rows": len(df),
        "distinct_themes": int(df["general_topic_l1"].nunique()),
        "top_n": top_n,
        "top_themes": top_counts.to_dict(),
        "plots": [p1, p2, p3, p4],
        "input_path": str(input_csv),
    }


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else settings.TOP_N_THEMES
    print(json.dumps(generate_insights(top_n=n), indent=2))
